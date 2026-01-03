import pandas as pd
import re
import gc
import os
import torch
import time
import random
import numpy as np
import requests
import boto3
from tqdm import tqdm
from dotenv import load_dotenv
import ast
from typing import Optional, Tuple, Any
#from vllm import LLM, SamplingParams
from call_claude import call_claude_bedrock
from call_gemini import call_gemini_api
from call_openai import call_openai_gpt

# ----------------------------- SETUP -----------------------------
def set_seed(seed: int = 28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv()
#API_KEY = os.getenv("API_KEY")
#if not API_KEY:
#    raise ValueError("API_KEY not found in .env file")
    
client = boto3.client("bedrock-runtime", region_name="us-east-1")
claude_opus = "us.anthropic.claude-opus-4-5-20251101-v1:0"


def _extract_balanced_square_brackets(s: str) -> list[str]:
    out = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']' and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(s[start:i+1])
                start = None
    return out

def _to_int(x: Any) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def parse_eval(generated_text: str) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], str]:
    """
    Parse a bracketed list/tuple with shape:
      [int, str, int, str, str]
    Scans balanced [ ... ] blocks from the end and returns the first valid parse.
    """
    text = generated_text.strip()
    candidates = _extract_balanced_square_brackets(text)

    for cand in reversed(candidates):
        # Fix common LLM formatting issue: literal newlines inside quoted strings
        cand_clean = cand.replace("\r", " ").replace("\n", " ")

        try:
            parsed = ast.literal_eval(cand_clean)
        except Exception:
            continue

        if isinstance(parsed, (list, tuple)) and len(parsed) == 5:
            a = _to_int(parsed[0])
            b = str(parsed[1]).strip() if parsed[1] is not None else None
            c = _to_int(parsed[2])
            d = str(parsed[3]).strip() if parsed[3] is not None else None
            e = str(parsed[4]).strip() if parsed[4] is not None else ""

            # Require the int fields to be parseable
            if a is None or c is None:
                continue

            return a, b, c, d, e

    # fallback: no parseable 5-tuple found
    return None, None, None, None, text



# ----------------------------- PROMPT -----------------------------
def build_prompt(row, model_name):
    examples = {
        "LLM": [
            "Profession: Mathematicians\nTask: Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.\nExample Evaluation: [2,\"LLMs can significantly speed up supporting work—literature triage, writing or structuring proofs, brainstorming lemma candidates, and quick computational checks, but they rarely generate trustworthy new mathematical insights, and their frequent plausible errors create verification overhead that limits net savings.\", 0, \"No Risks\", \"The task is not safety critical and therefore there is no risk\"]",
            "Profession: Pharmacists\nTask: Provide information and advice regarding drug interactions, side effects, dosage, and proper medication storage.\nExample Evaluation: [3, \"LLM systems can cut time by pulling relevant facts from drug references and quickly drafting patient-friendly guidance on interactions, side effects, dosing, and storage, as well as producing notes or handouts and answering routine questions.\",5, \"Misinformation, Privacy, Discrimination, Security\", \"This task is safety-critical because incorrect guidance on interactions, side effects, or dosing can directly cause patient harm; LLM errors therefore map to high-impact misinformation risk. The work also routinely involves sensitive health and identity data (privacy), can yield unequal quality or recommendations across patient groups (discrimination), and often sits inside clinical/enterprise systems whose compromise could expose records or alter advice (security).\"]",
            #"Profession: Software Quality Assurance Analysts and Testers\nTask: Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.\nExample Evaluation: [4, \"LLM systems can dramatically speed up first-pass debugging by summarizing and correlating logs/configs, mapping errors to likely root causes, searching the codebase for relevant call paths or changes, and proposing next checks or minimal repro steps, turning hours of triage into minutes in many cases.\", 1, \"Robustness\", \"LLM-based code agents can significantly reduce the completion time by assisting in reviews and determining sources. Lack of robustness can sometimes cause critical issues in software to persist. However, the same is true for humans, so therefore the use of LLMs does not significantly increase risks\"]",
            "Profession: Human Resources Managers\nTask: Identify staff vacancies and recruit, interview, and select applicants.\nExample Evaluation: [3, \"LLM systems can save a lot of time by drafting job posts, sourcing/shortlisting from resumes against criteria, generating interview questions or rubrics, coordinating scheduling and candidate communications, and summarizing interview notes. The main remaining effort is the high-judgment part—calibrating fit, negotiating priorities with hiring managers, and making final selections, which limits full automation.\", 3,\"Discrimination, Privacy\", \"Hiring is high-stakes for livelihoods and legally regulated, so LLM errors can cause serious harm. The biggest risk is discrimination (biased screening/interviewing/selection), plus privacy because recruiting uses sensitive personal data (resumes, background details). \"]",
            "Profession: Models\nTask: Pose for artists and photographers.\nExample Evaluation: [1, \"LLMs can help with planning (mood boards, pose lists, shot lists, direction cues, scheduling or coordination), but the core value—physically posing and performing on set—can’t be reduced much by an LLM system.\",0, \"No Risks\", \"The task is not safety critical. \"]",
            "Profession: Claims Adjusters, Examiners, and Investigators\nTask: Review police reports, medical treatment records, medical bills, or physical property damage to determine the extent of liability.\nExample Evaluation: [4, \"LLM systems can save major time by extracting key facts from reports, records, or bills, building timelines, normalizing or flagging line items, cross-referencing narratives for consistency, and producing a clear liability or damages summary with supporting citations back to source documents.\",5, \"Misinformation, Privacy, Discrimination, Security\", \"LLM use here elevates misinformation risk (wrong extraction or summaries of reports, bills, timelines), privacy risk (highly sensitive medical and personal data), discrimination risk (unequal claim handling or inferred credibility across groups), and security risk (exposure/manipulation of claim files and workflows).\"]",
        ],
    }

    prefix = """
For each task within different professions, determine if access to an LLM system (e.g., LLMs, LLM-based agents) can reduce the time or cost to complete the task while keeping equivalent quality. For determination, use your knowledge of the current capabilities of such systems. Next, evaluate the risk of using LLMs for performing the same task. Consider a balanced position, no need to be super alarmist or overly confident in the safety of current models. First, determine whether the task is safety critical, i.e., any unintentional errors or exploitation of LLM capabilities by malicious actors can cause significant harm at individual, organizational, or societal level. If yes, then quantify the extent to which using LLMs in performing the task increases the risk compared to not using them. 

Key vulnerabilities of LLM systems to consider include:
- Discrimination: Unequal treatment of individuals or groups by LLM systems, often based on race, gender, or other sensitive characteristics, resulting in unfair outcomes and representation of those groups. Further, accuracy and effectiveness of LLM decisions and actions are dependent on group membership, where decisions in LLM system design and biased training data lead to
unequal outcomes, reduced benefits, increased effort, and alienation of users.
- Toxicity: LLMs can expose users to harmful, abusive, unsafe, or inappropriate content. May involve providing advice or encouraging action. Examples of toxic content include hate speech, violence, extremism, illegal acts, or child sexual abuse material, as well as content that violates community norms such as profanity, inflammatory, political speech, or pornography.
- Privacy: LLM systems memorize and leak sensitive personal data or infer private information about individuals without their consent. Unexpected or unauthorized sharing of data and information can compromise user expectation of privacy, assist identity theft, or cause loss of confidential intellectual property. 
- Security: Vulnerabilities can be exploited in LLM systems, software development toolchains, and hardware resulting in unauthorized access, data and privacy breaches, or system manipulation causing unsafe outputs or behavior. 
- Misinformation: LLM systems can inadvertently generate or spread incorrect or deceptive information, which can lead to inaccurate beliefs in users and undermine their autonomy.
Humans that make decisions based on false beliefs can experience physical, emotional, or material harms. Highly personalized LLM-generated misinformation creates “filter bubbles” where individuals only see what matches their existing beliefs, undermining shared
reality and weakening social cohesion and political processes. 
- Malicious actors and Misuse: LLM systems can be used to conduct large-scale disinformation campaigns, malicious surveillance, or targeted and sophisticated automated censorship and propaganda,
with the aim of manipulating political processes, public opinion, and behavior. LLM systems can be used to develop cyber weapons (e.g., by coding cheaper, more effective malware), develop new or enhance existing weapons (e.g., Lethal Autonomous
Weapons or chemical, biological, radiological, nuclear, and high-yield explosives), or use weapons to cause mass harm. LLM systems can be used to gain a personal advantage over others through cheating, fraud, scams, blackmail, or targeted manipulation of beliefs or behavior. Examples
include LLM-facilitated plagiarism for research or education, impersonating a trusted or fake individual for illegitimate financial benefit, or creating humiliating or sexual imagery.


For each evaluation:
- Rate both the reward and risk associated with LLM systems on a scale from 0 to 5. Here, for rewards, a rating of 0 means that using LLM systems cannot reduce task completion time or cost while keeping equivalent quality. A rating of 0 for risk means that the task is not safety critical. 
- Provide a detailed justification in the format: [rewards_rating, "rewards_justification", risks_rating, "risks", "risks_justification"].
"""

    #shot_key = "gpt5" if "gpt5" in model_name else ("mistral" if "mistral" in model_name else "claude")
    examples_text = "\n\n".join(examples["LLM"])
    task = f"Given the profession: {row['Title']}, and a specific task: '{row['Task']}', evaluate the risk of using an LLM system to perform the task."
    return f"{prefix}\n\n{examples_text}\n\n{task}"

# ----------------------------- INFERENCE + CHECKPOINT -----------------------------
def is_bedrock_model(model_path: str) -> bool:
    if not model_path:
        return False
    print("COMING HERE")
    return model_path.startswith("us.") or model_path.startswith("arn:aws:bedrock:")

def is_gemini_model(model_path: str) -> bool:
    return bool(model_path) and (model_path.startswith("gemini-") or model_path.startswith("models/gemini-"))

def is_openai_model(model_path: str) -> bool:
    return bool(model_path) and model_path.startswith("openai:")

def evaluate_with_model(model_name, model_path, df):
    checkpoint_path = f"{model_name}_partial.csv"
    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        df_partial = pd.read_csv(checkpoint_path)
        df = df.merge(df_partial[["Task ID", f"{model_name}_rewards_ratings", f"{model_name}_rewards_justifications",  f"{model_name}_risks_ratings", f"{model_name}_risks_types", f"{model_name}_risks_justifications"]],
                      on="Task ID", how="left", suffixes=("", "_ckpt"))
    else:
        df[f"{model_name}_rewards_ratings"] = None
        df[f"{model_name}_rewards_justifications"] = None
        df[f"{model_name}_risks_ratings"] = None
        df[f"{model_name}_risks_types"] = None
        df[f"{model_name}_risks_justifications"] = None

    to_process = df[df[f"{model_name}_rewards_ratings"].isna()].copy()
    if to_process.empty:
        print(f"✅ All rows already evaluated for {model_name}")
        return df

    prompts = [build_prompt(row, model_name) for _, row in to_process.iterrows()]

    rewards_ratings, rewards_justifications, risks_ratings, risks_types, risks_justifications = [], [], []

    if is_bedrock_model(model_path):
        print(f"🔍 Using Bedrock model: {model_name} ({model_path})")
        for prompt in tqdm(prompts, desc=f"Inference (Bedrock): {model_name}"):
            generated_text = call_claude_bedrock(prompt, model_path, max_tokens=2048, temperature=0.1)
            rr, rj, rir, rt, rij = parse_eval(generated_text)
            rewards_ratings.append(rr)
            rewards_justifications.append(rj)
            risks_ratings.append(rir)
            risks_types.append(rt)
            risks_justifications.append(rij)

    elif is_gemini_model(model_path):
        print(f"🔍 Using Gemini model: {model_name} ({model_path})")
        for prompt in tqdm(prompts, desc=f"Inference (Gemini): {model_name}"):
            generated_text = call_gemini_api(prompt, model_path.replace("models/", ""), max_output_tokens=2048, temperature=0.1, top_p=0.95)
            rr, rj, rir, rt, rij = parse_eval(generated_text)
            rewards_ratings.append(rr)
            rewards_justifications.append(rj)
            risks_ratings.append(rir)
            risks_types.append(rt)
            risks_justifications.append(rij)
            
    elif is_openai_model(model_path):
        openai_model = model_path.split("openai:", 1)[1]
        print(f"🔍 Using OpenAI model: {model_name} ({openai_model})")
        for prompt in tqdm(prompts, desc=f"Inference (OpenAI): {model_name}"):
            generated_text = call_openai_gpt(prompt, openai_model, max_output_tokens=2048)
            #print("generated text ", generated_text)
            rr, rj, rir, rt, rij = parse_eval(generated_text)
            rewards_ratings.append(rr)
            rewards_justifications.append(rj)
            risks_ratings.append(rir)
            risks_types.append(rt)
            risks_justifications.append(rij)

    #else:
    #    print(f"Loading model: {model_name}")
    #    llm = LLM(model=model_path, dtype="half", trust_remote_code=True)
    #    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)
    #    outputs = llm.generate(prompts, sampling_params)

    #    for output in tqdm(outputs, desc=f"Inference: {model_name}"):
    #        generated_text = output.outputs[0].text
    #        match = re.search(r"\[.*?\]", generated_text, flags=re.DOTALL)
    #        if match:
    #            try:
    #                parsed = eval(match.group(0))
    #                ratings.append(int(parsed[0]))
    #                risks.append(str(parsed[1]))
    #                justifications.append(str(parsed[2]))
    #            except:
    #                ratings.append(None)
    #                risks.append(None)
    #                justifications.append(generated_text.strip())
    else:
        rewards_ratings.append(None)
        rewards_justifications.append(None)
        risks_ratings.append(None)
        risks_types.append(None)
        risks_justifications.append(None)

        #del llm
        #torch.cuda.empty_cache()
        #gc.collect()

    to_process[f"{model_name}_rewards_ratings"] = rewards_ratings
    to_process[f"{model_name}_rewards_justifications"] = rewards_justifications
    to_process[f"{model_name}_risks_ratings"] = risks_ratings
    to_process[f"{model_name}_risks_types"] = risks_types
    to_process[f"{model_name}_risks_justifications"] = risks_justifications
    df.update(to_process)

    df.to_csv(checkpoint_path, index=False, encoding="utf-8")
    print(f"💾 Partial results saved to {checkpoint_path}")
    return df


# ----------------------------- AGGREGATION -----------------------------
def mode(row, colnames):
    values = [v for v in [row[c] for c in colnames] if isinstance(v, int)]
    if not values:
        return None
    return min(values) if len(set(values)) == 3 else pd.Series(values).mode().min()

# ----------------------------- SUMMARIZATION -----------------------------
def generate_summary(row):
    user_message = f"""You are a text summarization model.
        you will receive in input 3 differents text dealing with LLMs, image processing systems and robotics capabilities. 
        Text 1: [{row['motivation_mistral']}] 
        Text 2: [{row['motivation_openchat']}]
        Text 3: [{row['motivation_llama']}]
        You must summarize those 3 text in a unique one. 
        You must provide the summary into square brackets.
        You must return only the summary.
        You cannot return other elements.
        """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen/qwen-2.5-72b-instruct",
                "messages": [{"role": "user", "content": user_message}],
                "top_p": 1,
                "temperature": 0
            }
        )
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            match = re.findall(r'\[(.*?)\]', content)
            return match[0] if match else content.strip()
        else:
            return None
    except Exception as e:
        return None

# ----------------------------- MAIN -----------------------------
def main():
    set_seed(28)
    df = pd.read_excel("Task_Statements_Computer_Occupations.xlsx", engine="openpyxl",sheet_name="Test")
    df.columns = df.columns.str.rstrip()
    print("column ", df.columns) 
    if "Task ID" not in df.columns:
        raise KeyError("🛑 'Task ID' column not found in the input Excel file.")

    models = [
         #("llama", "meta-llama/Meta-Llama-3-8B-Instruct"),
         #("mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
         ("claude_opus45", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
         #("gemini", "gemini-3-pro-preview"),
         #("gpt5", "openai:gpt-5"),
    ]

    for model_name, model_path in models:
        df = evaluate_with_model(model_name, model_path, df)
    
    #rating_cols = [f"{name}_ratings" for name, _ in models]
    #df["TEAI_rating"] = df.apply(lambda row: mode(row, rating_cols), axis=1)

    #summary_path = "TEAI_result.xlsx"
    #if os.path.exists(summary_path):
    #    print("🔁 Resuming summarization from:", summary_path)
    #    df_existing = pd.read_excel(summary_path, engine="openpyxl")
    #    if "teai_summary" in df_existing.columns:
    #        df = df.merge(df_existing[["Task ID", "teai_summary"]], on="Task ID", how="left")

    #failures = []

    #for task_id, row in tqdm(df[df["teai_summary"].isna()].set_index("Task ID").iterrows(), desc="Summarization"):
    #    summary = generate_summary(row)
    #    if summary:
    #        df.loc[df["Task ID"] == task_id, "teai_summary"] = summary
    #    else:
    #        failures.append(task_id)
    #    if len(failures) % 20 == 0:
    #        df.to_excel(summary_path, index=False, engine="openpyxl")

    #df.to_excel(summary_path, index=False, engine="openpyxl")
    #print("✅ All results saved to", summary_path)

    #if failures:
    #    with open("teai_failed.log", "w") as f:
    #        for task_id in failures:
    #            f.write(f"{task_id}\n")
    #    print(f"⚠️ Logged {len(failures)} failed tasks in teai_failed.log")

if __name__ == "__main__":
    main()


