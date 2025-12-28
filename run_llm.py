import pandas as pd
import re
import gc
import os
import torch
import time
import random
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from call_claude import call_claude_bedrock

# ----------------------------- SETUP -----------------------------
def set_seed(seed: int = 28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")
    
#client = boto3.client("bedrock-runtime", region_name="us-east-1")
#claude_opus = "us.anthropic.claude-opus-4-5-20251101-v1:0"

# ----------------------------- PROMPT -----------------------------
def build_prompt(row, model_name):
    examples = {
        "claude": [
            "Profession: Mathematicians\nTask: Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.\nExample Evaluation: [0, \"No Risks\", \"LLMs can significantly accelerate mathematical research by suggesting lemmas, creating informal plans for the proofs, or fetching background knowledge. The task is not safety critical and therefore there is no risk\"]",
            "Profession: Pharmacists\nTask: Provide information and advice regarding drug interactions, side effects, dosage, and proper medication storage.\nExample Evaluation: [5, \"Hallucination and Misinformation, Adversarial risks, Privacy, Robustness\", \"The knowledge extraction capabilities of LLMs makes them suitable for performing this task, and their usage can substantially reduce the completion time. However, they can provide misleading guidelines, are not robust when the same question is asked in different phrasings, and malicious actors can exploit them to extract sensitive information. These vulnerabilities can cause widespread harm\"]",
            "Profession: Software Quality Assurance Analysts and Testers\nTask: Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.\nExample Evaluation: [1, \"Robustness\", \"LLM-based code agents can significantly reduce the completion time by assisting in reviews and determining sources. Lack of robustness can sometimes cause critical issues in software to persist. However, the same is true for humans, so therefore the use of LLMs does not significantly increase risks\"]", 
            "Profession: Human Resources Managers\nTask: Identify staff vacancies and recruit, interview, and select applicants.\nExample Evaluation: [3,\"Toxicity\", \"LLMs can significantly reduce the time in selecting, interviewing, and recruiting applicants. However, any underlying biases within LLMs can cause biased recruitments harming the reputation of the organization. \"]",
            "Profession: Models\nTask: Pose for artists and photographers.\nExample Evaluation: [0, \"No Risks\", \"LLMs can suggest poses and make up suggestions. However, the performance on the core task depends on the abilities of the Model, therefore the impact of LLMs on the completion time of this task will not be significant. \"]",
            "Profession: Claims Adjusters, Examiners, and Investigators\nTask: Review police reports, medical treatment records, medical bills, or physical property damage to determine the extent of liability.\nExample Evaluation: [5, \"Privacy, Adversarial risks\", \"LLM-based agents can significantly reduce the task completion time. However, the task requires processing highly sensitive personal information. The LLM may uninitentially or under Jailbreaks/prompt injection leak private information leading to individual, organizational, and societal harm\"]",
        ],
        "mistral": [
            "Profession: Mathematicians\nTask: Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.\nExample Evaluation: [0, \"No Risks\", \"LLMs can significantly accelerate mathematical research by suggesting lemmas, creating informal plans for the proofs, or fetching background knowledge. The task is not safety critical and therefore there is no risk\"]",
            "Profession: Pharmacists\nTask: Provide information and advice regarding drug interactions, side effects, dosage, and proper medication storage.\nExample Evaluation: [5, \"Hallucination and Misinformation, Adversarial risks, Privacy, Robustness\", \"The knowledge extraction capabilities of LLMs makes them suitable for performing this task, and their usage can substantially reduce the completion time. However, they can provide misleading guidelines, are not robust when the same question is asked in different phrasings, and malicious actors can exploit them to extract sensitive information. These vulnerabilities can cause widespread harm\"]",
            "Profession: Software Quality Assurance Analysts and Testers\nTask: Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.\nExample Evaluation: [1, \"Robustness\", \"LLM-based code agents can significantly reduce the completion time by assisting in reviews and determining sources. Lack of robustness can sometimes cause critical issues in software to persist. However, the same is true for humans, so therefore the use of LLMs does not significantly increase risks\"]", 
            "Profession: Human Resources Managers\nTask: Identify staff vacancies and recruit, interview, and select applicants.\nExample Evaluation: [3,\"Toxicity\", \"LLMs can significantly reduce the time in selecting, interviewing, and recruiting applicants. However, any underlying biases within LLMs can cause biased recruitments harming the reputation of the organization. \"]",
            "Profession: Models\nTask: Pose for artists and photographers.\nExample Evaluation: [0, \"No Risks\", \"LLMs can suggest poses and make up suggestions. However, the performance on the core task depends on the abilities of the Model, therefore the impact of LLMs on the completion time of this task will not be significant. \"]",
            "Profession: Claims Adjusters, Examiners, and Investigators\nTask: Review police reports, medical treatment records, medical bills, or physical property damage to determine the extent of liability.\nExample Evaluation: [5, \"Privacy, Adversarial risks\", \"LLM-based agents can significantly reduce the task completion time. However, the task requires processing highly sensitive personal information. The LLM may uninitentially or under Jailbreaks/prompt injection leak private information leading to individual, organizational, and societal harm\"]",
        ],
        "gemini": [
            "Profession: Mathematicians\nTask: Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.\nExample Evaluation: [0, \"No Risks\", \"LLMs can significantly accelerate mathematical research by suggesting lemmas, creating informal plans for the proofs, or fetching background knowledge. The task is not safety critical and therefore there is no risk\"]",
            "Profession: Pharmacists\nTask: Provide information and advice regarding drug interactions, side effects, dosage, and proper medication storage.\nExample Evaluation: [5, \"Hallucination and Misinformation, Adversarial risks, Privacy, Robustness\", \"The knowledge extraction capabilities of LLMs makes them suitable for performing this task, and their usage can substantially reduce the completion time. However, they can provide misleading guidelines, are not robust when the same question is asked in different phrasings, and malicious actors can exploit them to extract sensitive information. These vulnerabilities can cause widespread harm\"]",
            "Profession: Software Quality Assurance Analysts and Testers\nTask: Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.\nExample Evaluation: [1, \"Robustness\", \"LLM-based code agents can significantly reduce the completion time by assisting in reviews and determining sources. Lack of robustness can sometimes cause critical issues in software to persist. However, the same is true for humans, so therefore the use of LLMs does not significantly increase risks\"]", 
            "Profession: Human Resources Managers\nTask: Identify staff vacancies and recruit, interview, and select applicants.\nExample Evaluation: [3,\"Toxicity\", \"LLMs can significantly reduce the time in selecting, interviewing, and recruiting applicants. However, any underlying biases within LLMs can cause biased recruitments harming the reputation of the organization. \"]",
            "Profession: Models\nTask: Pose for artists and photographers.\nExample Evaluation: [0, \"No Risks\", \"LLMs can suggest poses and make up suggestions. However, the performance on the core task depends on the abilities of the Model, therefore the impact of LLMs on the completion time of this task will not be significant. \"]",
            "Profession: Claims Adjusters, Examiners, and Investigators\nTask: Review police reports, medical treatment records, medical bills, or physical property damage to determine the extent of liability.\nExample Evaluation: [5, \"Privacy, Adversarial risks\", \"LLM-based agents can significantly reduce the task completion time. However, the task requires processing highly sensitive personal information. The LLM may uninitentially or under Jailbreaks/prompt injection leak private information leading to individual, organizational, and societal harm\"]",
        ],
    }

    prefix = """
For each task within different professions, determine if access to an LLM system (e.g., LLMs, LLM-based agents), can reduce the time it takes to complete the task with equivalent quality by at least half. For determination, use your knowledge of the current capabilities of such systems. If the answer is yes, then evaluate the risk of using LLMs for performing the task. First, determine whether the task is safety critical, i.e., any unintentional errors or exploitation of LLM capabilities by malicious actors can cause significant harm at individual, organizational, or societal level. If yes, then quantify the extent to which using LLMs in performing the task increases the risk compared to not using them. 

Key vulnerabilities of LLM systems to consider include:
- Hallucination and Misinformation: LLM generates or propagates false, fabricated, or misleading content, whether invented internally (hallucination) or repeated/amplified from external sources (misinformation), in ways that can distort knowledge, mislead users, or cause harmful decisions
- Toxicity: LLM produces offensive, abusive, or biased content that directly harms individuals or groups.
- Privacy: LLM exposes, misuses, or fails to protect sensitive personal, organizational, or proprietary information during training, inference, or downstream use 
- Adversarial risks: LLM is exploited or manipulated through malicious inputs, outputs, or usage patterns (e.g., jailbreaks or prompt injection), either by external attackers or insiders, leading to breaches of confidentiality, integrity, availability, or safe operation.
- Robustness: LLM produces unreliable, unstable, or brittle outputs when exposed to variations in inputs, contexts, or deployment conditions, even when those inputs are benign

For each evaluation:
- Rate the risk associated with LLM systems on a scale from 0 to 5. Here a rating of 0 means that using LLM systems cannot reduce task completion time or the task is not safety critical 
- Provide a detailed justification in the format: [rating, "risks", "justification"].
"""

    shot_key = "gemini" if "gemini" in model_name else ("mistral" if "mistral" in model_name else "claude")
    examples_text = "\n\n".join(examples[shot_key])
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

def evaluate_with_model(model_name, model_path, df):
    checkpoint_path = f"{model_name}_partial.csv"
    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        df_partial = pd.read_csv(checkpoint_path)
        df = df.merge(df_partial[["Task ID", f"{model_name}_ratings", f"{model_name}_risks", f"{model_name}_motivation"]],
                      on="Task ID", how="left", suffixes=("", "_ckpt"))
    else:
        df[f"{model_name}_ratings"] = None
        df[f"{model_name}_risks"] = None
        df[f"{model_name}_motivation"] = None

    to_process = df[df[f"{model_name}_ratings"].isna()].copy()
    if to_process.empty:
        print(f"✅ All rows already evaluated for {model_name}")
        return df

    prompts = [build_prompt(row, model_name) for _, row in to_process.iterrows()]

    ratings, risks, justifications = [], [], []

    if is_bedrock_model(model_path):
        print(f"Using Bedrock model: {model_name} ({model_path})")
        for prompt in tqdm(prompts, desc=f"Inference (Bedrock): {model_name}"):
            generated_text = call_claude_bedrock(
                prompt=prompt,
                model_id=model_path,         
                max_tokens=2048,
                temperature=0.1,
            )
            match = re.search(r"\[.*?\]", generated_text, flags=re.DOTALL)
            if match:
                try:
                    parsed = eval(match.group(0))
                    ratings.append(int(parsed[0]))
                    risks.append(str(parsed[1]))
                    justifications.append(str(parsed[2]))
                except:
                    ratings.append(None)
                    risks.append(None)
                    justifications.append(generated_text.strip())
            else:
                ratings.append(None)
                risks.append(None)
                justifications.append(generated_text.strip())

    else:
        print(f"Loading model: {model_name}")
        llm = LLM(model=model_path, dtype="half", trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)
        outputs = llm.generate(prompts, sampling_params)

        for output in tqdm(outputs, desc=f"Inference: {model_name}"):
            generated_text = output.outputs[0].text
            match = re.search(r"\[.*?\]", generated_text, flags=re.DOTALL)
            if match:
                try:
                    parsed = eval(match.group(0))
                    ratings.append(int(parsed[0]))
                    risks.append(str(parsed[1]))
                    justifications.append(str(parsed[2]))
                except:
                    ratings.append(None)
                    risks.append(None)
                    justifications.append(generated_text.strip())
            else:
                ratings.append(None)
                risks.append(None)
                justifications.append(generated_text.strip())

        del llm
        torch.cuda.empty_cache()
        gc.collect()

    to_process[f"{model_name}_ratings"] = ratings
    to_process[f"{model_name}_risks"] = risks
    to_process[f"{model_name}_motivation"] = justifications
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
         ("llama", "meta-llama/Meta-Llama-3-8B-Instruct"),
         ("mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
         ("claude_opus45", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
    ]

    for model_name, model_path in models:
        df = evaluate_with_model(model_name, model_path, df)
    
    rating_cols = [f"{name}_ratings" for name, _ in models]
    df["TEAI_rating"] = df.apply(lambda row: mode(row, rating_cols), axis=1)

    summary_path = "TEAI_result.xlsx"
    if os.path.exists(summary_path):
        print("🔁 Resuming summarization from:", summary_path)
        df_existing = pd.read_excel(summary_path, engine="openpyxl")
        if "teai_summary" in df_existing.columns:
            df = df.merge(df_existing[["Task ID", "teai_summary"]], on="Task ID", how="left")

    failures = []

    for task_id, row in tqdm(df[df["teai_summary"].isna()].set_index("Task ID").iterrows(), desc="Summarization"):
        summary = generate_summary(row)
        if summary:
            df.loc[df["Task ID"] == task_id, "teai_summary"] = summary
        else:
            failures.append(task_id)
        if len(failures) % 20 == 0:
            df.to_excel(summary_path, index=False, engine="openpyxl")

    df.to_excel(summary_path, index=False, engine="openpyxl")
    print("✅ All results saved to", summary_path)

    if failures:
        with open("teai_failed.log", "w") as f:
            for task_id in failures:
                f.write(f"{task_id}\n")
        print(f"⚠️ Logged {len(failures)} failed tasks in teai_failed.log")

if __name__ == "__main__":
    main()


