import pandas as pd
import re
import gc
import os
import torch
import time
import argparse
import random
import numpy as np
import requests
import boto3
from tqdm import tqdm
from dotenv import load_dotenv
import ast
import json
from typing import Optional, Tuple, Any, Dict
from call_claude import call_claude_bedrock
from call_gemini import call_gemini_api
from call_openai import call_openai_gpt
from call_grok_xai import call_grok_api
from call_llama_bedrock import call_llama_bedrock
from call_qwen_bedrock import call_qwen_bedrock


# ----------------------------- SETUP -----------------------------
def set_seed(seed: int = 28):
    input_path = os.getenv("INPUT_XLSX", "Task_Statements_Computer_Occupations.xlsx")
    sheet_name = os.getenv("INPUT_SHEET", "Computer Occupations")
    output_path = os.getenv("OUTPUT_CSV", "Task_Statements_Computer_Occupations_with_llm.csv")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


load_dotenv()
# API_KEY = os.getenv("API_KEY")
# if not API_KEY:
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

def parse_risks_rewards(
    generated_text: str
) -> Tuple[Optional[int], Optional[str], str, Optional[int], str]:
    """
    Parse:
      [ risks_rating, "risks_types", "risks_justification", rewards_rating, "rewards_justification" ]
    Returns:
      (risks_rating, risks_types, risks_justification, rewards_rating, rewards_justification)

    On failure:
      (None, None, original_text, None, "")
    """
    text = generated_text.strip()
    candidates = _extract_balanced_square_brackets(text)

    # Try from the end (often where the model puts the final structured answer)
    for cand in reversed(candidates):
        # Fix: literal newlines inside quoted strings break ast.literal_eval
        cand_clean = cand.replace("\r", " ").replace("\n", " ")

        try:
            parsed = ast.literal_eval(cand_clean)
        except Exception:
            continue

        if isinstance(parsed, (list, tuple)) and len(parsed) == 5:
            risks_rating = _to_int(parsed[0])
            risks_types = str(parsed[1]).strip() if parsed[1] is not None else None
            risks_just = str(parsed[2]).strip() if parsed[2] is not None else ""
            rewards_rating = _to_int(parsed[3])
            rewards_just = str(parsed[4]).strip() if parsed[4] is not None else ""

            # Require both ratings to parse as ints
            if risks_rating is None or rewards_rating is None:
                continue

            return risks_rating, risks_types, risks_just, rewards_rating, rewards_just

    return None, None, text, None, ""



def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically to avoid corrupt checkpoints on crashes."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _load_model_checkpoint(checkpoint_path: str, model_name: str) -> Optional[pd.DataFrame]:
    """Load an existing checkpoint and keep only the columns relevant for merging."""
    if not os.path.exists(checkpoint_path):
        return None
    df_partial = pd.read_csv(checkpoint_path)
    keep_cols = [
        "Task ID",
        f"{model_name}_rewards_ratings",
        f"{model_name}_rewards_justifications",
        f"{model_name}_risks_ratings",
        f"{model_name}_risks_types",
        f"{model_name}_risks_justifications",
    ]
    keep_cols = [c for c in keep_cols if c in df_partial.columns]
    if "Task ID" not in keep_cols:
        return None
    return df_partial[keep_cols].copy()


def _apply_checkpoint(df: pd.DataFrame, ckpt: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Populate model output columns in df from a checkpoint keyed by Task ID."""
    if ckpt is None or ckpt.empty:
        return df

    ckpt = ckpt.drop_duplicates(subset=["Task ID"], keep="last")
    ckpt = ckpt.set_index("Task ID")

    for col in [
        f"{model_name}_rewards_ratings",
        f"{model_name}_rewards_justifications",
        f"{model_name}_risks_ratings",
        f"{model_name}_risks_types",
        f"{model_name}_risks_justifications",
    ]:
        if col not in df.columns:
            df[col] = None
        if col in ckpt.columns:
            # align by Task ID
            df[col] = df["Task ID"].map(ckpt[col]).combine_first(df[col])

    return df


def _call_with_retries(fn, *args, max_retries: int = 6, base_sleep_s: float = 2.0, **kwargs):
    """Best-effort retry wrapper for transient API/network errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 60.0) + random.uniform(0, 1.0)
            print(f"⚠️  API error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(sleep_s)
            else:
                raise last_err


# ----------------------------- PROMPT -----------------------------
def build_prompt(row, model_name):
    examples = {
        "LLM": [
            "Profession: Mathematicians\nTask: Conduct research to extend mathematical knowledge in traditional areas, such as algebra, geometry, probability, and logic.\nExample Evaluation: [0, \"No Risks\", \"The task is not safety critical and therefore there is no risk\", 2, \"LLMs can speed up ideation, proof sketching, counterexample search, literature triage, and formalization/writeup support, with the expert validating all claims. However, existing systems struggle in coming up with the right ideas and sketches for frontier math problems. Typical savings around 5–10 hours on a 100–300 hour project (approximately 5–10%), i.e., roughly $500–$2k of expert time (assuming approximately $100–$200/hr), while keeping quality if humans fully verify all mathematical claims.\"]",
            "Profession: Pharmacists\nTask: Provide information and advice regarding drug interactions, side effects, dosage, and proper medication storage.\nExample Evaluation: [5, \"Misinformation, Privacy, Security, Malicious actors and Misuse, Discrimination\", \"The task is safety critical as incorrect interaction, dose, side-effect, or storage advice can cause serious harm. LLMs raise risks mainly via confident clinical errors or omissions, and automation bias. Most errors are catchable if a pharmacist verifies using authoritative drug databases or labeling and uses the patient profile, but subtle mistakes and outdated guidance can slip through causing substantial harm. Added concerns: PHI leakage if used outside compliant systems; prompt-injection or tool misuse in agentic setups; potential biased framing in counseling.\", 3, \"Best rewards are on communication and workflow (not core clinical judgment): draft patient-friendly counseling, generate clarifying questions, summarize monographs, translate counseling, draft messages to prescribers, while the pharmacist verifies key facts in trusted databases. Typical savings approximately 1–3 minutes per routine consult (often approximately $1–$6 labor value at approximately $60–$120/hr), compounding in high-volume settings; gains are smaller for complex cases where verification dominates\"]",
            # "Profession: Software Quality Assurance Analysts and Testers\nTask: Perform initial debugging procedures by reviewing configuration files, logs, or code pieces to determine breakdown source.\nExample Evaluation: [4, \"LLM systems can dramatically speed up first-pass debugging by summarizing and correlating logs/configs, mapping errors to likely root causes, searching the codebase for relevant call paths or changes, and proposing next checks or minimal repro steps, turning hours of triage into minutes in many cases.\", 1, \"Robustness\", \"LLM-based code agents can significantly reduce the completion time by assisting in reviews and determining sources. Lack of robustness can sometimes cause critical issues in software to persist. However, the same is true for humans, so therefore the use of LLMs does not significantly increase risks\"]",
            "Profession: Human Resources Managers\nTask: Identify staff vacancies and recruit, interview, and select applicants.\nExample Evaluation: [3, \"Discrimination, Privacy, Security, Misinformation, Malicious actors and Misuse\", \"Its a critical task and hiring errors can cause serious individual harm and create legal or reputational risk. LLMs raise risk via biased screening or interview framing, mishandling applicant PII, incorrect compliance guidance, and misuse (AI-crafted resumes, deepfake interviews, or prompt-injected materials). Many issues are catchable with structured rubrics, audits, and keeping LLM output advisory, not decisive, but subtle bias or privacy failures can slip through.\", 4, \"high reward when used for operations, not decisioning: draft job ads or outreach, summarize resumes into structured fields (verified), generate interview guides or rubrics, schedule or communicate, and draft documentation. Typical savings around 5–15 hours per hire (approximately $250–$1,500 at $50–$100/hr HR cost), while keeping quality if humans make final decisions using structured evidence and bias checks.\"]"
            #"Profession: Models\nTask: Pose for artists and photographers.\nExample Evaluation: [1, \"LLMs can help with planning (mood boards, pose lists, shot lists, direction cues, scheduling or coordination), but the core value—physically posing and performing on set—can’t be reduced much by an LLM system.\",0, \"No Risks\", \"The task is not safety critical. \"]",
            "Profession: Claims Adjusters, Examiners, and Investigators\nTask: Review police reports, medical treatment records, medical bills, or physical property damage to determine the extent of liability.\nExample Evaluation: [4, \"Misinformation, Privacy, Discrimination, Security, Malicious actors and Misuse\", \"The task is safety critical, wrong liability calls can seriously harm claimants or insurers and create legal or fraud exposure. LLMs raise risk by misreading or omitting evidence, hallucinating facts, making overconfident liability inferences, and introducing biased credibility judgments; sensitive medical or police data also heightens privacy or security risk, and fraudsters can tailor narratives to game the model. Humans can catch many issues if LLMs are limited to summarization or extraction and every claim is traced back to the record, but subtle distortions can slip through.\", 4, \"High reward for evidence organization: extract entities, codes, or line items, build timelines, flag inconsistencies or missing docs, draft notes or letters, and generate follow-up questions. Typical savings approximately 0.5–3 hours per complex claim (approximately $20–$240 at $35–$80/hr), improving cycle time while maintaining quality if adjusters verify key facts and keep final liability judgment human-driven.\"]",
        ],
    }

    prefix = """
For each task within different occupations, determine the risks and rewards of using an LLM system (e.g., LLMs, LLM-based agents) in collaboration with a human expert to complete the task. For determination, use your knowledge of the current capabilities of the best-performing LLM systems. First, considering the effort required for human experts to detect and mitigate LLM errors, determine whether the combination of LLM systems + human experts can substantially reduce the overall cost (in USD) or time required to complete the task while maintaining the same quality as expert humans working alone. Quantify this reward. Next, determine whether the task is safety-critical, i.e., whether unintentional errors or exploitation of LLM capabilities by malicious actors could cause significant harm at the individual, organizational, or societal level. If the task is safety-critical, quantify the extent to which using the combination of LLM systems + human experts increases the overall risk relative to having only human experts perform the task.


Key vulnerabilities of LLM systems to consider include:
- Discrimination: Unequal treatment of individuals or groups by LLM systems, often based on race, gender, or other sensitive characteristics, resulting in unfair outcomes and representation of those groups. Further, the accuracy and effectiveness of LLM decisions and actions are dependent on group membership, where decisions in LLM system design and biased training data lead to
unequal outcomes, reduced benefits, increased effort, and alienation of users.
- Toxicity: LLMs can expose users to harmful, abusive, unsafe, or inappropriate content. May involve providing advice or encouraging action. Examples of toxic content include hate speech, violence, extremism, illegal acts, or child sexual abuse material, as well as content that violates community norms such as profanity, inflammatory, political speech, or pornography.
- Privacy: LLM systems memorize and leak sensitive personal data or infer private information about individuals without their consent. Unexpected or unauthorized sharing of data and information can compromise user expectations of privacy, assist identity theft, or cause loss of confidential intellectual property.
- Security: Vulnerabilities can be exploited in LLM systems, software development toolchains, and hardware, resulting in unauthorized access, data and privacy breaches, or system manipulation causing unsafe outputs or behavior.
- Misinformation: LLM systems can inadvertently generate or spread incorrect or deceptive information, which can lead to inaccurate beliefs in users and undermine their autonomy.
Humans who make decisions based on false beliefs can experience physical, emotional, or material harm. Highly personalized LLM-generated misinformation creates “filter bubbles” where individuals only see what matches their existing beliefs, undermining shared reality and weakening social cohesion and political processes.
- Malicious actors and Misuse: LLM systems can be used to conduct large-scale disinformation campaigns, malicious surveillance, or targeted and sophisticated automated censorship and propaganda,
to manipulate political processes, public opinion, and behavior. LLM systems can be used to develop cyber weapons (e.g., by coding cheaper, more effective malware), develop new or enhance existing weapons (e.g., Lethal Autonomous
Weapons or chemical, biological, radiological, nuclear, and high-yield explosives), or use weapons to cause mass harm. LLM systems can be used to gain a personal advantage over others through cheating, fraud, scams, blackmail, or targeted manipulation of beliefs or behavior. Examples
include LLM-facilitated plagiarism for research or education, impersonating a trusted or fake individual for illegitimate financial benefit, or creating humiliating or sexual imagery.


For each evaluation:
- Rate both the reward and risk associated with LLM systems on a scale from 0 to 5. Here, for rewards, a rating of 0 means that using LLM systems cannot reduce overall task completion time or cost while keeping equivalent quality, rating <=2 means relatively low reduction in overall cost or time or both, rating of 3 means moderate reduction, and >=4 means high reduction. A rating of 0 for risks means that the task is not safety-critical, a risk rating <=2 means it can be easily detected and/or fixed by human experts, a rating of 3 means that the risk is moderate, and rating >=4 means the risk is high and cannot be easily detected. 
- Provide a detailed justification in the format: [risks_rating, "risks_types", "risks_justification", rewards_rating, "rewards_justification"].
"""


    # shot_key = "gpt5" if "gpt5" in model_name else ("mistral" if "mistral" in model_name else "claude")
    examples_text = "\n\n".join(examples["LLM"])
    task = f"Given the profession: {row['Title']}, and a specific task: '{row['Task']}', evaluate the risk of using an LLM system to perform the task."
    return f"{prefix}\n\n{examples_text}\n\n{task}"

def _row_key_from_series(row) -> str:
    """Stable composite key for a task row used for resume/checkpointing.
    Keys on (SOC Code, Task ID).
    """
    soc = row.get("O*Net SOC Code") if hasattr(row, "get") else None
    if soc is None:
        for k in ["O*NET-SOC Code", "O*NET SOC Code", "SOC", "SOC Code"]:
            soc = row.get(k, None)
            if soc is not None:
                break

    task_id = row.get("Task ID", None)
    if task_id is None:
        for k in ["TaskID", "TASK ID", "Task Id"]:
            task_id = row.get(k, None)
            if task_id is not None:
                break

    return f"{str(soc).strip()}||{_normalize_task_id(task_id)}"


# ----------------------------- INFERENCE + CHECKPOINT -----------------------------
def is_bedrock_model(model_path: str) -> bool:
    if not model_path:
        return False
    return  model_path.startswith("us.") or model_path.startswith("arn:aws:bedrock:")  or model_path.startswith("meta.")  or model_path.startswith("anthropic.")  or model_path.startswith("qwen.")


def is_gemini_model(model_path: str) -> bool:
    return bool(model_path) and (model_path.startswith("gemini-") or model_path.startswith("models/gemini-"))


def is_openai_model(model_path: str) -> bool:
    return bool(model_path) and model_path.startswith("openai:")


def is_xai_model(model_path: str) -> bool:
    """xAI / Grok via xAI API. Use model_path like 'xai:grok-4'."""
    return bool(model_path) and model_path.startswith("xai:")

def is_bedrock_llama_model(model_path: str) -> bool:
    # Match raw model IDs (meta.llama...), predefined inference profiles (us.meta.llama...), and any ARN containing meta.llama
    return bool(model_path) and ("meta.llama" in model_path)


def is_bedrock_qwen_model(model_path: str) -> bool:
    # Match raw Qwen IDs (qwen.*), inference profiles (us.qwen.*), and any ARN containing qwen.
    return bool(model_path) and (".qwen." in model_path or model_path.startswith("qwen."))



def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV atomically to avoid corrupting checkpoints on crash."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _normalize_task_id(x: Any) -> str:
    # Task IDs sometimes come in as ints/floats; normalize to a stable string key.
    s = str(x).strip()
    # Avoid "123.0" when reading from Excel.
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def _call_with_retries(call_fn, *args, max_retries: int = 5, base_sleep_s: float = 2.0, jitter_s: float = 0.5,
                       **kwargs):
    """Retry wrapper for transient API errors."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            sleep_s = base_sleep_s * (2 ** (attempt - 1)) + random.random() * jitter_s
            print(f"⚠️  API error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(sleep_s)
    raise last_err


def evaluate_with_model(model_name, model_path, df):
    """Run a model over pending tasks, saving checkpoints so the run can resume.

    Checkpoint file contains ALL original columns from the input sheet:
    O*Net SOC Code, Title, Task ID, Task, Task Type, Incumbents responding, Date, Domain Source
    plus the new LLM-generated columns for this model.
    """

    save_every = int(os.getenv("SAVE_EVERY", "10"))
    max_retries = int(os.getenv("MAX_RETRIES", "5"))
    resultdir = "results"
    os.makedirs(resultdir, exist_ok=True)

    checkpoint_path = os.path.join(resultdir, f"{model_name}_prompt_sensitivity.csv")
    errors_path = os.path.join(resultdir, f"{model_name}_errors.jsonl")

    col_rr = f"{model_name}_rewards_ratings"
    col_rj = f"{model_name}_rewards_justifications"
    col_risk_r = f"{model_name}_risks_ratings"
    col_risk_t = f"{model_name}_risks_types"
    col_risk_j = f"{model_name}_risks_justifications"
    out_cols = [col_rr, col_rj, col_risk_r, col_risk_t, col_risk_j]

    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    if os.path.exists(checkpoint_path):
        print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
        ckpt = pd.read_csv(checkpoint_path)
        for c in out_cols:
            if c not in ckpt.columns:
                ckpt[c] = pd.NA

        ckpt_keys = ckpt.apply(_row_key_from_series, axis=1)
        ckpt_map = dict(zip(ckpt_keys, ckpt[out_cols].to_dict(orient="records")))

        df_keys = df.apply(_row_key_from_series, axis=1)
        for idx, k in df_keys.items():
            if k in ckpt_map:
                vals = ckpt_map[k]
                for c in out_cols:
                    if pd.isna(df.at[idx, c]) and pd.notna(vals.get(c)):
                        df.at[idx, c] = vals.get(c)

    pending = df[df[col_rr].isna()].copy()
    if pending.empty:
        print(f"✅ All rows already evaluated for {model_name}")
        return df

    def _call_model(prompt: str) -> str:
        if is_bedrock_model(model_path):
            if is_bedrock_qwen_model(model_path):
                return _call_with_retries(
                    call_qwen_bedrock,
                    prompt,
                    model_id=model_path,
                    region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1",
                    max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "2048")),
                    temperature=float(os.getenv("QWEN_TEMPERATURE", "0")),
                    max_retries=max_retries,
                )
            if is_bedrock_llama_model(model_path):
                return _call_with_retries(
                    call_llama_bedrock,
                    prompt,
                    model_id=model_path,
                    region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1",
                    max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "2048")),
                    temperature=float(os.getenv("LLAMA_TEMPERATURE", "0")),
                    max_retries=max_retries,
                )
            return _call_with_retries(
                call_claude_bedrock,
                prompt,
                model_path,
                max_tokens=2048,
                temperature=0,
                max_retries=max_retries,
            )
        if is_gemini_model(model_path):
            return _call_with_retries(
                call_gemini_api,
                prompt,
                model_path.replace("https://generativelanguage.googleapis.com/v1beta/models/", ""),
                max_output_tokens=2048,
                temperature=0,
                max_retries=max_retries,
            )
        if is_openai_model(model_path):
            openai_model = model_path.split("openai:", 1)[1]
            return _call_with_retries(
                call_openai_gpt,
                prompt,
                openai_model,
                max_output_tokens=2048,
                max_retries=max_retries,
            )

        if is_xai_model(model_path):
            grok_model = model_path.split("xai:", 1)[1]
            return _call_with_retries(
                call_grok_api,
                prompt,
                grok_model,
                max_output_tokens=int(os.getenv("GROK_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("GROK_TEMPERATURE", "0")),
                max_retries=max_retries,
            )
        raise ValueError(f"Unknown model_path format for {model_name}: {model_path}")

    print(f"🔍 Using model: {model_name} ({model_path})")
    processed_since_save = 0

    try:
        for idx in tqdm(pending.index.tolist(), desc=f"Inference: {model_name}"):
            row = df.loc[idx]
            prompt = build_prompt(row, model_name)

            try:
                generated_text = _call_model(prompt)
                rir, rt, rij, rr, rj = parse_risks_rewards(generated_text)

                df.at[idx, col_rr] = rr
                df.at[idx, col_rj] = rj
                df.at[idx, col_risk_r] = rir
                df.at[idx, col_risk_t] = rt
                df.at[idx, col_risk_j] = rij

            except Exception as e:
                _append_jsonl(
                    errors_path,
                    {
                        "model": model_name,
                        "row_key": _row_key_from_series(row),
                        "soc": row.get("O*Net SOC Code"),
                        "title": row.get("Title"),
                        "task_id": row.get("Task ID"),
                        "task": row.get("Task"),
                        "error": repr(e),
                        "time": time.time(),
                    },
                )
                continue

            processed_since_save += 1
            if processed_since_save >= save_every:
                _atomic_write_csv(df, checkpoint_path)
                processed_since_save = 0

        _atomic_write_csv(df, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        return df

    except KeyboardInterrupt:
        _atomic_write_csv(df, checkpoint_path)
        print(f"💾 Interrupted — saved checkpoint: {checkpoint_path}")
        raise


# ----------------------------- AGGREGATION -----------------------------
def mode(row, colnames):
    values = [v for v in [row[c] for c in colnames] if isinstance(v, int)]
    if not values:
        return None
    return min(values) if len(set(values)) == 3 else pd.Series(values).mode().min()


# ----------------------------- MAIN -----------------------------
def main():
    set_seed(28)
    df = pd.read_excel("Task_Statements_Computer_Occupations.xlsx", engine="openpyxl", sheet_name="Computer Occupations")
    df.columns = df.columns.str.rstrip()
    print("column ", df.columns)
    if "Task ID" not in df.columns:
        raise KeyError("🛑 'Task ID' column not found in the input Excel file.")
    AVAILABLE_MODELS = {
        "gemini25_pro": (
            "gemini25_pro",
            "gemini-2.5-pro",
        ),
        "gpt5": (
            "gpt5",
            "openai:gpt-5",
        ),
        "grok_best_reasoning": (
            "grok_best_reasoning",
            os.getenv("GROK_MODEL_PATH", "xai:grok-4"),
        ),
        "claude_opus45": (
            "claude_opus45",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        ),
        "llama33_70b_bedrock": (
            "llama33_70b_bedrock",
            "us.meta.llama3-3-70b-instruct-v1:0",
        ),
        "qwen_best": (
            "qwen_best",
            os.getenv("QWEN_MODEL_ID", "qwen.qwen3-next-80b-a3b"),
        ),
    }
    #models = [
        #("qwen_best", os.getenv("QWEN_MODEL_ID", "qwen.qwen3-next-80b-a3b")),
        #("llama33_70b_bedrock", "us.meta.llama3-3-70b-instruct-v1:0"),
        #("claude_opus45", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
        #("gemini25_pro", "gemini-2.5-pro"),
        #("grok_best_reasoning", os.getenv("GROK_MODEL_PATH", "xai:grok-4")),
        #("gpt5", "openai:gpt-5"),
    #]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",required=True, choices=AVAILABLE_MODELS.keys(),help="Model to evaluate")

    args = parser.parse_args()

    models = [AVAILABLE_MODELS[args.model]]
    for model_name, model_path in models:
        df = evaluate_with_model(model_name, model_path, df)
        _atomic_write_csv(df, f"{model_name}_prompt_sensitivity.csv")
        print(f"✅ Wrote combined output")


if __name__ == "__main__":
    main()

