"""Paths and data loading shared by the scripts in this folder."""

from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CLUSTERING = ROOT / "data" / "clustering_results.csv"

# Folder with each domain's LLM ratings. The paper analyses the computer occupations.
DOMAINS = {
    "computer": ROOT.parent / "results",
    "legal": ROOT.parent / "results_legal",
    "healthcare": ROOT.parent / "results_healthcare",
}
RESULTS_DIR = DOMAINS["computer"]

MODELS = ["claude_opus45", "gemini25_pro", "gpt5", "grok_best_reasoning", "llama33_70b_bedrock", "qwen_best"]
LEVELS = ["Low", "Medium", "High"]

# A task is assigned a risk category if at least this many of the six LLMs name it.
MAJORITY = 4

# The LLMs' risk labels vary in capitalisation and spacing, so categories are
# matched by keyword after lower-casing and removing spaces.
RISK_KEYWORDS = {
    "malicious": "Malicious Actors and Misuse",
    "misuse": "Malicious Actors and Misuse",
    "discrimination": "Discrimination",
    "toxic": "Toxicity",
    "privacy": "Privacy",
    "security": "Security",
    "misinformation": "Misinformation",
}


def level(rating):
    """Low (<= 2), Medium (between 2 and 4) or High (>= 4)."""
    return "Low" if rating <= 2 else "Medium" if rating < 4 else "High"


def load_tasks(domain="computer"):
    """The domain's tasks (609 for the computer occupations) with their mean risk and
    reward over the six LLMs, sorted by Task_ID."""
    df = pd.read_excel(DOMAINS[domain] / "Combined Ratings.xlsx", sheet_name="Combined")
    df.columns = df.columns.astype(str).str.strip()
    df = df[pd.to_numeric(df["Task ID"], errors="coerce").notna()]  # drop the summary table below the data
    df = df.rename(columns={"Task ID": "Task_ID", "Title": "Profession",
                            "Average Risks": "Risk_Mean", "Average Reward": "Reward_Mean"})
    df = df.astype({"Task_ID": int, "Risk_Mean": float, "Reward_Mean": float})
    df["Profession"] = df["Profession"].str.strip()
    df["Task"] = df["Task"].str.strip()
    df["Risk_Level"] = df["Risk_Mean"].map(level)
    df["Reward_Level"] = df["Reward_Mean"].map(level)
    columns = ["Task_ID", "Profession", "Task", "Risk_Mean", "Reward_Mean", "Risk_Level", "Reward_Level"]
    return df[columns].sort_values("Task_ID").reset_index(drop=True)


def load_clusters():
    """The clustering reported in the paper (Task_ID, Cluster)."""
    return pd.read_csv(CLUSTERING)[["Task_ID", "Cluster"]]


def load_tasks_with_clusters():
    """load_tasks() with each task's cluster in the paper."""
    return load_tasks().merge(load_clusters(), on="Task_ID", validate="1:1")


def save(table, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / name, index=False)
    print(f"Saved {out_dir / name}")


def load_majority_risk_types():
    """Task_ID -> risk categories named by at least MAJORITY of the six LLMs."""
    counts = {}
    for model in MODELS:
        df = pd.read_csv(RESULTS_DIR / f"{model}_partial.csv")
        for task_id, labels in zip(df["Task ID"].astype(int), df[f"{model}_risks_types"]):
            key = str(labels).lower().replace(" ", "")
            categories = {category for keyword, category in RISK_KEYWORDS.items() if keyword in key}
            counts.setdefault(task_id, Counter()).update(categories)
    return {task_id: {c for c, n in counter.items() if n >= MAJORITY} for task_id, counter in counts.items()}
