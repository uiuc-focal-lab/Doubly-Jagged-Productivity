#!/usr/bin/env python3
"""
Comparison of the LLM risk and reward scores with the Workbank survey.

Workbank (Shao et al., 2025) has task-level ratings from 1,361 domain workers and
AI experts for O*NET tasks. Tasks are matched to ours by O*NET Task ID, and every
Workbank variable is averaged over the people who rated the task. For yes/no
questions we use the proportion of raters who answered yes. The script then

  (a) uses Kolmogorov-Smirnov tests to verify that the matched subset is
      representative of (has the same distribution as) our reward and risk data,
  (b) plots the distribution of gain (reward - risk) for the matched tasks, and
  (c) computes the Spearman correlation of each Workbank variable with gain.

The Workbank files are downloaded once from Hugging Face (pinned to the version
used in the paper) into data/.

Usage (with the repository's requirements.txt installed):

    python workbank_validation.py

Output: outputs/figure3_workbank.png
        outputs/workbank_correlations.csv
"""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RATINGS = ROOT.parent / "results" / "Combined Ratings.xlsx"

WORKBANK_URL = "https://huggingface.co/datasets/SALT-NLP/WORKBank/resolve/8d9f94136118fdc6fb225caa98f47e0256235522/"
WORKER_FILE = "worker_data/domain_worker_desires.csv"
EXPERT_FILE = "expert_ratings/expert_rated_technological_capability.csv"

# Workbank column -> label in Figure 3, in the groups and order shown there.
VARIABLE_GROUPS = [
    {
        "Automation Desire Rating": "Automation Desire",
        "Human Agency Scale Rating": "Human Agency Scale",
        "Physical Action Requirement": "Physical Action Req.",
        "Interpersonal Communication Requirement": "Interpersonal Comm. Req.",
        "Involved Uncertainty": "Involved Uncertainty",
        "Domain Expertise Requirement": "Domain Expertise Req.",
    },
    {
        "Time": "Time",
        "Core Skill Rating": "Core Skill",
        "Job Security Rating": "Job Security",
        "Enjoyment Rating": "Enjoyment",
    },
    {
        "Reasons for Automation Desire - Free Time": "AutoR: Free Time",
        "Reasons for Automation Desire - Repetitive": "AutoR: Repetitive",
        "Reasons for Automation Desire - Human Error": "AutoR: Human Error",
        "Reasons for Automation Desire - Stress": "AutoR: Stress",
        "Reasons for Automation Desire - Difficulty": "AutoR: Difficulty",
        "Reasons for Automation Desire - Scale": "AutoR: Scale",
    },
    {
        "Reasons for Human Agency - Physical": "AgencyR: Physical",
        "Reasons for Human Agency - Control": "AgencyR: Control",
        "Reasons for Human Agency - Domain Knowledge": "AgencyR: Domain Knowledge",
        "Reasons for Human Agency - Empathy": "AgencyR: Empathy",
        "Reasons for Human Agency - Quality Oversight": "AgencyR: Quality Oversight",
        "Reasons for Human Agency - Dynamic": "AgencyR: Dynamic",
        "Reasons for Human Agency - Ethical": "AgencyR: Ethical",
    },
    {
        "Automation Capacity Rating": "Expert Automation Capacity",
    },
]
VARIABLES = {column: label for group in VARIABLE_GROUPS for column, label in group.items()}


def workbank_file(name):
    path = DATA_DIR / name.replace("/", "__")
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urlretrieve(WORKBANK_URL + name, path)
    return pd.read_csv(path)


def load_llm_scores():
    """Task ID, mean reward and mean risk over the six LLMs for the 609 tasks."""
    df = pd.read_excel(RATINGS, sheet_name="Combined")
    df.columns = df.columns.astype(str).str.strip()
    df = df[pd.to_numeric(df["Task ID"], errors="coerce").notna()]  # drop the summary table below the data
    scores = pd.DataFrame({
        "Task_ID": df["Task ID"].astype(int),
        "Occupation": df["Title"],
        "Reward": df.filter(regex="_rewards_ratings$").astype(float).mean(axis=1),
        "Risk": df.filter(regex="_risks_ratings$").astype(float).mean(axis=1),
    })
    return scores.assign(Gain=scores.Reward - scores.Risk)


def load_workbank():
    """One row per Workbank task: each variable averaged over the people who rated it."""
    # Experts also rate some task characteristics; like the paper we use the workers' ratings
    # for those and take only the automation capacity rating from the experts.
    expert_columns = ["Automation Capacity Rating"]
    worker_columns = [c for c in VARIABLES if c not in expert_columns]
    workers = workbank_file(WORKER_FILE).astype({c: float for c in worker_columns})  # True/False -> 1/0
    experts = workbank_file(EXPERT_FILE)
    return (workers.groupby("Task ID")[worker_columns].mean()
            .join(experts.groupby("Task ID")[expert_columns].mean()))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def plot(scores, matched, correlations):
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.3, wspace=0.45)

    # (a) reward and risk distributions: all tasks vs. matched tasks
    panel_a = grid[0, :].subgridspec(1, 2, wspace=0.2)
    for i, (col, title) in enumerate([("Reward", "Reward"), ("Risk", "Risk")]):
        ax = fig.add_subplot(panel_a[0, i])
        bins = np.histogram_bin_edges(scores[col], bins=25)
        ax.hist(scores[col], bins=bins, density=True, alpha=0.6, label=f"Full (N={len(scores)})")
        ax.hist(matched[col], bins=bins, density=True, alpha=0.6, label=f"Subset (N={len(matched)})")
        ks = ks_2samp(scores[col], matched[col])
        ax.text(0.97, 0.95, f"KS D={ks.statistic:.2f}\np={ks.pvalue:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        ax.set(title=title, xlabel=f"{col} Mean", ylabel="Density")
        ax.legend(loc="upper left", fontsize=8)
        if i == 0:
            ax.set_title("(a)", loc="left", fontweight="bold")

    # (b) gain distribution of the matched tasks
    ax = fig.add_subplot(grid[1, 0])
    ax.hist(matched.Gain, bins=25, color="tab:green", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title=f"Distribution of LLM Gain (mean={matched.Gain.mean():.2f}, sd={matched.Gain.std():.2f})",
           xlabel="Gain (Reward - Risk)", ylabel="Count")
    ax.set_title("(b)", loc="left", fontweight="bold")

    # (c) Spearman correlations with gain
    ax = fig.add_subplot(grid[1, 1])
    rho = correlations.rho.values[:, None]
    image = ax.imshow(rho, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    for i, row in enumerate(correlations.itertuples()):
        ax.text(0, i, f"{row.rho:z.2f}{stars(row.p)}", ha="center", va="center", fontsize=8, fontweight="bold")
    for boundary in np.cumsum([len(group) for group in VARIABLE_GROUPS])[:-1]:  # separate the groups
        ax.axhline(boundary - 0.5, color="black", linewidth=0.8)
    ax.set_yticks(range(len(correlations)), correlations.Variable, fontsize=8)
    ax.set_xticks([0], ["Gain"])
    ax.set_title("Spearman rho")
    ax.set_title("(c)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="rho", shrink=0.6)
    return fig


def main():
    scores = load_llm_scores()
    workbank = load_workbank()
    matched = scores.merge(workbank, left_on="Task_ID", right_index=True)
    print(f"{len(matched)} of {len(scores)} tasks matched to Workbank, "
          f"covering {matched.Occupation.nunique()} of {scores.Occupation.nunique()} occupations")
    for col in ("Reward", "Risk"):
        ks = ks_2samp(scores[col], matched[col])
        print(f"KS test, {col.lower()}: D={ks.statistic:.2f}, p={ks.pvalue:.2f}")
    print(f"Gain of matched tasks: mean {matched.Gain.mean():.2f}, sd {matched.Gain.std():.2f}\n")

    rows = []
    for column, label in VARIABLES.items():
        rho, p = spearmanr(matched[column], matched.Gain)
        rows.append({"Variable": label, "rho": rho, "p": p})
    correlations = pd.DataFrame(rows)
    print(correlations.assign(sig=correlations.p.map(stars)).to_string(index=False, float_format="{:.3f}".format))

    OUTPUT_DIR.mkdir(exist_ok=True)
    correlations.to_csv(OUTPUT_DIR / "workbank_correlations.csv", index=False)
    plot(scores, matched, correlations).savefig(OUTPUT_DIR / "figure3_workbank.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved {OUTPUT_DIR / 'workbank_correlations.csv'}\nSaved {OUTPUT_DIR / 'figure3_workbank.png'}")


if __name__ == "__main__":
    main()
