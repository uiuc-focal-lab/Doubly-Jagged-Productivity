import re
import numpy as np
import pandas as pd


FILES = {
    "claude": "res_occ_claude.txt",
    "gemini": "res_occ_gemini.txt",
    "gpt5": "res_occ_gpt5.txt",
    "grok": "res_occ_grok.txt",
    "llama": "res_occ_llama.txt",
    "qwen": "res_occ_qwen.txt",
}


LINE_RE = re.compile(
    r"^Processed:\s*(?P<soc>.*?)\s*\|\s*reward=(?P<reward>-?\d+(?:\.\d+)?)"
    r",\s*risk=(?P<risk>-?\d+(?:\.\d+)?)"
    r",\s*gain=(?P<gain>-?\d+(?:\.\d+)?)"
    r",\s*tasks=(?P<tasks>\d+)\s*$"
)


def parse_occ_file(filepath, model_name):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = LINE_RE.match(line)
            if not m:
                continue

            rows.append(
                {
                    "model": model_name,
                    "soc_name": m.group("soc"),
                    "reward": float(m.group("reward")),
                    "risk": float(m.group("risk")),
                    "gain": float(m.group("gain")),
                    "tasks": int(m.group("tasks")),
                }
            )
    return pd.DataFrame(rows)


def load_all_files(file_map):
    dfs = []
    for model_name, filepath in file_map.items():
        df = parse_occ_file(filepath, model_name)
        if df.empty:
            print("Warning: no parsed rows for", model_name, "from", filepath)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compute_mean_std(df):
    summary = (
        df.groupby("soc_name")
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", lambda x: np.std(x, ddof=1)),
            risk_mean=("risk", "mean"),
            risk_std=("risk", lambda x: np.std(x, ddof=1)),
            gain_mean=("gain", "mean"),
            gain_std=("gain", lambda x: np.std(x, ddof=1)),
            n_models=("model", "nunique"),
            tasks_min=("tasks", "min"),
            tasks_max=("tasks", "max"),
        )
        .reset_index()
    )

    return summary.sort_values("soc_name").reset_index(drop=True)


def main():
    long_df = load_all_files(FILES)

    if long_df.empty:
        print("No data parsed.")
        return

    # Save per-model long table
    long_df.to_csv("occupation_scores_per_model.csv", index=False)

    # Compute summary stats across models
    summary_df = compute_mean_std(long_df)
    summary_df.to_csv("occupation_scores_mean_std.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nPer-model parsed data:")
    print(long_df.head(20).to_string(index=False))

    print("\nOccupation-level mean and standard deviation across models:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()