import pandas as pd
from typing import Optional


class ONETTaskRatings:
    def __init__(self, filepath: str):
        self.df = pd.read_excel(filepath)
        self.df.columns = [c.strip() for c in self.df.columns]

        required_cols = {
            "O*NET-SOC Code",
            "Title",
            "Task ID",
            "Task",
            "Scale ID",
            "Category",
            "Data Value",
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                "Missing required columns in Task Ratings file: {}".format(missing)
            )

    @staticmethod
    def normalize_importance(x: float) -> float:
        return (x - 1.0) / 4.0

    @staticmethod
    def normalize_relevance(x: float) -> float:
        return x / 100.0

    @staticmethod
    def normalize_frequency(x: float) -> float:
        return (x - 1.0) / 6.0

    def _compute_frequency_value(self, task_rows: pd.DataFrame) -> Optional[float]:
        ft_rows = task_rows[task_rows["Scale ID"] == "FT"].copy()

        if ft_rows.empty:
            return None

        ft_rows["Category"] = pd.to_numeric(ft_rows["Category"], errors="coerce")
        ft_rows["Data Value"] = pd.to_numeric(ft_rows["Data Value"], errors="coerce")
        ft_rows = ft_rows.dropna(subset=["Category", "Data Value"])

        total = ft_rows["Data Value"].sum()
        if total == 0:
            return None

        weighted_avg = (ft_rows["Category"] * ft_rows["Data Value"]).sum() / total
        return float(weighted_avg)

    def get_soc_task_values(self, soc_name: str, exact_match: bool = True) -> pd.DataFrame:
        if exact_match:
            soc_rows = self.df[self.df["Title"] == soc_name].copy()
        else:
            soc_rows = self.df[
                self.df["Title"].str.contains(soc_name, case=False, na=False)
            ].copy()

        if soc_rows.empty:
            raise ValueError(
                "SOC name '{}' not found in Task Ratings file.".format(soc_name)
            )

        task_meta = (
            soc_rows[["O*NET-SOC Code", "Title", "Task ID", "Task"]]
            .drop_duplicates()
            .sort_values("Task ID")
        )

        results = []
        for _, row in task_meta.iterrows():
            task_id = row["Task ID"]
            task_rows = soc_rows[soc_rows["Task ID"] == task_id].copy()

            im_rows = task_rows[task_rows["Scale ID"] == "IM"]
            importance = None if im_rows.empty else float(im_rows.iloc[0]["Data Value"])

            rt_rows = task_rows[task_rows["Scale ID"] == "RT"]
            relevance = None if rt_rows.empty else float(rt_rows.iloc[0]["Data Value"])

            frequency = self._compute_frequency_value(task_rows)

            results.append(
                {
                    "soc_code": row["O*NET-SOC Code"],
                    "title": row["Title"],
                    "task_id": task_id,
                    "task": row["Task"],
                    "importance_norm": (
                        self.normalize_importance(importance)
                        if importance is not None
                        else None
                    ),
                    "relevance_norm": (
                        self.normalize_relevance(relevance)
                        if relevance is not None
                        else None
                    ),
                    "frequency_norm": (
                        self.normalize_frequency(frequency)
                        if frequency is not None
                        else None
                    ),
                }
            )

        return pd.DataFrame(results)


class OccupationRatings:
    def __init__(self, task_ratings_file: str, combined_ratings_file: str, reward_col: str, risk_col: str):
        self.onet = ONETTaskRatings(task_ratings_file)

        # Reads first sheet by default
        self.combined = pd.read_excel(combined_ratings_file)
        self.combined.columns = [c.strip() for c in self.combined.columns]
        self.reward_col = reward_col
        self.risk_col = risk_col
        required_cols = {
            "O*Net SOC Code",
            "Title",
            "Task ID",
            "Task",
            self.reward_col,
            self.risk_col,
        }
        missing = required_cols - set(self.combined.columns)
        if missing:
            raise ValueError(
                "Missing required columns in Combined Ratings file: {}".format(missing)
            )

    def compute_occupation_rating(self, soc_name: str, exact_match: bool = True) -> dict:
        task_values = self.onet.get_soc_task_values(
            soc_name, exact_match=exact_match
        ).copy()

        task_values["raw_weight"] = (
            task_values["importance_norm"]
            * task_values["relevance_norm"]
            * task_values["frequency_norm"]
        )

        task_values = task_values.dropna(subset=["raw_weight"])
        task_values = task_values[task_values["raw_weight"] > 0].copy()

        if task_values.empty:
            raise ValueError(
                "No valid task weights found for SOC '{}'.".format(soc_name)
            )

        task_values["norm_weight"] = task_values["raw_weight"] / task_values["raw_weight"].sum()

        if exact_match:
            combined_soc = self.combined[self.combined["Title"] == soc_name].copy()
        else:
            combined_soc = self.combined[
                self.combined["Title"].str.contains(soc_name, case=False, na=False)
            ].copy()

        if combined_soc.empty:
            raise ValueError(
                "SOC name '{}' not found in Combined Ratings file.".format(soc_name)
            )

        merged = pd.merge(
            task_values,
            combined_soc[["Task ID", self.reward_col, self.risk_col]],
            left_on="task_id",
            right_on="Task ID",
            how="inner",
        )

        if merged.empty:
            raise ValueError(
                "No matching tasks found after merging for SOC '{}'.".format(soc_name)
            )
        print(merged["Task ID"], merged["norm_weight"], merged[self.risk_col])
        occupation_reward = (merged["norm_weight"] * merged[self.reward_col]).sum()
        occupation_risk = (merged["norm_weight"] * merged[self.risk_col]).sum()
        occupation_gain = occupation_reward - occupation_risk

        return {
            "soc_name": soc_name,
            "soc_code": merged["soc_code"].iloc[0] if "soc_code" in merged.columns else None,
            "n_tasks_used": int(len(merged)),
            "occupation_reward": float(occupation_reward),
            "occupation_risk": float(occupation_risk),
            "occupation_gain": float(occupation_gain),
        }


if __name__ == "__main__":
    task_ratings_file = "DFA Task Ratings.xlsx"
    combined_ratings_file = "results/Combined Ratings.xlsx"
    soc_names = [
       "Digital Forensics Analysts"
    ]

    occ = OccupationRatings(task_ratings_file, combined_ratings_file,"Average Reward", "Average Risks")

    results = []
    missing_or_failed = []

    for soc_name in soc_names:
        try:
            result = occ.compute_occupation_rating(soc_name, exact_match=True)
            results.append(result)
            print(
                "Processed: {} | reward={:.4f}, risk={:.4f}, gain={:.4f}, tasks={}".format(
                    result["soc_name"],
                    result["occupation_reward"],
                    result["occupation_risk"],
                    result["occupation_gain"],
                    result["n_tasks_used"],
                )
            )
        except Exception as e:
            print("Skipping: {} | {}".format(soc_name, e))
            missing_or_failed.append(
                {
                    "soc_name": soc_name,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results)
    missing_df = pd.DataFrame(missing_or_failed)

    if not results_df.empty:
        results_df = results_df.sort_values("occupation_gain", ascending=False)
        results_df.to_csv("occupation_ratings.csv", index=False)
        print("\nSaved successful results to occupation_ratings.csv")

    if not missing_df.empty:
        missing_df.to_csv("occupation_ratings_missing_or_failed.csv", index=False)
        print("Saved skipped SOCs to occupation_ratings_missing_or_failed.csv")

    print("\nDone.")