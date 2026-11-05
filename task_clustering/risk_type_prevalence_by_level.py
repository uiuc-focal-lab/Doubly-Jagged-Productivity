#!/usr/bin/env python3
"""
Table 8: how often each risk category occurs at each risk level.

A task carries a category if at least 4 of the 6 LLMs named it. A task can carry
several categories, so the percentages in a risk level do not sum to 100.
"""

import pandas as pd

from common import LEVELS, OUTPUT_DIR, load_majority_risk_types, load_tasks, save

CATEGORIES = ["Misinformation", "Security", "Privacy", "Malicious Actors and Misuse", "Discrimination", "Toxicity"]


def main(out_dir=OUTPUT_DIR):
    tasks = load_tasks()
    risk_types = tasks.Task_ID.map(load_majority_risk_types())

    rows = []
    for level in LEVELS:
        in_level = tasks.Risk_Level == level
        counts = {c: sum(c in types for types in risk_types[in_level]) for c in CATEGORIES}
        for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            rows.append({"Risk_Level": level, "Risk_Category": category, "Count": n, "Total": in_level.sum(),
                         "Percentage": round(100 * n / in_level.sum(), 1)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    save(table, out_dir, "table8_risk_types_by_level.csv")


if __name__ == "__main__":
    main()
