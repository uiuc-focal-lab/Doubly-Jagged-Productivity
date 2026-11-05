#!/usr/bin/env python3
"""
This code reproduces Table 5 in the paper. It shows how the tasks in each
risk-reward group are distributed over clusters.

Output: outputs/table5_cluster_distribution.csv
"""

import pandas as pd

from common import LEVELS, OUTPUT_DIR, load_tasks_with_clusters, save


def main(tasks=None, out_dir=OUTPUT_DIR):
    tasks = load_tasks_with_clusters() if tasks is None else tasks
    rows = []
    for risk in LEVELS:
        for reward in LEVELS:
            group = tasks[(tasks.Risk_Level == risk) & (tasks.Reward_Level == reward)]
            shares = group.Cluster.value_counts(normalize=True)  # largest share first
            rows.append({"Risk_Level": risk, "Reward_Level": reward, "N": len(group),
                         "Distribution": ", ".join(f"C{c} ({100 * s:.1f}%)" for c, s in shares.items())})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    save(table, out_dir, "table5_cluster_distribution.csv")


if __name__ == "__main__":
    main()
