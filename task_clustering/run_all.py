#!/usr/bin/env python3
"""
Produce Tables 4, 5 and 8 and Figures 10-19.

    python run_all.py
        Reproduces the paper from the saved clustering (data/clustering_results.csv)
        and the final, hand-edited cluster labels. Output: outputs/

    python run_all.py --from-scratch [--seed 53] [--model gpt-6-luna]
        Runs the whole pipeline again: embeds and clusters the tasks, numbers the
        clusters as in the paper, drafts the labels with an LLM (needs an
        OPENAI_API_KEY) and builds the tables and figures from that.
        Output: outputs/from_scratch/
"""

import argparse

import cluster_distribution_by_risk_reward
import cluster_tasks
import label_clusters
import risk_type_prevalence_by_level
import visualize_clusters
from common import OUTPUT_DIR, load_tasks_with_clusters


def run(tasks, labels, out_dir, seed):
    print("\n== Table 4: cluster labels ==\n")
    label_clusters.main(labels, out_dir)
    print("\n== Table 5: cluster distribution by risk-reward group ==\n")
    cluster_distribution_by_risk_reward.main(tasks, out_dir)
    print("\n== Table 8: risk categories by risk level ==\n")
    risk_type_prevalence_by_level.main(out_dir)
    print("\n== Figures 10-19: cluster maps ==\n")
    visualize_clusters.main(tasks, labels, out_dir, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-scratch", action="store_true", help="rerun the clustering and the labelling")
    parser.add_argument("--seed", type=int, default=cluster_tasks.SEED, help="UMAP seed (default: %(default)s)")
    parser.add_argument("--model", default=label_clusters.MODEL, help="LLM for the labels (default: %(default)s)")
    args = parser.parse_args()

    if args.from_scratch:
        out_dir = OUTPUT_DIR / "from_scratch"
        print("== Clustering ==\n")
        tasks = cluster_tasks.main(args.seed, out_dir)
        print(f"\n== Drafting cluster labels with {args.model} ==\n")
        labels = label_clusters.draft_labels(tasks, args.model)
        run(tasks, labels, out_dir, args.seed)
    else:
        run(load_tasks_with_clusters(), label_clusters.CLUSTER_LABELS, OUTPUT_DIR, cluster_tasks.SEED)
