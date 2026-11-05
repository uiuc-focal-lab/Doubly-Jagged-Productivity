#!/usr/bin/env python3
"""
Table 4: cluster labels.

Each label was drafted by an LLM and then reviewed and edited by hand. For every
cluster the model sees all of the cluster's tasks, its most common occupations,
and a few example tasks from each of the other clusters, so that the label
describes what sets the cluster apart. The final labels are listed below.

To repeat the drafting step (needs `pip install openai` and an OPENAI_API_KEY):

    python label_clusters.py --draft [--model MODEL]

The drafts are written to outputs/cluster_label_drafts.csv next to the final labels.
"""

import argparse
import json
import random

import pandas as pd

from common import OUTPUT_DIR, load_tasks_with_clusters, save

MODEL = "gpt-6-luna"
EXAMPLES_PER_OTHER_CLUSTER = 5

CLUSTER_LABELS = {
    0:  "Analyzing and visualizing geospatial data through specialized GIS applications",
    1:  "Health information technology solutions",
    2:  "Designing interactive game experiences through iterative prototyping and user feedback",
    3:  "Maintaining and updating web infrastructure and server configurations",
    4:  "Testing security vulnerabilities and implementing cybersecurity measures",
    5:  "Integrating and optimizing data warehouse systems and processes",
    6:  "Architecting and optimizing databases for performance and scalability",
    7:  "Training users and coordinating technical project teams and activities",
    8:  "Managing IT project resources, schedules, and stakeholder communications",
    9:  "Creating and optimizing document management systems",
    10: "Designing and developing user-focused web interfaces and applications",
    11: "Analyzing and designing computer systems architecture and solutions",
    12: "Evaluating and testing software quality through structured methodologies",
    13: "Developing and documenting system design procedures and data workflows",
    14: "Administering network systems and maintaining logs and automation",
    15: "Analyzing and designing network systems to meet organizational requirements",
    16: "Evaluating and testing system performance and new technology solutions",
    17: "Installing and supporting telecommunications equipment and systems",
    18: "Troubleshooting and resolving user network and software issues",
    19: "Monitoring network performance and configuring network equipment and capacity",
    20: "Maintaining, repairing, and upgrading computer hardware, software, and systems",
}

INSTRUCTIONS = """You are labeling clusters of work tasks from O*NET computer occupations. The tasks were \
grouped by the similarity of their descriptions.

Write a label for the target cluster that:
- describes the activity its tasks have in common,
- distinguishes it from the other clusters shown below,
- is a single phrase of 6 to 12 words starting with a gerund, in sentence case,
  for example "Scheduling patient appointments and maintaining medical records",
- avoids filler such as "tasks", "focus on", "various" or "related to".

Respond in JSON: {"label": "..."}"""


def describe(tasks, n=None, seed=0):
    sample = tasks.Task.tolist() if n is None else random.Random(seed).sample(tasks.Task.tolist(), min(n, len(tasks)))
    return "\n".join(f"- {t}" for t in sample)


def draft_label(client, model, cluster, tasks):
    target = tasks[tasks.Cluster == cluster]
    others = "\n\n".join(f"Cluster {c}:\n{describe(group, EXAMPLES_PER_OTHER_CLUSTER, seed=c)}"
                         for c, group in tasks[tasks.Cluster != cluster].groupby("Cluster"))
    occupations = "\n".join(f"- {p}: {n}" for p, n in target.Profession.value_counts().head(5).items())
    prompt = (f"{INSTRUCTIONS}\n\nTARGET CLUSTER ({len(target)} tasks)\n\nTasks:\n{describe(target)}\n\n"
              f"Most common occupations:\n{occupations}\n\nOTHER CLUSTERS (example tasks)\n\n{others}")
    response = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)["label"]


def draft_labels(tasks, model=MODEL):
    """Draft a label for every cluster in `tasks` with an LLM. Returns {cluster: label}."""
    from openai import OpenAI

    client = OpenAI()
    return {c: draft_label(client, model, c, tasks) for c in sorted(tasks.Cluster.unique())}


def main(labels=CLUSTER_LABELS, out_dir=OUTPUT_DIR):
    table = pd.DataFrame({"Cluster": [f"C{c}" for c in labels], "Label": labels.values()})
    print(table.to_string(index=False))
    save(table, out_dir, "table4_cluster_labels.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write Table 4 (cluster labels).")
    parser.add_argument("--draft", action="store_true", help="also draft the labels with an LLM")
    parser.add_argument("--model", default=MODEL, help=f"model used for drafting (default: {MODEL})")
    args = parser.parse_args()
    main()
    if args.draft:
        drafts = draft_labels(load_tasks_with_clusters(), args.model)
        save(pd.DataFrame({"Cluster": [f"C{c}" for c in drafts], "Draft": drafts.values(),
                           "Final": [CLUSTER_LABELS.get(c, "") for c in drafts]}),
             OUTPUT_DIR, "cluster_label_drafts.csv")
