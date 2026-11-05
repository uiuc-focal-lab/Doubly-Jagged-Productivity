#!/usr/bin/env python3
"""
Semantic clustering of the 609 task descriptions.

  1. Embed each task description with all-MiniLM-L6-v2 (384 dimensions).
     Some occupations share identical task descriptions; each distinct
     description is embedded and clustered once and its cluster is assigned to
     every task that uses it.
  2. Reduce to 5 dimensions with UMAP (n_neighbors=15, min_dist=0, cosine).
  3. Cluster with HDBSCAN (min_cluster_size=10, min_samples=3).
  4. Assign the points HDBSCAN marks as outliers to the majority cluster of
     their 5 nearest clustered neighbours.

HDBSCAN numbers clusters in an arbitrary order, so each cluster is given the ID
of the paper cluster it shares the most tasks with.

The clustering reported in the paper (seed 53) is saved in
data/clustering_results.csv. Running this script writes a fresh copy to
outputs/clustering_results.csv (use --seed to try another seed).

UMAP results are exactly reproducible for a fixed seed on the same platform and
package versions (see requirements.txt). On other platforms small numerical
differences can change some cluster assignments.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

from common import OUTPUT_DIR, load_clusters, load_tasks, save

# UMAP warns that a fixed random_state disables parallelism, which is intended here.
warnings.filterwarnings("ignore", message=".*n_jobs value .* overridden.*")

SEED = 53
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.0, metric="cosine")


def embed_tasks(tasks):
    """Embed each distinct description once; embeddings[index[i]] belongs to task i."""
    texts, index = np.unique(tasks.Task.values, return_inverse=True)
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")
    return model.encode(list(texts), batch_size=32), index


def cluster(embeddings, seed=SEED):
    """Return the cluster of each embedding and the fraction HDBSCAN marked as outliers."""
    reduced = UMAP(n_components=5, random_state=seed, **UMAP_PARAMS).fit_transform(embeddings)
    labels = HDBSCAN(min_cluster_size=10, min_samples=3).fit_predict(reduced)
    outliers = labels == -1
    knn = KNeighborsClassifier(n_neighbors=5).fit(reduced[~outliers], labels[~outliers])
    labels[outliers] = knn.predict(reduced[outliers])
    return labels, outliers.mean()


def number_like_paper(labels, task_ids):
    """Give each cluster the ID of the paper cluster it shares the most tasks with.

    Clusters are matched one-to-one so that the total Jaccard similarity of matched
    pairs is as large as possible. Clusters with no counterpart get new IDs after
    the paper's last one.
    """
    paper = pd.Series(task_ids).map(load_clusters().set_index("Task_ID").Cluster).values
    overlap = pd.crosstab(labels, paper)
    union = overlap.sum(axis=1).values[:, None] + overlap.sum(axis=0).values - overlap.values
    jaccard = overlap.values / union
    rows, cols = linear_sum_assignment(-jaccard)
    ids = {overlap.index[r]: overlap.columns[c] for r, c in zip(rows, cols) if jaccard[r, c] > 0}
    unmatched = [c for c in overlap.sum(axis=1).sort_values(ascending=False).index if c not in ids]
    ids.update({c: paper.max() + 1 + i for i, c in enumerate(unmatched)})
    return np.array([ids[c] for c in labels])


def cluster_all_tasks(seed=SEED):
    """Cluster all tasks and number the clusters as in the paper."""
    tasks = load_tasks()
    embeddings, index = embed_tasks(tasks)
    labels, outlier_fraction = cluster(embeddings, seed)
    tasks["Cluster"] = number_like_paper(labels[index], tasks.Task_ID)
    print(f"Seed {seed}: {tasks.Cluster.nunique()} clusters; {outlier_fraction:.1%} of the {len(embeddings)} "
          f"distinct descriptions were HDBSCAN outliers and were reassigned")
    return tasks


def main(seed=SEED, out_dir=OUTPUT_DIR):
    tasks = cluster_all_tasks(seed)
    save(tasks[["Task_ID", "Profession", "Task", "Cluster"]], out_dir, "clustering_results.csv")
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster the task descriptions.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"UMAP random seed (default: {SEED}, as in the paper)")
    main(parser.parse_args().seed)
