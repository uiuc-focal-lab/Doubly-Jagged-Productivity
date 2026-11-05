# Task-level clustering

This folder contains the code for the task-level analysis in Section 4.2 and Appendices D, H and I:
we cluster the 609 O\*NET tasks of computer occupations by the similarity of their descriptions and
look at how the clusters are spread across risk and reward levels.

## Setup

```bash
cd task_clustering
pip install -r requirements.txt
```

On Windows, create the Python environment in a short path (for example `C:\venvs\clustering`)
or enable long paths, otherwise installing torch can fail on the 260-character path limit.

## Reproducing the paper

```bash
python run_all.py
```

This builds Tables 4, 5 and 8 (CSV) and Figures 10–19 (interactive HTML) in `outputs/` from the
clustering used in the paper (`data/clustering_results.csv`) and the final cluster labels.

## Running the whole pipeline again

```bash
python run_all.py --from-scratch
```

This repeats every step: it embeds and clusters the tasks, re-numbers the clusters to match the
paper, drafts a label for each cluster with GPT-6 Luna, and builds the tables and figures from
the result. Everything is written to `outputs/from_scratch/`. Drafting the labels needs the
`openai` package and an `OPENAI_API_KEY`.

With the package versions in `requirements.txt` (we used Python 3.11 on Windows, CPU), the rerun
gives the same clusters as the paper. UMAP only guarantees identical results for the same seed on
the same platform and package versions, though, so on other systems a few tasks may end up in
different clusters. The labels drafted by GPT-6 Luna may also read differently from Table 4, even
when the clusters match, because the labels in the paper were edited by hand after drafting.

`--seed` runs the clustering with a different UMAP seed (the paper uses 53) and `--model` drafts
the labels with a different OpenAI model.

## Scripts

`run_all.py` calls the scripts below, which can also be run on their own.

| Script | What it does |
|--------|--------------|
| `cluster_tasks.py` | Clusters the tasks and writes `outputs/clustering_results.csv` |
| `label_clusters.py` | Table 4: the final cluster labels (`--draft` also drafts new labels with the LLM) |
| `cluster_distribution_by_risk_reward.py` | Table 5: clusters within each risk-reward group |
| `risk_type_prevalence_by_level.py` | Table 8: risk categories by risk level |
| `visualize_clusters.py` | Figures 10–19 |

Risk and reward levels are Low (≤ 2), Medium (between 2 and 4) and High (≥ 4), based on each
task's mean rating over the six LLMs. The ratings come from the `results/` folder at the top of
the repository: `Combined Ratings.xlsx` has each task's mean risk and reward, and
`<model>_partial.csv` has the risk categories each model assigned.

## Clustering pipeline

Each task description is embedded with `all-MiniLM-L6-v2`. A few occupations share identical
task descriptions; each distinct description is embedded once. UMAP reduces the embeddings to 5
dimensions (`n_neighbors=15`, `min_dist=0`, cosine distance, seed 53) and HDBSCAN clusters them
(`min_cluster_size=10`, `min_samples=3`). Tasks that HDBSCAN leaves unassigned are put in the most
common cluster among their 5 nearest neighbours.

HDBSCAN numbers its clusters in no particular order. To match the cluster numbers with the paper, we give each cluster the number of the paper cluster that shares the most tasks with it. Clusters with no counterpart get new numbers.

## Label Generation

For each cluster, the LLM sees all of the cluster's task descriptions, its five most common
occupations and five example tasks from every other cluster, and is asked for a 6–12 word
phrase that says what sets the cluster apart.
