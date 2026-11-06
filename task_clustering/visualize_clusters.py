#!/usr/bin/env python3
"""
Interactive maps of the task clusters.

Tasks are placed by a 2-D UMAP projection of their embeddings (same UMAP settings
and seed as the clustering) and coloured by cluster.

  Figure 10      all tasks
  Figures 11-19  one per risk-reward group

Output: outputs/figures/fig<NN>_<group>.html
"""

from textwrap import wrap

import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP

from cluster_tasks import SEED, UMAP_PARAMS, embed_tasks
from common import OUTPUT_DIR, load_tasks_with_clusters
from label_clusters import CLUSTER_LABELS

# (figure number, risk level, reward level), in the order of Appendix I
FIGURES = [
    (11, "High", "Medium"), (12, "High", "High"), (13, "High", "Low"),
    (14, "Medium", "Medium"), (15, "Medium", "High"), (16, "Medium", "Low"),
    (17, "Low", "Medium"), (18, "Low", "High"), (19, "Low", "Low"),
]
COLORS = px.colors.qualitative.Dark24
NO_LABEL = "(no label)"


def hover_text(task, label):
    label = "<br>".join(wrap(label, 60))
    description = "<br>".join(wrap(task.Task, 60))
    return (f"<b>C{task.Cluster}: {label}</b><br><br><b>Task {task.Task_ID}:</b> {description}"
            f"<br><b>Profession:</b> {task.Profession}"
            f"<br><b>Risk</b> {task.Risk_Mean:.2f}, <b>Reward</b> {task.Reward_Mean:.2f}")


def plot(tasks, title, labels):
    fig = go.Figure()
    for cluster, group in sorted(tasks.groupby("Cluster"), key=lambda item: -len(item[1])):
        label = labels.get(cluster, NO_LABEL)
        fig.add_trace(go.Scatter(
            x=group.X, y=group.Y, mode="markers",
            name=f"<b>{100 * len(group) / len(tasks):.1f}%</b>  C{cluster}: {label}",
            marker=dict(size=8, color=COLORS[cluster % len(COLORS)], opacity=0.75, line=dict(width=0.5, color="white")),
            text=[hover_text(t, label) for t in group.itertuples()], hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="UMAP dimension 1", yaxis_title="UMAP dimension 2",
                      width=1400, height=900, plot_bgcolor="white", margin=dict(r=380),
                      legend=dict(x=1.01, y=1, font=dict(size=10)))
    fig.update_xaxes(gridcolor="#e5e5e5")
    fig.update_yaxes(gridcolor="#e5e5e5")
    return fig


def main(tasks=None, labels=CLUSTER_LABELS, out_dir=OUTPUT_DIR, seed=SEED):
    tasks = load_tasks_with_clusters() if tasks is None else tasks
    embeddings, index = embed_tasks(tasks)
    xy = UMAP(n_components=2, random_state=seed, **UMAP_PARAMS).fit_transform(embeddings)[index]
    tasks = tasks.assign(X=xy[:, 0], Y=xy[:, 1])

    figures = [("fig10_all_tasks", tasks, f"Figure 10: {len(tasks)} tasks in {tasks.Cluster.nunique()} clusters")]
    for number, risk, reward in FIGURES:
        group = tasks[(tasks.Risk_Level == risk) & (tasks.Reward_Level == reward)]
        figures.append((f"fig{number}_{risk.lower()}_risk_{reward.lower()}_reward", group,
                        f"Figure {number}: {risk}-risk, {reward.lower()}-reward tasks (N={len(group)})"))

    out = out_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, data, title in figures:
        plot(data, title, labels).write_html(out / f"{name}.html", include_plotlyjs="cdn")
        print(f"Saved {out / name}.html")


if __name__ == "__main__":
    main()
