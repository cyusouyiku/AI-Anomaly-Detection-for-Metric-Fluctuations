# -*- coding: utf-8 -*-
"""步骤 2：读取清洗后的 embedding，HDBSCAN 聚类，输出聚类结果与可视化"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if _root not in sys.path:
        sys.path.insert(0, str(_root))
from config import (
    EMBEDDED_DATA_FILE,
    CLUSTERING_RESULT_FILE,
    CLUSTERING_PLOT_FILE,
    ensure_dirs,
    CLUSTERING_MIN_CLUSTER_SIZE,
    CLUSTERING_MIN_SAMPLES,
    CLUSTERING_METRIC,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_embedding(x):
    """将 embedding 字符串解析为 numpy 数组，无效则返回 None"""
    x_str = str(x).strip()
    if x_str.startswith("["):
        x_str = x_str[1:]
    if x_str.endswith("]"):
        x_str = x_str[:-1]
    parts = [i.strip() for i in x_str.split(",") if i.strip()]
    if not parts:
        return None
    try:
        return np.array([float(i) for i in parts])
    except ValueError:
        return None


def run(input_path=None, output_csv=None, output_plot=None, show_plot=False):
    """
    执行聚类：解析 embedding → 归一化 → 距离矩阵 → HDBSCAN → 保存结果与图表。
    :return: (带 cluster 等列的 DataFrame, 结果 CSV 路径, 图表路径)
    """
    ensure_dirs()
    in_path = input_path or EMBEDDED_DATA_FILE
    out_csv = output_csv or CLUSTERING_RESULT_FILE
    out_plot = output_plot or CLUSTERING_PLOT_FILE

    if not in_path.exists():
        raise FileNotFoundError(f"embedding 数据不存在: {in_path}，请先运行 embedding 步骤")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    embeddings_series = df["embedding_features"].apply(parse_embedding)
    valid_mask = embeddings_series.notna()
    df = df[valid_mask].copy()
    embeddings = embeddings_series[valid_mask].tolist()

    if len(df) == 0:
        raise ValueError("没有有效的 embedding 数据，请检查 embedding_features 格式")

    X = np.array(embeddings)
    X_normalized = normalize(X, norm="l2")
    distance_matrix = cosine_distances(X_normalized)

    n_samples = len(X)
    if n_samples < 6:
        # 极少样本时 HDBSCAN 易全标为噪声，改用层次聚类保证有簇
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = min(2, max(1, n_samples - 1))
        ac = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        cluster_labels = ac.fit_predict(distance_matrix)
        clusterer = None
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=CLUSTERING_MIN_CLUSTER_SIZE,
            min_samples=CLUSTERING_MIN_SAMPLES,
            metric=CLUSTERING_METRIC,
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
    df["cluster"] = cluster_labels

    if clusterer is not None:
        if hasattr(clusterer, "probabilities_"):
            df["cluster_probability"] = clusterer.probabilities_
        if hasattr(clusterer, "outlier_scores_"):
            df["outlier_score"] = clusterer.outlier_scores_

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"聚类完成: 总样本 {len(df)}，簇数 {n_clusters}，噪声点 {n_noise}")

    # 可视化
    _save_plots(df, X_normalized, cluster_labels, clusterer or object(), out_plot)
    if show_plot:
        plt.show()
    plt.close("all")

    # 保存结果 CSV（使用 id 作为行标识）
    id_col = "id" if "id" in df.columns else "callout_id"
    result_cols = [id_col, "cluster"]
    if "cluster_probability" in df.columns:
        result_cols.append("cluster_probability")
    if "outlier_score" in df.columns:
        result_cols.append("outlier_score")
    df[result_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"聚类结果已保存: {out_csv}")

    return df, out_csv, out_plot


def _save_plots(df, X_normalized, cluster_labels, clusterer, out_plot):
    """生成 2x3 子图并保存"""
    fig = plt.figure(figsize=(16, 10))

    # 1. PCA 2D
    ax1 = fig.add_subplot(2, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_normalized)
    scatter = ax1.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=cluster_labels, cmap="tab10", s=50, alpha=0.6,
        edgecolors="black", linewidth=0.5,
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("PCA 2D Projection")
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Cluster ID")

    # 2. 簇大小分布
    ax2 = fig.add_subplot(2, 3, 2)
    cluster_counts = df[df["cluster"] != -1]["cluster"].value_counts().sort_index()
    if len(cluster_counts) > 0:
        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color="steelblue", alpha=0.7, edgecolor="black")
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}",
                     ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Cluster Size Distribution")
    ax2.grid(axis="y", alpha=0.3)

    # 3. 饼图
    ax3 = fig.add_subplot(2, 3, 3)
    sizes = df["cluster"].value_counts()
    labels = [f"Cluster {i}\n({sizes[i]} samples)" if i != -1 else f"Noise\n({sizes[i]} samples)" for i in sizes.index]
    colors = plt.cm.tab10(range(len(sizes)))
    ax3.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax3.set_title("Cluster Proportion")

    # 4. 聚类概率箱线图
    if "cluster_probability" in df.columns and len(cluster_counts) > 0:
        ax4 = fig.add_subplot(2, 3, 4)
        cluster_data = df[df["cluster"] != -1]
        cids = sorted(cluster_data["cluster"].unique())
        data_to_plot = [cluster_data[cluster_data["cluster"] == i]["cluster_probability"].values for i in cids]
        ax4.boxplot(data_to_plot, labels=cids)
        ax4.set_xlabel("Cluster ID")
        ax4.set_ylabel("Cluster Probability")
        ax4.set_title("Cluster Probability Distribution")
        ax4.grid(axis="y", alpha=0.3)

    # 5. 异常值分数
    if "outlier_score" in df.columns:
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(df["outlier_score"], bins=30, color="coral", alpha=0.7, edgecolor="black")
        ax5.axvline(df["outlier_score"].mean(), color="red", linestyle="--", label=f"Mean: {df['outlier_score'].mean():.3f}")
        ax5.set_xlabel("Outlier Score")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Outlier Score Distribution")
        ax5.legend()
        ax5.grid(axis="y", alpha=0.3)

    # 6. 簇质量排名
    ax6 = fig.add_subplot(2, 3, 6)
    if "cluster_probability" in df.columns and len(cluster_counts) > 0:
        stats = df[df["cluster"] != -1].groupby("cluster")["cluster_probability"].mean().sort_values(ascending=False)
        if len(stats) > 0:
            ax6.barh(range(len(stats)), stats.values, color="lightgreen", alpha=0.7, edgecolor="black")
            ax6.set_yticks(range(len(stats)))
            ax6.set_yticklabels([f"Cluster {i}" for i in stats.index])
    ax6.set_xlabel("Average Probability")
    ax6.set_title("Cluster Quality Ranking")
    ax6.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"可视化已保存: {out_plot}")


if __name__ == "__main__":
    run(show_plot=False)
