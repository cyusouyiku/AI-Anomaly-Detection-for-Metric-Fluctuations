# -*- coding: utf-8 -*-
"""步骤 3：合并聚类结果与原始样本，按簇调用 LLM 总结进线原因，输出三份模型监控报告"""

import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if _root not in sys.path:
        sys.path.insert(0, str(_root))
from config import (
    CLUSTERING_RESULT_FILE,
    EMBEDDED_DATA_FILE,
    CLUSTER_REPORT_2_11,
    CLUSTER_REPORT_2_12,
    CLUSTER_REPORT_DIFF,
    get_llm_client,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_SAMPLES_PER_CLUSTER,
)


def load_data(cluster_path=None, source_path=None):
    """加载聚类结果与 embedding 数据（含 content）"""
    cluster_path = cluster_path or CLUSTERING_RESULT_FILE
    source_path = source_path or EMBEDDED_DATA_FILE
    if not cluster_path.exists():
        raise FileNotFoundError(f"聚类结果不存在: {cluster_path}，请先运行聚类步骤")
    if not source_path.exists():
        raise FileNotFoundError(f"embedding 数据不存在: {source_path}，请先运行 embedding 步骤")
    cluster_df = pd.read_csv(cluster_path, encoding="utf-8-sig")
    source_df = pd.read_csv(source_path, encoding="utf-8-sig")
    return cluster_df, source_df


def merge_data(cluster_df, source_df):
    """按 id（或 callout_id）合并聚类结果与 embedding 数据"""
    join_key = "id" if "id" in cluster_df.columns and "id" in source_df.columns else "callout_id"
    if join_key not in cluster_df.columns or join_key not in source_df.columns:
        raise KeyError(f"两个表均需包含 {join_key} 列")
    cluster_cols = [join_key, "cluster"]
    for c in ("cluster_probability", "outlier_score"):
        if c in cluster_df.columns:
            cluster_cols.append(c)
    source_cols = [join_key, "content"]
    if "period" in source_df.columns:
        source_cols.append("period")
    merged = pd.merge(
        cluster_df[cluster_cols],
        source_df[source_cols],
        on=join_key,
        how="left",
    )
    if "period" in merged.columns:
        merged["period"] = merged["period"].apply(_norm_period)
    return merged


def _norm_period(p):
    """统一 period 为字符串 2.11 / 2.12（兼容 CSV 读成 float 的情况）"""
    if pd.isna(p):
        return "unknown"
    s = str(p).strip()
    if s in ("2.11", "2.12"):
        return s
    if s.startswith("2.11") or p == 2.11:
        return "2.11"
    if s.startswith("2.12") or p == 2.12:
        return "2.12"
    return s


def group_by_cluster_and_period(merged_df):
    """按 (cluster_id, period) 分组，返回 {(cluster, period): [convs]}"""
    groups = defaultdict(list)
    for _, row in merged_df.iterrows():
        if row["cluster"] == -1:
            continue
        period = _norm_period(row.get("period"))
        groups[(row["cluster"], period)].append({
            "id": row.get("id", row.get("callout_id")),
            "content": str(row.get("content", "无内容")) if pd.notna(row.get("content")) else "无内容",
            "period": period,
            "probability": row.get("cluster_probability"),
            "outlier_score": row.get("outlier_score"),
        })
    return dict(groups)


def summarize_cluster(client, cluster_id, conversations, period_label, max_samples=None):
    """调用 LLM 总结单个簇在指定日期的进线原因"""
    max_samples = max_samples or LLM_MAX_SAMPLES_PER_CLUSTER
    if conversations and conversations[0].get("probability") is not None:
        sample_convs = sorted(conversations, key=lambda x: x.get("probability") or 0, reverse=True)[:max_samples]
    else:
        sample_convs = conversations[:max_samples]

    prompt = f"""以下是模型监控分析中，第 {cluster_id} 号聚类簇在 **{period_label}** 这一天的客服对话样本。

请分析这些对话，总结该簇在当天的**共同进线原因**。

要求：
1. 输出为**编号列表**（1. 2. 3. …），每条一行，简洁概括一个进线原因
2. 按重要性排序，最多 3–5 条
3. 只关注客户核心诉求，用业务语言表达
4. 每条控制在 30 字以内

对话样本（共 {len(conversations)} 条，展示前 {len(sample_convs)} 条）：

"""
    for i, conv in enumerate(sample_convs, 1):
        content_preview = conv["content"][:400] if len(conv["content"]) > 400 else conv["content"]
        prompt += f"""
### 样本 {i}
**对话内容：**
{content_preview}

---
"""
    prompt += f"\n请基于以上样本，用编号列表格式输出簇 {cluster_id} 在 {period_label} 的**进线原因**（1. 2. 3. …）："

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的客服数据分析师，擅长从对话中提炼客户诉求。请用简洁的业务语言总结。"},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM调用失败: {str(e)}"


def analyze_by_period(client, grouped, period_val, period_label):
    """分析指定 period 的所有簇，返回 {cluster_id: {count, summary}}"""
    results = {}
    period_convs = {cid: convs for (cid, p), convs in grouped.items() if p == period_val}
    for cluster_id, conversations in sorted(period_convs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  正在分析 {period_label} 簇 {cluster_id}（共 {len(conversations)} 条）...")
        try:
            summary = summarize_cluster(client, cluster_id, conversations, period_label)
            results[cluster_id] = {"count": len(conversations), "summary": summary}
        except Exception as e:
            results[cluster_id] = {"count": len(conversations), "summary": f"分析失败: {str(e)}"}
    return results


def save_period_report(results, output_path, period_label, merged_df=None, period_val=None):
    """保存单日簇分析报告"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_df is not None and period_val is not None:
        sub = merged_df[merged_df["period"] == period_val]
        n_period = len(sub)
        n_in_clusters = int((sub["cluster"] != -1).sum())
        n_noise_period = n_period - n_in_clusters
    else:
        n_period = n_in_clusters = sum(r["count"] for r in results.values())
        n_noise_period = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"模型监控：{period_label} 改签场景进线主题分析\n")
        f.write("=" * 80 + "\n\n")
        f.write("【统计概览】\n")
        f.write(f"当日样本: {n_period}  |  簇内: {n_in_clusters}  |  噪声: {n_noise_period}\n\n")
        for cluster_id, info in sorted(results.items(), key=lambda x: x[1]["count"], reverse=True):
            f.write(f"\n【簇 {cluster_id}】\n")
            f.write(f"对话数量: {info['count']}\n")
            f.write(f"进线原因:\n{info['summary'].strip()}\n")
            f.write("-" * 80 + "\n")
    print(f"报告已保存: {output_path}")


def summarize_diff(client, new_clusters_info):
    """针对 2.12 新增簇（后天有、前一天没有）调用 LLM 总结关注点"""
    if not new_clusters_info:
        return "2.12 无新增进线主题（即无不曾在 2.11 出现的簇）。"

    lines = []
    for cid, info in sorted(new_clusters_info.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"簇 {cid}（2.12 共 {info['count']} 条）")
        lines.append(f"进线原因: {info['summary'][:300]}")
        lines.append("")

    prompt = f"""以下是 2.12（对比日）**新出现的进线主题**——这些主题在 2.11（基准日）没有进线，仅在 2.12 出现。

新增主题详情：
{chr(10).join(lines)}

请从**模型监控**视角，针对这些新增主题总结：
1. 这些新主题可能反映的业务变化或风险
2. 需要重点关注的方向或建议

输出为 2-3 段简洁总结。"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是模型监控分析师，擅长从数据中提炼关键变化与洞察。"},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM调用失败: {str(e)}"


def save_diff_report(diff_summary, output_path, results_211, results_212, merged_df):
    """保存差异总结报告：重点为 2.12 有、2.11 没有的簇"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(merged_df)
    n_noise = int((merged_df["cluster"] == -1).sum())
    n_211 = int((merged_df["period"] == "2.11").sum())
    n_212 = int((merged_df["period"] == "2.12").sum())

    # 2.12 有、2.11 没有的簇（后天新增的进线主题）
    new_clusters = {
        cid: results_212[cid]
        for cid in results_212
        if results_211.get(cid, {}).get("count", 0) == 0
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("模型监控：2.12 新增进线主题（后天有、前一天没有的簇）\n")
        f.write("=" * 80 + "\n\n")
        f.write("【统计概览】\n")
        f.write(f"2.11: {n_211} 条  |  2.12: {n_212} 条  |  新增主题簇数: {len(new_clusters)}\n\n")

        if new_clusters:
            f.write("【2.12 新增簇】前一天无进线，仅在 2.12 出现的主题：\n\n")
            for cid, info in sorted(new_clusters.items(), key=lambda x: -x[1]["count"]):
                f.write(f"簇 {cid}（2.12 共 {info['count']} 条）\n")
                f.write(f"进线原因:\n{info['summary'].strip()}\n")
                f.write("-" * 80 + "\n")
        else:
            f.write("【2.12 新增簇】无。所有 2.12 进线主题在 2.11 均已有出现。\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("【差异总结】\n")
        f.write("=" * 80 + "\n\n")
        f.write(diff_summary)
    print(f"报告已保存: {output_path}")


def run(cluster_path=None, source_path=None):
    """
    执行完整 LLM 分析流程，输出三份报告：
    - cluster_report_2.11.txt：前一天（基准日）的簇
    - cluster_report_2.12.txt：后一天（对比日）的簇
    - cluster_report_diff.txt：两期差异总结
    """
    cluster_df, source_df = load_data(cluster_path, source_path)
    merged_df = merge_data(cluster_df, source_df)
    if "period" not in merged_df.columns:
        raise ValueError("embedding 数据需包含 period 列（2.11 / 2.12）")

    grouped = group_by_cluster_and_period(merged_df)
    if not grouped:
        print("没有有效簇（仅噪声点），跳过 LLM 分析")
        return {}, None

    client = get_llm_client()

    # 1. 2.11 的簇
    print("\n[1/3] 分析 2.11（基准日）的簇...")
    results_211 = analyze_by_period(client, grouped, "2.11", "2.11（基准日）")
    save_period_report(results_211, CLUSTER_REPORT_2_11, "2.11（基准日）", merged_df, "2.11")

    # 2. 2.12 的簇
    print("\n[2/3] 分析 2.12（对比日）的簇...")
    results_212 = analyze_by_period(client, grouped, "2.12", "2.12（对比日）")
    save_period_report(results_212, CLUSTER_REPORT_2_12, "2.12（对比日）", merged_df, "2.12")

    # 3. 差异总结（聚焦 2.12 有、2.11 没有的簇）
    new_clusters = {cid: results_212[cid] for cid in results_212 if results_211.get(cid, {}).get("count", 0) == 0}
    print(f"\n[3/3] 生成差异总结（2.12 新增 {len(new_clusters)} 个主题簇）...")
    diff_summary = summarize_diff(client, new_clusters)
    save_diff_report(diff_summary, CLUSTER_REPORT_DIFF, results_211, results_212, merged_df)

    return {"2.11": results_211, "2.12": results_212, "diff": diff_summary}


if __name__ == "__main__":
    run()
