# -*- coding: utf-8 -*-
"""
AI 指标分析完整流水线 - 单文件合并版

流程：Embedding（多语种）→ 聚类 → LLM 总结 → 簇内容格式化

使用方式：
  python pipeline.py              # 运行全流程
  python pipeline.py --skip-llm   # 只做 embedding + 聚类 + 格式化（不调 LLM）
  python pipeline.py --step 1     # 只运行步骤 1（embedding）
  python pipeline.py --step 2     # 只运行步骤 2（聚类）
  python pipeline.py --step 3     # 只运行步骤 3（LLM）
  python pipeline.py --step 4     # 只运行步骤 4（格式化簇内容汇总）
"""

import argparse
import os
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


_load_env()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"


def _resolve_raw_data_file():
    base = os.environ.get("RAW_DATA_FILE", "raw.csv").strip()
    fn = Path(base).name or "raw.csv"
    candidates = [RAW_DIR / fn, PROJECT_ROOT / fn]
    if base != fn and Path(base).is_absolute():
        candidates.insert(0, Path(base))
    for p in candidates:
        if p.exists():
            return p
    return RAW_DIR / fn


RAW_DATA_FILE = _resolve_raw_data_file()
RAW_CONTENT_COLUMN = os.environ.get("RAW_CONTENT_COLUMN", "content")
EMBEDDED_DATA_FILE = PROCESSED_DIR / "embedded_data.csv"
CLUSTERING_RESULT_FILE = OUTPUT_DIR / "clustering_result.csv"
_default_periods = "12.10,12.11,12.12"
PERIOD_LABELS = tuple(p.strip() for p in os.environ.get("PERIOD_LABELS", _default_periods).split(",") if p.strip())
if not PERIOD_LABELS:
    PERIOD_LABELS = ("12.10", "12.11", "12.12")
PERIOD_BASELINE = os.environ.get("PERIOD_BASELINE", "").strip() or PERIOD_LABELS[0]
_default_display = {p: ("基日" if i == 0 else f"对比日{i}") for i, p in enumerate(PERIOD_LABELS)}
try:
    _env_names = os.environ.get("PERIOD_DISPLAY_NAMES")
    PERIOD_DISPLAY_NAMES = json.loads(_env_names) if _env_names else _default_display
except (json.JSONDecodeError, TypeError):
    PERIOD_DISPLAY_NAMES = _default_display
for p in PERIOD_LABELS:
    if p not in PERIOD_DISPLAY_NAMES:
        PERIOD_DISPLAY_NAMES[p] = p


def _report_path(period_key):
    return OUTPUT_DIR / f"cluster_report_{period_key.replace('.', '_')}.txt"


CLUSTER_REPORT_BY_PERIOD = {p: _report_path(p) for p in PERIOD_LABELS}
CLUSTER_REPORT_DIFF = OUTPUT_DIR / "cluster_report_diff.txt"

def _diff_pair_path(period_a, period_b):
    sa, sb = period_a.replace(".", "_"), period_b.replace(".", "_")
    return OUTPUT_DIR / f"cluster_report_diff_{sa}_vs_{sb}.txt"

CLUSTER_REPORT_DIFF_PAIRS = [
    (_diff_pair_path(PERIOD_LABELS[i], PERIOD_LABELS[i + 1]), PERIOD_LABELS[i], PERIOD_LABELS[i + 1])
    for i in range(len(PERIOD_LABELS) - 1)
]

N_CLUSTERS = 20
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ANOMALY_PCT_THRESHOLD = 5.0
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 500
LLM_MAX_SAMPLES_PER_CLUSTER = 10


def get_llm_client():
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# 步骤 1：Embedding

def _parse_period_from_createtime(createtime):
    if pd.isna(createtime):
        return None
    s = str(createtime).strip()
    m = re.search(r"(?:(\d{4})-)?(\d{1,2})-(\d{1,2})", s)
    if m:
        month = int(m.group(2)) if m.group(2).isdigit() else 0
        day = int(m.group(3)) if m.group(3).isdigit() else 0
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{month}.{day:02d}" if day < 10 else f"{month}.{day}"
    return None


def load_and_prepare_data(input_path=None):
    in_path = input_path or RAW_DATA_FILE
    if not in_path.exists():
        raise FileNotFoundError(f"原始数据不存在: {in_path}")
    df = pd.read_csv(in_path, encoding="utf-8-sig")
    content_col = None
    for c in (RAW_CONTENT_COLUMN, "content", "usermsg", "msg_dialogue", "看看"):
        if c in df.columns:
            content_col = c
            break
    if content_col is None:
        raise KeyError(f"数据需包含内容列")
    id_col = None
    for c in ("sessionid", "id", "callout_id"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise KeyError("数据需包含 id 列")
    if "createtime" not in df.columns:
        raise KeyError("数据需包含 createtime 列")
    df = df.copy()
    df["period"] = df["createtime"].apply(_parse_period_from_createtime)
    df["content"] = df[content_col].fillna("").astype(str)
    df["id"] = df[id_col].astype(str)
    valid = df["period"].isin(PERIOD_LABELS)
    df = df[valid].copy()
    if len(df) == 0:
        raise ValueError(f"createtime 解析后无有效 period（需为 {PERIOD_LABELS}）")
    return df[["id", "content", "period"]].copy()


def embed_texts(texts, model_name=None):
    cache_dir = PROJECT_ROOT / "data" / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]
    from sentence_transformers import SentenceTransformer
    model_name = model_name or EMBEDDING_MODEL
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 100)
    return np.array(embeddings, dtype=np.float64)


def run_embedding(input_path=None, output_path=None):
    ensure_dirs()
    out_path = output_path or EMBEDDED_DATA_FILE
    print("  加载原始数据...")
    df = load_and_prepare_data(input_path)
    print(f"  共 {len(df)} 条记录，period 分布: {df['period'].value_counts().to_dict()}")
    texts = df["content"].tolist()
    print(f"  使用多语种模型 {EMBEDDING_MODEL} 进行 embedding...")
    embeddings = embed_texts(texts)
    df["embedding_features"] = [f"[{','.join(map(str, emb))}]" for emb in embeddings]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {out_path}")
    return df, out_path


# 步骤 2：聚类

def parse_embedding(x):
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


def run_clustering(input_path=None, output_csv=None):
    ensure_dirs()
    in_path = input_path or EMBEDDED_DATA_FILE
    out_csv = output_csv or CLUSTERING_RESULT_FILE
    if not in_path.exists():
        raise FileNotFoundError(f"embedding 数据不存在: {in_path}")
    df = pd.read_csv(in_path, encoding="utf-8-sig")
    embeddings_series = df["embedding_features"].apply(parse_embedding)
    valid_mask = embeddings_series.notna()
    df = df[valid_mask].copy()
    embeddings = embeddings_series[valid_mask].tolist()
    if len(df) == 0:
        raise ValueError("没有有效的 embedding 数据")
    X = np.array(embeddings)
    X_normalized = normalize(X, norm="l2")
    distance_matrix = euclidean_distances(X_normalized)
    n_samples = len(X)
    if n_samples < 6:
        n_clusters = min(2, max(1, n_samples - 1))
        ac = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        cluster_labels = ac.fit_predict(distance_matrix)
        clusterer = None
    else:
        n_clusters_use = min(N_CLUSTERS, max(2, n_samples - 1))
        clusterer = KMeans(n_clusters=n_clusters_use, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(X_normalized)
    df["cluster"] = cluster_labels
    if clusterer is not None:
        if hasattr(clusterer, "transform"):
            dist_to_centroids = clusterer.transform(X_normalized)
            dist_to_own = np.min(dist_to_centroids, axis=1)
            df["cluster_probability"] = 1.0 / (1.0 + dist_to_own)
        else:
            df["cluster_probability"] = 1.0
    else:
        df["cluster_probability"] = 1.0
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"聚类完成: 总样本 {len(df)}，簇数 {n_clusters}，噪声点 {n_noise}")
    id_col = "id" if "id" in df.columns else "callout_id"
    result_cols = [id_col, "cluster", "cluster_probability"]
    df[result_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"聚类结果已保存: {out_csv}")
    return df, out_csv


# 步骤 3：LLM 分析

def _display_name(period):
    return PERIOD_DISPLAY_NAMES.get(period, period)


def _display_name_in_pair(period, period_a, period_b):
    """在 pairwise 比较中：period_a 为基日，period_b 为对比日"""
    if period == period_a:
        return "基日"
    if period == period_b:
        return "对比日"
    return PERIOD_DISPLAY_NAMES.get(period, period)


def _norm_period(p):
    """将 period 规范为 PERIOD_LABELS 中的字符串，兼容 float/str 及 12.1->12.10 等情形"""
    if pd.isna(p) or p == "" or (isinstance(p, str) and not p.strip()):
        return "unknown"
    s = str(p).strip()
    if s in PERIOD_LABELS:
        return s
    try:
        v = float(p)
        for pk in PERIOD_LABELS:
            try:
                pk_f = float(pk)
                if abs(v - pk_f) < 0.005:  # 12.1 匹配 12.10, 12.11/12.12 精确匹配
                    return pk
            except ValueError:
                pass
    except (ValueError, TypeError):
        pass
    for pk in PERIOD_LABELS:
        alt = pk.replace(".", "-")
        if pk in s or alt in s:
            return pk
    return s


def llm_load_data(cluster_path=None, source_path=None):
    cluster_path = cluster_path or CLUSTERING_RESULT_FILE
    source_path = source_path or EMBEDDED_DATA_FILE
    if not cluster_path.exists():
        raise FileNotFoundError(f"聚类结果不存在: {cluster_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"embedding 数据不存在: {source_path}")
    cluster_df = pd.read_csv(cluster_path, encoding="utf-8-sig")
    source_df = pd.read_csv(source_path, encoding="utf-8-sig")
    # 强制 period 为字符串，避免 pandas 把 12.10 读成 12.1、导致与 "12.10" 比较失败
    if "period" in source_df.columns:
        source_df = source_df.copy()
        source_df["period"] = source_df["period"].apply(
            lambda x: str(x).strip() if pd.notna(x) else ""
        )
    return cluster_df, source_df


def llm_merge_data(cluster_df, source_df):
    join_key = "id" if "id" in cluster_df.columns and "id" in source_df.columns else "callout_id"
    if join_key not in cluster_df.columns or join_key not in source_df.columns:
        raise KeyError(f"两个表均需包含 {join_key} 列")
    # 统一 id 类型确保 merge 能正确匹配（避免 int/str 混用导致对不上）
    cluster_df = cluster_df.copy()
    source_df = source_df.copy()
    cluster_df[join_key] = cluster_df[join_key].astype(str)
    source_df[join_key] = source_df[join_key].astype(str)
    cluster_cols = [join_key, "cluster"]
    for c in ("cluster_probability", "outlier_score"):
        if c in cluster_df.columns:
            cluster_cols.append(c)
    source_cols = [join_key, "content"]
    if "period" in source_df.columns:
        source_cols.append("period")
    merged = pd.merge(cluster_df[cluster_cols], source_df[source_cols], on=join_key, how="left")
    if "period" in merged.columns:
        merged["period"] = merged["period"].apply(_norm_period)
    return merged


def group_by_cluster_and_period(merged_df):
    groups = defaultdict(list)
    for _, row in merged_df.iterrows():
        if row["cluster"] == -1:
            continue
        period = _norm_period(row.get("period"))
        rid = row.get("id")
        if rid is None or (hasattr(rid, "__len__") and pd.isna(rid)):
            rid = row.get("callout_id")
        groups[(row["cluster"], period)].append({
            "id": rid, "callout_id": row.get("callout_id"),
            "content": str(row.get("content", "无内容")) if pd.notna(row.get("content")) else "无内容",
            "period": period, "probability": row.get("cluster_probability"),
        })
    return dict(groups)


def compute_proportions_and_anomalies_pair(merged_df, grouped, results_by_period, period_a, period_b, threshold_pct=None):
    """计算两期之间簇占比及异动（|pct_a - pct_b| >= threshold 则为异动）"""
    threshold_pct = threshold_pct or ANOMALY_PCT_THRESHOLD
    pair = (period_a, period_b)
    period_counts = merged_df.groupby("period").size()
    n_by_period = {p: period_counts.get(p, 0) for p in pair}
    cluster_period_counts = defaultdict(lambda: {p: 0 for p in pair})
    for (cid, p), convs in grouped.items():
        if p in pair:
            cluster_period_counts[cid][p] = len(convs)
    proportion_rows = []
    anomaly_clusters = {}
    for cid in sorted(cluster_period_counts.keys()):
        c = cluster_period_counts[cid]
        pcts = {p: 100 * c[p] / n_by_period.get(p, 1) if n_by_period.get(p) else 0 for p in pair}
        proportion_rows.append({"cluster": cid, **pcts})
        dev = abs(pcts[period_a] - pcts[period_b])
        info = results_by_period.get(period_a, {}).get(cid) or results_by_period.get(period_b, {}).get(cid) or {}
        if dev >= threshold_pct:
            suf = lambda x: x.replace(".", "_")
            anomaly_clusters[cid] = {
                **{f"pct_{suf(p)}": pcts[p] for p in pair},
                **{f"count_{suf(p)}": c[p] for p in pair},
                "max_dev": dev, "summary_reason": info.get("summary_reason", ""),
                "summary": info.get("summary", ""), "ids": info.get("ids", []),
            }
    proportion_df = pd.DataFrame(proportion_rows)
    all_clusters_info = {}
    for cid in sorted(cluster_period_counts.keys()):
        c = cluster_period_counts[cid]
        pcts = {p: 100 * c[p] / n_by_period.get(p, 1) if n_by_period.get(p) else 0 for p in pair}
        info = results_by_period.get(period_a, {}).get(cid) or results_by_period.get(period_b, {}).get(cid) or {}
        suf = lambda x: x.replace(".", "_")
        all_clusters_info[cid] = {
            **{f"pct_{suf(p)}": pcts[p] for p in pair},
            **{f"count_{suf(p)}": c[p] for p in pair},
            "summary_reason": info.get("summary_reason", ""),
            "summary": info.get("summary", ""),
            "reason_items": info.get("reason_items", []),
        }
    return proportion_df, anomaly_clusters, all_clusters_info


def compute_proportions_and_anomalies(merged_df, grouped, results_by_period, threshold_pct=None):
    threshold_pct = threshold_pct or ANOMALY_PCT_THRESHOLD
    period_counts = merged_df.groupby("period").size()
    n_by_period = {p: period_counts.get(p, 0) for p in PERIOD_LABELS}
    cluster_period_counts = defaultdict(lambda: {p: 0 for p in PERIOD_LABELS})
    for (cid, p), convs in grouped.items():
        if p in PERIOD_LABELS:
            cluster_period_counts[cid][p] = len(convs)
    proportion_rows = []
    anomaly_clusters = {}
    for cid in sorted(cluster_period_counts.keys()):
        c = cluster_period_counts[cid]
        pcts = {p: 100 * c[p] / n_by_period.get(p, 1) if n_by_period.get(p) else 0 for p in PERIOD_LABELS}
        pct_vals = [pcts[p] for p in PERIOD_LABELS]
        proportion_rows.append({"cluster": cid, **pcts})
        max_dev = max(pct_vals) - min(pct_vals)
        info = {}
        for p in PERIOD_LABELS:
            info = results_by_period.get(p, {}).get(cid) or info
        if max_dev >= threshold_pct:
            suf = lambda x: x.replace(".", "_")
            anomaly_clusters[cid] = {
                **{f"pct_{suf(p)}": pcts[p] for p in PERIOD_LABELS},
                **{f"count_{suf(p)}": c[p] for p in PERIOD_LABELS},
                "max_dev": max_dev, "summary_reason": info.get("summary_reason", ""),
                "summary": info.get("summary", ""), "ids": info.get("ids", []),
            }
    proportion_df = pd.DataFrame(proportion_rows)
    all_clusters_info = {}
    for cid in sorted(cluster_period_counts.keys()):
        c = cluster_period_counts[cid]
        pcts = {p: 100 * c[p] / n_by_period.get(p, 1) if n_by_period.get(p) else 0 for p in PERIOD_LABELS}
        info = {}
        for p in PERIOD_LABELS:
            info = results_by_period.get(p, {}).get(cid) or info
        suf = lambda x: x.replace(".", "_")
        all_clusters_info[cid] = {
            **{f"pct_{suf(p)}": pcts[p] for p in PERIOD_LABELS},
            **{f"count_{suf(p)}": c[p] for p in PERIOD_LABELS},
            "summary_reason": info.get("summary_reason", ""),
            "summary": info.get("summary", ""),
            "reason_items": info.get("reason_items", []),
        }
    return proportion_df, anomaly_clusters, all_clusters_info


def summarize_cluster(client, cluster_id, conversations, period_label, max_samples=None):
    """调用 LLM 总结单个簇在指定日期的进线原因"""
    max_samples = max_samples or LLM_MAX_SAMPLES_PER_CLUSTER
    if conversations and conversations[0].get("probability") is not None:
        sample_convs = sorted(conversations, key=lambda x: x.get("probability") or 0, reverse=True)[:max_samples]
    else:
        sample_convs = conversations[:max_samples]
    prompt = f"""以下是模型监控分析中，第 {cluster_id} 号聚类簇在**{period_label}**这一天的客服对话样本。
请分析这些对话，完成两项输出。

你的角色是AI指标异动检测模型，擅长从对话中抓取关键词并提炼具体因为特殊事件导致客人进线导致指标比例有波动，然后通过这些具体的对话来分析出指标异动的具体原因。

**务必从样本中抓取具体关键词**，避免笼统表述。

一、簇的总体原因（用一句话概括该簇进线主题，25 字以内）。必须包含样本中出现的**具体触发因素**——直接从样本中提取，禁止臆测。禁止使用「航班变动」「退改签咨询」等笼统词汇。

二、进线原因（具体）：编号列表（1. 2. 3. …），每条必须引用样本中的**原词或关键信息**，能对应到真实对话。按重要性排序，最多 3～5 条，每条 35 字以内。**每条后必须注明该条在簇内的大致占比**，格式为（约X%），各条占比之和约 100%。

请严格按以下格式输出（先总体原因，再空行，再具体原因）：
---
簇的总体原因：（含具体关键词，30字数以内）

进线原因：
1. …（约X%）
2. …（约X%）
---
对话样本（共 {len(conversations)} 条，展示前 {len(sample_convs)} 条）：
"""
    for i, conv in enumerate(sample_convs, 1):
        content_preview = conv["content"][:400] if len(conv["content"]) > 400 else conv["content"]
        prompt += f"\n### 样本 {i}\n**对话内容：**\n{content_preview}\n---\n"
    prompt += f"\n请基于以上样本，按格式输出簇 {cluster_id} 在 {period_label} 的**簇的总体原因**与**进线原因**。务必从样本中抓取具体关键词，禁止笼统表述。**全文必须以中文输出，若引用日文/英文等外语，须翻译为中文。**\n"
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是客服数据分析师，擅长从对话中抓取关键词并提炼具体原因。必须从样本中提取具体信息（地域、灾难事件、产品、航司、航班、日期等），禁止臆测或使用笼统词如「航班变动」「退改签咨询」，要用样本中的原词。输出全文必须为中文，如有日文/英文等外语须翻译成中文。"},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return _parse_cluster_summary(response.choices[0].message.content.strip())
    except Exception as e:
        return {"summary_reason": "", "summary": f"LLM调用失败: {str(e)}", "reason_items": []}


def _parse_cluster_summary(raw):
    summary_reason = ""
    summary = raw
    reason_items = []
    for line in raw.split("\n"):
        line = line.strip()
        if "簇的总体原因" in line or ("总体原因" in line and "进线" not in line):
            part = line.replace("簇的总体原因", "").replace("总体原因", "").strip("：（）。() \t")
            if part and len(part) <= 80:
                summary_reason = part
            break
    if "进线原因" in raw:
        idx = raw.find("进线原因")
        rest = raw[idx:].split("\n", 1)[-1].strip() if "\n" in raw[idx:] else raw[idx:]
        summary = rest.strip("：\n")
        for line in rest.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+\.\s*(.+)$", line)
            if m:
                line_text = m.group(1).strip()
                pct_match = re.search(r"（约\s*(\d+(?:\.\d+)?)\s*%?\s*）", line_text)
                desc = re.sub(r"\s*（约\s*\d+(?:\.\d+)?\s*%?\s*）\s*", "", line_text).strip()
                reason_items.append((desc, pct_match.group(1) if pct_match else None))
    return {"summary_reason": (summary_reason or "")[:80], "summary": summary or raw, "reason_items": reason_items}


def analyze_by_period(client, grouped, period_val, period_label):
    results = {}
    period_convs = {cid: convs for (cid, p), convs in grouped.items() if p == period_val}
    for cluster_id, conversations in sorted(period_convs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  正在分析 {period_label} 簇 {cluster_id}（共 {len(conversations)} 条）...")
        try:
            out = summarize_cluster(client, cluster_id, conversations, period_label)
            ids = [str(int(c["id"])) if isinstance(c.get("id"), float) and c.get("id") == int(c.get("id")) else str(c.get("id", "")) for c in conversations if c.get("id") is not None and pd.notna(c.get("id"))]
            ids = [x for x in ids if x]
            results[cluster_id] = {
                "count": len(conversations),
                "summary_reason": out.get("summary_reason", ""),
                "summary": out.get("summary", ""),
                "reason_items": out.get("reason_items", []),
                "ids": ids,
            }
        except Exception as e:
            results[cluster_id] = {"count": len(conversations), "summary_reason": "", "summary": f"分析失败: {str(e)}", "reason_items": [], "ids": []}
    return results


def save_period_report(results, output_path, period_label, merged_df=None, period_val=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_df is not None and period_val is not None:
        sub = merged_df[merged_df["period"] == period_val]
        n_period, n_in_clusters = len(sub), int((sub["cluster"] != -1).sum())
        n_noise_period = n_period - n_in_clusters
    else:
        n_period = n_in_clusters = sum(r["count"] for r in results.values())
        n_noise_period = 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"模型监控：{period_label} 改签场景进线主题分析\n")
        f.write("=" * 80 + "\n\n【统计概览】\n")
        f.write(f"当日样本: {n_period}  |  簇内: {n_in_clusters}  |  噪声: {n_noise_period}\n\n")
        for cluster_id, info in sorted(results.items(), key=lambda x: x[1]["count"], reverse=True):
            f.write(f"\n【簇 {cluster_id}】\n对话数量: {info['count']}\n")
            if info.get("ids"):
                f.write(f"id: {', '.join(info['ids'])}\n")
            if info.get("summary_reason"):
                f.write(f"簇的总体原因: {info['summary_reason'].strip()}\n")
            f.write(f"进线原因:\n{info['summary'].strip()}\n")
            f.write("-" * 80 + "\n")
    print(f"报告已保存: {output_path}")


def summarize_diff_pair(client, anomaly_clusters_info, proportion_table_text, period_a, period_b, threshold_pct=None):
    """对两期对比生成 LLM 异动汇总。period_a 为基日，period_b 为对比日"""
    threshold_pct = threshold_pct or ANOMALY_PCT_THRESHOLD
    if not anomaly_clusters_info:
        return f"基日→对比日：各簇占比波动均在 {threshold_pct}% 以内，未检测到异动。"
    lines = []
    for cid, info in sorted(anomaly_clusters_info.items(), key=lambda x: -x[1]["max_dev"]):
        lines.append(f"簇{cid}：{_fmt_period_stats(info, [period_a, period_b], period_a, period_b)}")
        if info.get("summary_reason"):
            lines.append(f"  簇的总体原因: {info['summary_reason']}")
        lines.append(f"  进线原因: {info.get('summary', '')[:300]}")
        lines.append("")
    name_a, name_b = "基日", "对比日"
    prompt = f"""以下是统一聚类后，{name_a}→{name_b} 各簇占比变化。占比差异≥{threshold_pct}% 视为异动。
各簇两期占比：\n{proportion_table_text}\n\n异动簇详情：\n{chr(10).join(lines)}

请从模型监控视角，完成：一、{name_a}→{name_b} 异动原因汇总（3～5条）；二、业务变化与风险；三、建议关注方向。全文必须中文。"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是模型监控分析师，擅长从异动中识别具体触发因素。输出全文必须为中文。"},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM调用失败: {str(e)}"


def summarize_diff(client, anomaly_clusters_info, proportion_table_text, threshold_pct=None):
    threshold_pct = threshold_pct or ANOMALY_PCT_THRESHOLD
    if not anomaly_clusters_info:
        return f"三期各簇占比波动均在 {threshold_pct}% 以内，未检测到异动。"
    lines = []
    for cid, info in sorted(anomaly_clusters_info.items(), key=lambda x: -x[1]["max_dev"]):
        lines.append(f"簇{cid}：{_fmt_period_stats(info)}")
        if info.get("summary_reason"):
            lines.append(f"  簇的总体原因: {info['summary_reason']}")
        lines.append(f"  进线原因: {info.get('summary', '')[:300]}")
        lines.append("")
    period_names = " / ".join(_display_name(p) for p in PERIOD_LABELS)
    prompt = f"""以下是统一聚类后各簇在三期（{period_names}）的占比数据。占比差异大（波动≥{threshold_pct}%）则为异动。
各簇三期占比：\n{proportion_table_text}\n\n异动簇详情：\n{chr(10).join(lines)}

请从模型监控视角，完成：一、异动原因汇总（3～5条）；二、业务变化与风险；三、建议关注方向。全文必须中文。"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是模型监控分析师，擅长从异动中识别具体触发因素。输出全文必须为中文。"},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM调用失败: {str(e)}"


def _fmt_period_stats(info, periods=None, period_a=None, period_b=None):
    periods = periods or PERIOD_LABELS
    if period_a is not None and period_b is not None:
        name_fn = lambda p: _display_name_in_pair(p, period_a, period_b)
    else:
        name_fn = _display_name
    parts = []
    for p in periods:
        suf = p.replace('.', '_')
        parts.append(f"{name_fn(p)} {info.get(f'pct_{suf}', 0):.2f}%({info.get(f'count_{suf}', 0)})")
    return " | ".join(parts)


def save_diff_report_pair(diff_summary, output_path, anomaly_clusters, proportion_df, merged_df, all_clusters_info, period_a, period_b):
    """保存两期对比的波动异动报告。period_a 为基日，period_b 为对比日"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair = (period_a, period_b)
    merged_period = merged_df["period"].astype(str)
    period_counts = {p: int((merged_period == p).sum()) for p in pair}
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"模型监控：基日 → 对比日 进线主题对比 · 占比波动异动汇总\n")
        f.write(f"（基日={period_a}，对比日={period_b}）\n")
        f.write("=" * 80 + "\n\n【统计概览】\n")
        f.write(f"基日: {period_counts.get(period_a, 0)} 条 | 对比日: {period_counts.get(period_b, 0)} 条")
        f.write(f" |  异动簇数: {len(anomaly_clusters)}\n\n【各簇两期占比】\n")
        if proportion_df is not None and len(proportion_df) > 0:
            fmt_df = proportion_df.copy()
            for p in pair:
                if p in fmt_df.columns:
                    col_name = "基日" if p == period_a else "对比日"
                    fmt_df[col_name] = fmt_df[p].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
                    fmt_df = fmt_df.drop(columns=[p])
            f.write(fmt_df.to_string(index=False) + "\n\n")
        f.write(f"【各簇进线原因】\n\n")
        for cid in sorted(all_clusters_info.keys()):
            info = all_clusters_info[cid]
            f.write(f"簇 {cid}：{_fmt_period_stats(info, pair, period_a, period_b)}\n")
            if info.get("summary_reason"):
                f.write(f"  簇的总体原因: {info['summary_reason'].strip()}\n")
            if info.get("reason_items"):
                for desc, pct in info["reason_items"]:
                    f.write(f"    · {desc}{'（' + pct + '%）' if pct else '（未标注）'}\n")
            else:
                f.write(f"  {info.get('summary', '')[:400]}\n")
            f.write("-" * 80 + "\n")
        f.write("\n【异动簇】\n\n")
        for cid, info in sorted(anomaly_clusters.items(), key=lambda x: -x[1]["max_dev"]):
            f.write(f"簇{cid}：{_fmt_period_stats(info, pair, period_a, period_b)}\n")
            f.write(f"  进线原因:\n  {info.get('summary', '')[:400]}\n")
            f.write("-" * 80 + "\n")
        f.write("\n【异动原因汇总】\n" + "=" * 80 + "\n\n")
        f.write(diff_summary)
    print(f"报告已保存: {output_path}")


def save_diff_report(diff_summary, output_path, anomaly_clusters, proportion_df, merged_df, all_clusters_info):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_period = merged_df["period"].astype(str)
    period_counts = {p: int((merged_period == p).sum()) for p in PERIOD_LABELS}
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"模型监控：{' vs '.join(_display_name(p) for p in PERIOD_LABELS)} 进线主题对比 · 占比波动异动汇总\n")
        f.write("=" * 80 + "\n\n【统计概览】\n")
        f.write(" | ".join(f"{_display_name(p)}: {period_counts.get(p, 0)} 条" for p in PERIOD_LABELS))
        f.write(f" |  异动簇数: {len(anomaly_clusters)}\n\n【各簇三期占比】\n")
        if proportion_df is not None and len(proportion_df) > 0:
            fmt_df = proportion_df.copy()
            for p in PERIOD_LABELS:
                if p in fmt_df.columns:
                    fmt_df[_display_name(p)] = fmt_df[p].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
                    fmt_df = fmt_df.drop(columns=[p])
            f.write(fmt_df.to_string(index=False) + "\n\n")
        f.write(f"【各簇进线原因】\n\n")
        for cid in sorted(all_clusters_info.keys()):
            info = all_clusters_info[cid]
            f.write(f"簇 {cid}：{_fmt_period_stats(info)}\n")
            if info.get("summary_reason"):
                f.write(f"  簇的总体原因: {info['summary_reason'].strip()}\n")
            if info.get("reason_items"):
                for desc, pct in info["reason_items"]:
                    f.write(f"    · {desc}{'（' + pct + '%）' if pct else '（未标注）'}\n")
            else:
                f.write(f"  {info.get('summary', '')[:400]}\n")
            f.write("-" * 80 + "\n")
        f.write("\n【异动簇】\n\n")
        for cid, info in sorted(anomaly_clusters.items(), key=lambda x: -x[1]["max_dev"]):
            f.write(f"簇{cid}：{_fmt_period_stats(info)}\n")
            f.write(f"  进线原因:\n  {info.get('summary', '')[:400]}\n")
            f.write("-" * 80 + "\n")
        f.write("\n【异动原因汇总】\n" + "=" * 80 + "\n\n")
        f.write(diff_summary)
    print(f"报告已保存: {output_path}")


def run_llm(cluster_path=None, source_path=None):
    cluster_df, source_df = llm_load_data(cluster_path, source_path)
    merged_df = llm_merge_data(cluster_df, source_df)
    if "period" not in merged_df.columns:
        raise ValueError("embedding 数据需包含 period 列")
    period_dist = merged_df["period"].value_counts().to_dict()
    print(f"  merged 后 period 分布: {period_dist}")
    unknown_cnt = (merged_df["period"] == "unknown").sum()
    if unknown_cnt > 0:
        print(f"  警告: {unknown_cnt} 条 id 未匹配到 period，可能 clustering 与 embedding 数据不一致，请重新跑全流程")
    grouped = group_by_cluster_and_period(merged_df)
    if not grouped:
        print("没有有效簇，跳过 LLM 分析")
        return {}, None
    client = get_llm_client()
    results_by_period = {}
    n_total = len(PERIOD_LABELS) + 1
    for i, period in enumerate(PERIOD_LABELS):
        dname = _display_name(period)
        label = f"{dname}（基准日）" if period == PERIOD_BASELINE else f"{dname}（对比日）"
        print(f"\n[{i + 1}/{n_total}] 分析 {label} 的簇...")
        res = analyze_by_period(client, grouped, period, label)
        results_by_period[period] = res
        save_period_report(res, CLUSTER_REPORT_BY_PERIOD[period], label, merged_df, period)
    # 相邻两期对比：12.10→12.11、12.11→12.12 各生成一份波动异动报告
    diff_summaries = []
    for idx, (out_path, period_a, period_b) in enumerate(CLUSTER_REPORT_DIFF_PAIRS):
        print(f"\n[{n_total + idx}/{n_total + len(CLUSTER_REPORT_DIFF_PAIRS)}] 生成 {period_a}→{period_b}（基日→对比日）波动异动报告...")
        proportion_df_p, anomaly_p, all_info_p = compute_proportions_and_anomalies_pair(
            merged_df, grouped, results_by_period, period_a, period_b
        )
        if proportion_df_p is not None and len(proportion_df_p) > 0:
            fmt_pct = proportion_df_p.copy()
            for p in (period_a, period_b):
                if p in fmt_pct.columns:
                    fmt_pct[_display_name(p)] = fmt_pct[p].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
                    fmt_pct = fmt_pct.drop(columns=[p])
            proportion_table_text = fmt_pct.to_string(index=False)
        else:
            proportion_table_text = "（无）"
        diff_summary = summarize_diff_pair(client, anomaly_p, proportion_table_text, period_a, period_b)
        save_diff_report_pair(diff_summary, out_path, anomaly_p, proportion_df_p, merged_df, all_info_p, period_a, period_b)
        diff_summaries.append(diff_summary)
    return {**results_by_period, "diff": diff_summaries}


# 步骤 4：簇内容格式化

def _period_from_filename(path):
    m = re.search(r"cluster_report_(.+)", path.stem)
    return m.group(1).replace("_", ".") if m else None


def format_parse_report(path, period=None):
    text = path.read_text(encoding="utf-8-sig")
    period = period or _period_from_filename(path) or PERIOD_LABELS[0]
    clusters = []
    parts = re.split(r"【簇 (\d+)】", text)
    for i in range(1, len(parts), 2):
        cid = int(parts[i])
        block = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.match(r"\s*对话数量:\s*(\d+)\s+", block)
        if not m:
            continue
        count = int(m.group(1))
        reason = ""
        if "簇的总体原因:" in block or "簇的总原因:" in block:
            r = re.search(r"簇的总体?原因:\s*(.+?)(?=\n|$)", block)
            if r:
                reason = r.group(1).strip()
        ids_line = ""
        if "id:" in block:
            ids_m = re.search(r"id:\s*(.+?)(?=\n簇的总体?原因|\n进线原因:|\n---|\Z)", block, re.DOTALL)
            if ids_m:
                ids_line = ids_m.group(1).strip().replace("\n", " ")[:300]
        details = ""
        if "进线原因:" in block:
            d = re.search(r"进线原因:\s*\n(.+?)(?=\n---{20,}|\Z)", block, re.DOTALL)
            if d:
                details = d.group(1).strip()
        clusters.append((cid, count, period, ids_line, reason, details))
    return clusters


def _parse_diff_report(path, period_a=None, period_b=None):
    """从 cluster_report_diff*.txt 解析各簇数量与主题。pairwise 时传 period_a,period_b 以正确映射基日/对比日"""
    text = path.read_text(encoding="utf-8-sig")
    if period_a is not None and period_b is not None:
        display_to_period = {"基日": period_a, "对比日": period_b}
    else:
        display_to_period = {v: k for k, v in PERIOD_DISPLAY_NAMES.items()}
    clusters_data = []
    idx = text.find("【各簇进线原因】")
    if idx < 0:
        return clusters_data
    block = text[idx:]
    for m in re.finditer(r"簇\s*(\d+)[：:]\s*(.+?)(?=\n簇\s+\d+[：:]|\n【异动簇】|\Z)", block, re.DOTALL):
        cid = int(m.group(1))
        head_line = m.group(2).split("\n")[0]
        body = "\n".join(m.group(2).split("\n")[1:]) if "\n" in m.group(2) else ""
        period_counts = {}
        for pm in re.finditer(r"([^\s|]+)\s+[\d.]+%\((\d+)\)", head_line):
            dname, cnt = pm.group(1).strip(), int(pm.group(2))
            if dname in display_to_period:
                period_counts[display_to_period[dname]] = cnt
        reason = ""
        r = re.search(r"簇的总体?原因:\s*(.+?)(?=\n|$)", body)
        if r:
            reason = r.group(1).strip()
        details_lines = []
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("·") or line.startswith("•"):
                details_lines.append(line.lstrip("·• \t"))
        details = "\n".join(details_lines) if details_lines else ""
        clusters_data.append((cid, period_counts, reason, details))
    return clusters_data


def run_format():
    def _init():
        d = {f"c{p.replace('.', '_')}": 0 for p in PERIOD_LABELS}
        d.update({"ids": "", "reason": "", "details": "", "best_count": 0})
        return d
    by_cluster = defaultdict(_init)
    seen_periods = set()
    for period, report_path in CLUSTER_REPORT_BY_PERIOD.items():
        if report_path.exists() and period not in seen_periods:
            seen_periods.add(period)
            col = f"c{period.replace('.', '_')}"
            for cid, count, _, ids, reason, details in format_parse_report(report_path, period):
                by_cluster[cid][col] = count
                if count > by_cluster[cid]["best_count"]:
                    by_cluster[cid]["best_count"] = count
                    by_cluster[cid]["ids"] = ids
                    by_cluster[cid]["reason"] = reason
                    by_cluster[cid]["details"] = details
    for name, idx in [("baseline", 0), ("compare1", 1), ("compare2", 2)]:
        if idx < len(PERIOD_LABELS):
            path = OUTPUT_DIR / f"cluster_report_{name}.txt"
            if path.exists() and PERIOD_LABELS[idx] not in seen_periods:
                period = PERIOD_LABELS[idx]
                seen_periods.add(period)
                col = f"c{period.replace('.', '_')}"
                for cid, count, _, ids, reason, details in format_parse_report(path, period):
                    by_cluster[cid][col] = count
                    if count > by_cluster[cid]["best_count"]:
                        by_cluster[cid]["best_count"] = count
                        by_cluster[cid]["ids"] = ids
                        by_cluster[cid]["reason"] = reason
                        by_cluster[cid]["details"] = details
    if not by_cluster:
        merged_from_pairs = {}
        if CLUSTER_REPORT_DIFF.exists():
            for cid, period_counts, reason, details in _parse_diff_report(CLUSTER_REPORT_DIFF):
                merged_from_pairs[cid] = (period_counts, reason, details)
        else:
            for out_path, pa, pb in CLUSTER_REPORT_DIFF_PAIRS:
                if out_path.exists():
                    for cid, period_counts, reason, details in _parse_diff_report(out_path, pa, pb):
                        if cid not in merged_from_pairs:
                            merged_from_pairs[cid] = ({}, reason, details)
                        pc, _, _ = merged_from_pairs[cid]
                        for p, cnt in period_counts.items():
                            pc[p] = cnt
        for cid, (period_counts, reason, details) in merged_from_pairs.items():
            for period, count in period_counts.items():
                col = f"c{period.replace('.', '_')}"
                by_cluster[cid][col] = count
            total_c = sum(period_counts.values())
            if total_c > by_cluster[cid]["best_count"]:
                by_cluster[cid]["best_count"] = total_c
                by_cluster[cid]["reason"] = reason
                by_cluster[cid]["details"] = details
    def total(c):
        return sum(c.get(f"c{p.replace('.', '_')}", 0) for p in PERIOD_LABELS)
    sorted_clusters = sorted(by_cluster.items(), key=lambda x: -total(x[1]))
    lines = ["=" * 80, "簇内容汇总（统一聚合，每个簇各期数量分布 + 主题摘要）", "=" * 80, ""]
    for i, (cid, d) in enumerate(sorted_clusters, 1):
        parts = [f"{PERIOD_DISPLAY_NAMES.get(p, p)} 共 {d.get('c' + p.replace('.', '_'), 0)} 条" for p in PERIOD_LABELS]
        lines.extend([f"--- Top{i} ---", f"簇 {cid}（{' | '.join(parts)}）"])
        if d["ids"]:
            lines.append(f"id: {d['ids']}")
        if d["reason"]:
            lines.append(f"簇的总体原因: {d['reason']}")
        lines.extend(["进线原因:", d["details"], "---", "-" * 80, ""])
    out = OUTPUT_DIR / "cluster_content_formatted.txt"
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"已生成: {out}")


# 主入口

def main():
    parser = argparse.ArgumentParser(description="AI 指标分析完整流水线")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="只运行指定步骤（1=embedding 2=聚类 3=LLM 4=格式化）")
    parser.add_argument("--skip-llm", action="store_true", help="全流程但跳过步骤 3（不调用 LLM）")
    args = parser.parse_args()

    ensure_dirs()
    run_s1 = run_s2 = run_s3 = run_s4 = False
    if args.step:
        run_s1, run_s2, run_s3, run_s4 = args.step == 1, args.step == 2, args.step == 3, args.step == 4
    else:
        run_s1 = run_s2 = True
        run_s3 = not args.skip_llm
        run_s4 = True

    print("=" * 60)
    print("AI 指标分析流水线（模型监控：基日 / 对比日一 / 对比日二）")
    print("=" * 60)

    if run_s1:
        print("\n[步骤 1/4] Embedding（多语种）...")
        run_embedding()
        print("步骤 1 完成。\n")

    if run_s2:
        print("\n[步骤 2/4] 聚类...")
        run_clustering()
        print("步骤 2 完成。\n")

    if run_s3:
        print("\n[步骤 3/4] LLM 分析（进线主题）...")
        run_llm()
        print("步骤 3 完成。\n")

    if run_s4:
        print("\n[步骤 4/4] 簇内容格式化...")
        run_format()
        print("步骤 4 完成。\n")

    print("=" * 60)
    print("流水线执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
