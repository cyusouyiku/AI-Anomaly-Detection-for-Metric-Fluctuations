# -*- coding: utf-8 -*-
"""项目配置：路径与运行参数统一管理"""

import os
from pathlib import Path

# 项目根目录（本文件所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------- 数据路径 ----------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# 输入：2.11/2.12 两期改签数据（用于模型监控对比）
RAW_DATA_2_11 = RAW_DIR / "2.11改签.csv"
RAW_DATA_2_12 = RAW_DIR / "2.12改签.csv"
# 合并 + embedding 后的数据（供聚类使用）
EMBEDDED_DATA_FILE = PROCESSED_DIR / "embedded_data.csv"
# 聚类结果 CSV
CLUSTERING_RESULT_FILE = OUTPUT_DIR / "clustering_result.csv"
# 聚类可视化图
CLUSTERING_PLOT_FILE = OUTPUT_DIR / "clustering_analysis.png"
# LLM 分析报告（模型监控：三份文档）
CLUSTER_REPORT_2_11 = OUTPUT_DIR / "cluster_report_2.11.txt"   # 前一天（基准日）的簇
CLUSTER_REPORT_2_12 = OUTPUT_DIR / "cluster_report_2.12.txt"   # 后一天（对比日）的簇
CLUSTER_REPORT_DIFF = OUTPUT_DIR / "cluster_report_diff.txt"    # 两期差异总结

# ---------- LLM 配置（从环境变量读取，勿提交密钥到仓库）----------
def get_llm_client():
    """延迟初始化 LLM 客户端，避免未配置时直接报错"""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "请在项目根目录创建 .env 文件并设置 DEEPSEEK_API_KEY=你的密钥，或设置环境变量 OPENAI_API_KEY"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ---------- 聚类参数（HDBSCAN 自动发现簇数，无需像 K-Means 那样指定 k）----------
CLUSTERING_MIN_CLUSTER_SIZE = 3   # 3≈10 簇较粗，2≈31 簇太碎，5+噪声多
CLUSTERING_MIN_SAMPLES = 2
CLUSTERING_METRIC = "precomputed"

# ---------- Embedding 参数（多语种）----------
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------- LLM 参数 ----------
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 500
LLM_MAX_SAMPLES_PER_CLUSTER = 10


def ensure_dirs():
    """确保所需目录存在"""
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
