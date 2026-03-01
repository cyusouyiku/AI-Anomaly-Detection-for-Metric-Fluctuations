# -*- coding: utf-8 -*-
"""步骤 1：加载 2.11/2.12 两期改签数据，按行（每条记录）多语种 embedding"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if _root not in sys.path:
        sys.path.insert(0, str(_root))
from config import (
    PROJECT_ROOT,
    RAW_DATA_2_11,
    RAW_DATA_2_12,
    EMBEDDED_DATA_FILE,
    ensure_dirs,
    EMBEDDING_MODEL,
)


def load_and_merge_data(path_2_11=None, path_2_12=None):
    """加载两期数据并合并，保留每行独立（不按 callout_id 聚合）"""
    path_2_11 = path_2_11 or RAW_DATA_2_11
    path_2_12 = path_2_12 or RAW_DATA_2_12
    if not path_2_11.exists():
        raise FileNotFoundError(f"2.11 数据不存在: {path_2_11}")
    if not path_2_12.exists():
        raise FileNotFoundError(f"2.12 数据不存在: {path_2_12}")

    df1 = pd.read_csv(path_2_11, encoding="utf-8-sig")
    df2 = pd.read_csv(path_2_12, encoding="utf-8-sig")
    df1["period"] = "2.11"
    df2["period"] = "2.12"
    df = pd.concat([df1, df2], ignore_index=True)

    if "content" not in df.columns:
        raise KeyError("数据需包含 content 列")
    if "id" not in df.columns:
        raise KeyError("数据需包含 id 列（行唯一标识）")

    # 按行保留，每条记录独立 embedding（便于聚类发现更多主题）
    return df[["id", "callout_id", "content", "period"]].copy()


def embed_texts(texts, model_name=None):
    """使用多语种模型对文本进行 embedding"""
    # 缓存到项目目录，避免权限问题
    cache_dir = PROJECT_ROOT / "data" / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]  # 避免镜像 404

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("请安装: pip install sentence-transformers")

    model_name = model_name or EMBEDDING_MODEL
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 100)
    return np.array(embeddings, dtype=np.float64)


def run(path_2_11=None, path_2_12=None, output_path=None):
    """
    执行：加载两期数据 → 聚合 content → 多语种 embedding → 保存。
    :return: (DataFrame, 输出路径)
    """
    ensure_dirs()
    out_path = output_path or EMBEDDED_DATA_FILE

    print("  加载 2.11、2.12 两期数据...")
    agg = load_and_merge_data(path_2_11, path_2_12)
    print(f"  合并后共 {len(agg)} 条记录")

    texts = agg["content"].fillna("").astype(str).tolist()
    print(f"  使用多语种模型 {EMBEDDING_MODEL} 进行 embedding...")
    embeddings = embed_texts(texts)

    agg["embedding_features"] = [f"[{','.join(map(str, emb))}]" for emb in embeddings]
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {out_path}")
    return agg, out_path


if __name__ == "__main__":
    run()
