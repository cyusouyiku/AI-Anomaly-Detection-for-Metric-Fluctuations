# -*- coding: utf-8 -*-
"""
AI 指标分析自动化流水线入口。

流程：Embedding（多语种）→ 聚类 → LLM 总结
- 步骤 1：加载 2.11/2.12 改签数据，按 content 多语种 embedding → data/processed/embedded_data.csv
- 步骤 2：HDBSCAN 聚类 → data/output/clustering_result.csv、clustering_analysis.png
- 步骤 3：按簇调用 LLM 总结进线主题（模型监控）→ 三份报告：cluster_report_2.11.txt、cluster_report_2.12.txt、cluster_report_diff.txt

使用方式：
  python main.py              # 运行全流程
  python main.py --skip-llm   # 只做 embedding + 聚类（不调 API）
  python main.py --step 1     # 只运行步骤 1（embedding）
  python main.py --step 2     # 只运行步骤 2（聚类）
  python main.py --step 3     # 只运行步骤 3（LLM，需先有步骤 1、2 结果）
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="AI 指标分析流水线：Embedding → 聚类 → LLM 总结（模型监控 2.11 vs 2.12）")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="只运行指定步骤（1=embedding 2=聚类 3=LLM）")
    parser.add_argument("--skip-llm", action="store_true", help="全流程但跳过步骤 3（不调用 LLM）")
    args = parser.parse_args()

    ensure_dirs()
    run_step1 = run_step2 = run_step3 = False
    if args.step:
        run_step1 = args.step == 1
        run_step2 = args.step == 2
        run_step3 = args.step == 3
    else:
        run_step1 = run_step2 = True
        run_step3 = not args.skip_llm

    print("=" * 60)
    print("AI 指标分析流水线（模型监控：2.11 vs 2.12 改签）")
    print("=" * 60)

    if run_step1:
        print("\n[步骤 1/3] Embedding（多语种）...")
        from src.embedding import run as run_embedding
        run_embedding()
        print("步骤 1 完成。\n")

    if run_step2:
        print("\n[步骤 2/3] 聚类...")
        from src.clustering import run as run_clustering
        run_clustering(show_plot=False)
        print("步骤 2 完成。\n")

    if run_step3:
        print("\n[步骤 3/3] LLM 分析（进线主题）...")
        from src.llm_analysis import run as run_llm
        run_llm()
        print("步骤 3 完成。\n")

    print("=" * 60)
    print("流水线执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
