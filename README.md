# AI 指标分析流水线（模型监控）

客服对话数据：**多语种 Embedding → 聚类 → LLM 总结进线主题**。用于模型监控场景，对比 **2.11（固定 scenario 基准日）** 与 **2.12（对比日）** 两期改签数据，发现不同对话主题导致的进线差异。

## 项目结构

```
ai指标/
├── main.py              # 入口：运行全流程或单步
├── config.py            # 路径与参数配置
├── run.sh               # 一键运行脚本
├── requirements.txt
├── .env.example         # 复制为 .env，填写 DEEPSEEK_API_KEY
├── src/
│   ├── embedding.py     # 步骤 1：多语种 embedding（基于 content）
│   ├── clustering.py   # 步骤 2：HDBSCAN 聚类 + 可视化
│   └── llm_analysis.py # 步骤 3：按簇 LLM 总结进线主题
└── data/
    ├── raw/             # 原始输入数据
    │   ├── 2.11改签.csv # 基准日改签数据
    │   └── 2.12改签.csv # 对比日改签数据
    ├── processed/       # 中间结果
    │   └── embedded_data.csv
    └── output/          # 最终结果
        ├── clustering_result.csv
        ├── clustering_analysis.png
        ├── cluster_report_2.11.txt   # 前一天（基准日）的簇
        ├── cluster_report_2.12.txt   # 后一天（对比日）的簇
        └── cluster_report_diff.txt   # 两期差异总结
```

## 快速开始（已配好环境）

```bash
# 1. 编辑 .env，填入 DEEPSEEK_API_KEY
# 2. 一键运行全流程
./run.sh

# 或跳过 LLM（仅 embedding + 聚类）
./run.sh --skip-llm
```

## 手动配置

1. **安装依赖**  
   `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

2. **准备数据**  
   将 `2.11改签.csv`、`2.12改签.csv` 放在 `data/raw/` 目录。需包含列：`date`, `id`, `callout_id`, `product`, `language`, `scenario`, `content`。

3. **配置 LLM（仅步骤 3 需要）**  
   编辑 `.env`，将 `DEEPSEEK_API_KEY=your_deepseek_api_key_here` 替换为你的真实密钥。

4. **运行**
   - 全流程：`python main.py`
   - 只做 embedding + 聚类：`python main.py --skip-llm`
   - 单步：`python main.py --step 1`（embedding）/ `--step 2`（聚类）/ `--step 3`（LLM）

## 步骤说明

| 步骤 | 输入 | 输出 |
|------|------|------|
| 1 Embedding | `2.11改签.csv`、`2.12改签.csv` | `data/processed/embedded_data.csv` |
| 2 聚类 | `data/processed/embedded_data.csv` | `data/output/clustering_result.csv`、`clustering_analysis.png` |
| 3 LLM | `clustering_result.csv` + `embedded_data.csv` | `cluster_report_2.11.txt`、`cluster_report_2.12.txt`、`cluster_report_diff.txt` |

## 输出说明（模型监控三份文档）

- **cluster_report_2.11.txt**：2.11（基准日）各簇的对话数量与进线原因
- **cluster_report_2.12.txt**：2.12（对比日）各簇的对话数量与进线原因
- **cluster_report_diff.txt**：两期差异总结（数量变化、主题异同、关注点）

## Embedding 模型

默认使用 `paraphrase-multilingual-MiniLM-L12-v2`，支持中文、英文、日文、韩文等多语种。

## 聚类算法说明（HDBSCAN vs K-Means）

本项目使用 **HDBSCAN**（密度聚类），与 K-Means 有本质区别：

| 特性 | HDBSCAN | K-Means |
|------|---------|---------|
| **簇数量** | **自动发现**，无需事先指定 | 需手动指定 k |
| **噪声处理** | 自动识别离群点，标为 -1 | 每个点都会强制归属某个簇 |
| **簇形状** | 可识别任意形状 | 假设簇为凸形（球形） |

配置中的 `min_cluster_size`、`min_samples` 只影响「什么算一个有效簇」，不会预先定义簇的个数。簇数完全由数据密度结构自动决定。
