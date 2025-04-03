# Microstructure Analyzer

本项目旨在从微观结构图像（如 SEM 图像）中自动分割晶粒和晶界，提取全面的定量特征，并进行后续分析，以探索微观结构与材料性能/工艺参数之间的关系。项目代码基于 `BatchProcessing0312.ipynb` 重构而来。

## 功能

*   **图像分割**: 使用 SAM2 (Segment Anything Model 2) 对输入的微观结构图像进行自动分割，生成晶粒掩码。
*   **掩码处理**: 去除重叠掩码，过滤小面积区域，并使用 `msgpack` 高效存储/加载掩码数据。
*   **特征提取**:
    *   区分并标记晶粒、薄壁晶界、三叉晶界。
    *   计算几何、拓扑、形态、空间分布、网络结构等多维度特征。
    *   计算单个区域特征以及整个样本的全局特征和区域特征统计量。
*   **特征分析与筛选**:
    *   评估特征的内部一致性 (ICC) 和样本间区分性 (ANOVA)。
    *   基于方差和相关性进行特征降维。
*   **相关性分析**: 将提取的微观结构特征与外部数据（如材料性能、工艺参数）合并，计算相关性。

## 项目结构

```
.
├── microstructure_analyzer/    # 核心功能模块包
│   ├── __init__.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── feature_utils.py
│   ├── topology.py
│   ├── analysis.py
│   └── correlation.py
├── run_segmentation.py         # 脚本：执行图像分割
├── run_feature_extraction.py   # 脚本：执行特征提取
├── run_analysis.py             # 脚本：执行特征分析与筛选
├── run_correlation.py          # 脚本：执行相关性分析
└── README.md                   # 本文件
```

## 安装

1.  **克隆仓库** (如果适用)
    ```bash
    git clone <repository_url>
    cd microstructure_analyzer
    ```
2.  **创建虚拟环境** (推荐)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **安装依赖**
    *   (需要先确定具体的依赖包和版本，可以基于原始 Notebook 的导入语句生成 `requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```
    *   **SAM2 模型**: 需要下载 SAM2 检查点文件 (`sam2_checkpoint`) 并配置模型配置文件 (`model_cfg`) 的路径。

## 使用

项目包含四个独立的脚本，用于按顺序执行分析流程的各个阶段。请在项目根目录下运行这些脚本。

**1. 图像分割 (`run_segmentation.py`)**

```bash
python run_segmentation.py --input_dir <图像目录> --mask_dir <掩码输出目录> --sam_checkpoint <模型检查点路径> --sam_config <模型配置路径> [--device cuda/cpu]
```
*   `--input_dir`: 包含 `.tif` 图像的输入目录 (必需)。
*   `--mask_dir`: 保存 `.mask` 分割结果的目录 (必需)。
*   `--sam_checkpoint`: SAM2 模型检查点文件路径 (必需)。
*   `--sam_config`: SAM2 模型配置文件路径 (必需)。
*   `--device`: 使用的设备 (`cuda` 或 `cpu`)，默认为 `cuda`。

**2. 特征提取 (`run_feature_extraction.py`)**

```bash
python run_feature_extraction.py --mask_dir <掩码输入目录> --output_csv <原始特征输出路径>
```
*   `--mask_dir`: 包含 `.mask` 文件的目录 (必需)。
*   `--output_csv`: 保存原始提取特征的 CSV 文件路径 (必需, 例如 `results/raw_features.csv`)。

**3. 特征分析与筛选 (`run_analysis.py`)**

```bash
python run_analysis.py --input_csv <原始特征输入路径> --final_output_csv <最终特征输出路径> [--filtered_icc_anova_output <ICC/ANOVA结果路径>] [--icc_threshold 0.75] [--corr_threshold 0.95] [...]
```
*   `--input_csv`: 原始特征 CSV 文件路径 (必需, 来自上一步)。
*   `--final_output_csv`: 保存最终筛选特征 (样本均值) 的 CSV 文件路径 (必需, 例如 `results/final_features.csv`)。
*   `--filtered_icc_anova_output`: (可选) 保存 ICC/ANOVA 筛选结果的路径 (例如 `results/icc_anova.csv`)。
*   `--icc_threshold`: (可选) ICC 筛选阈值，默认为 0.75。
*   `--corr_threshold`: (可选) 相关性筛选阈值，默认为 0.95。
*   `--max_raters`: (可选) 用于 ICC 计算的最大 Rater 数，默认为 8。
*   `--anova_alpha`: (可选) ANOVA 显著性水平，默认为 0.05。
*   `--variance_threshold`: (可选) 方差筛选阈值，默认为 0.0。

**4. 相关性分析 (`run_correlation.py`)**

```bash
python run_correlation.py --input_features_csv <最终特征输入路径> --external_data_csv <外部数据路径> --output_txt <关联结果输出路径>
```
*   `--input_features_csv`: 最终筛选特征 CSV 文件路径 (必需, 来自上一步)。
*   `--external_data_csv`: 包含外部数据 (性能、工艺等) 的 CSV 文件路径 (必需)。
*   `--output_txt`: 保存相关性分析结果的文本文件路径 (必需, 例如 `results/correlations.txt`)。

## 注意

*   确保已正确安装所有依赖项，特别是 `torch` (可能需要根据 CUDA 版本选择合适的安装方式) 和 `sam2` 相关的库。
*   确保 SAM2 模型文件路径正确。
*   输入图像应为 `.tif` 格式。
