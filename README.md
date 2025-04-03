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
*   **模型分析与解释**: (新增)
    *   使用基于模型的特征筛选方法 (如 RFECV) 进一步精简特征集。
    *   训练和评估多种回归模型 (如 RandomForest, GradientBoosting, XGBoost, SVR, GPR, Lasso/Ridge) 预测目标变量。
    *   进行超参数优化和交叉验证，选择最佳模型。
    *   提取最佳模型的特征重要性。
    *   使用 SHAP 分析解释特征对模型预测的影响，捕捉非线性关系。

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
├── run_model_analysis.py       # 脚本：执行基于模型的特征分析与解释 (新增)
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
    *   **注意**: `requirements.txt` 包含了运行所有脚本所需的主要依赖，包括新增的 `xgboost`, `shap`, `joblib`。请参考文件内的注释，特别是关于 `torch` (CPU 版本) 的安装说明。
    *   **SAM2 模型**: (仅当运行 `run_segmentation.py` 时需要) 需要下载 SAM2 检查点文件 (`sam2_checkpoint`) 并配置模型配置文件 (`model_cfg`) 的路径。

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
*   `--input_features_csv`: 最终筛选特征 CSV 文件路径 (必需, 来自上一步)。
*   `--external_data_csv`: 包含外部数据 (性能、工艺等) 的 CSV 文件路径 (必需)。
*   `--output_txt`: 保存相关性分析结果的文本文件路径 (必需, 例如 `results/correlations.txt`)。

**5. 模型分析与解释 (`run_model_analysis.py`)** (新增)

此脚本在特征筛选和相关性分析之后运行，用于更深入地理解特征与目标变量之间的关系。

```bash
python run_model_analysis.py --feature_csv <筛选后特征路径> --external_csv <外部数据路径> [--target_col <目标列名>] [--output_dir <结果保存目录>] [--cv_folds <折数>] [--random_state <随机种子>]
```
*   `--feature_csv`: 经过 `run_analysis.py` 筛选后的特征 CSV 文件路径 (必需, 例如 `results/final_features.csv`)。
*   `--external_csv`: 包含外部数据 (包括目标变量) 的 CSV 文件路径 (必需)。
*   `--target_col`: (可选) 要预测的目标变量列名，默认为 'Hc20'。
*   `--output_dir`: (可选) 保存模型分析结果（模型比较、选定特征列表、最佳模型、特征重要性、SHAP 图等）的目录，默认为 `results/model_analysis`。
*   `--cv_folds`: (可选) 交叉验证的折数，默认为 5。
*   `--random_state`: (可选) 用于保证结果可复现的随机种子，默认为 42。

脚本执行流程：
1.  加载并合并数据。
2.  使用 RFECV (基于 RandomForest) 进行特征筛选。
3.  在筛选后的特征上训练、调优和评估多种回归模型。
4.  保存模型比较结果和最佳模型。
5.  计算并保存最佳模型的特征重要性。
6.  对最佳模型进行 SHAP 分析并保存图表。

## 注意

*   确保已正确安装所有依赖项，特别是 `torch` (可能需要根据 CUDA 版本选择合适的安装方式) 和 `sam2` 相关的库。
*   确保 SAM2 模型文件路径正确。
*   输入图像应为 `.tif` 格式。

## 结果解释

本节旨在帮助理解项目各个脚本生成的关键输出文件和图表。

### `run_analysis.py` 输出

*   **`results/filterd_features.csv` (或指定的 `--final_output_csv`)**:
    *   **内容**: 经过 ICC/ANOVA、方差和相关性筛选后最终保留的特征集，数值为每个样本的平均特征值。
    *   **用途**: 作为后续 `run_correlation.py` 和 `run_model_analysis.py` 的输入，代表了可靠、有区分度且相对不冗余的微观结构量化指标。

*   **`results/icc_anova.csv` (或指定的 `--filtered_icc_anova_output`, 可选)**:
    *   **内容**: 每个原始特征的组内相关系数 (ICC) 和 ANOVA 分析的 P 值。
    *   **解释**:
        *   **ICC**: 衡量特征测量的可靠性或一致性。通常认为 ICC 值大于 0.75 的特征较为可靠。
        *   **P_value (ANOVA)**: 检验特征在不同样本组之间是否存在显著差异。经过多重比较校正后，P 值小于显著性水平（例如 0.05）通常表示该特征具有区分不同样本的能力。
        *   **Good_Feature**: 结合 ICC 和 ANOVA 结果判断该特征是否通过筛选。

### `run_correlation.py` 输出

*   **`results/correlations.txt` (或指定的 `--output_txt`)**:
    *   **内容**: 微观结构特征与外部数据（如性能、工艺参数）之间的相关系数矩阵或列表。
    *   **解释**: 相关系数范围从 -1 到 1。
        *   接近 1 表示强正相关。
        *   接近 -1 表示强负相关。
        *   接近 0 表示线性相关性很弱。
    *   **注意**: 相关性仅表示线性关系，不代表因果关系。

### `run_model_analysis.py` 输出 (默认在 `results/model_analysis/` 目录下)

*   **`rfecv_performance.png`**:
    *   **内容**: 递归特征消除 (RFE) 过程中，随着特征数量变化，模型在交叉验证中的性能得分（如 R²）曲线。
    *   **解释**: 用于辅助判断 RFE 选择的最佳特征数量。通常选择性能得分最高点对应的特征数。

*   **`rfecv_selected_features.txt`**:
    *   **内容**: RFE 过程最终选定的特征名称列表。
    *   **用途**: 这个特征子集将用于后续所有模型的训练和评估。

*   **`model_comparison_results.csv`**:
    *   **内容**: 包含多个模型（如 RandomForest, XGBoost, SVR 等）经过超参数优化后的交叉验证性能指标（R², MAE, RMSE）和最佳参数。
    *   **解释**: 用于比较不同模型的预测能力，选择综合表现最优的模型进行深入分析和解释。

*   **`best_model_<model_name>.joblib`**:
    *   **内容**: 使用 `joblib` 保存的最佳模型对象。
    *   **用途**: 可以加载此文件以重新使用训练好的模型进行预测或进一步分析。

*   **`<model_name>_feature_importance.csv` / `.png`**:
    *   **内容**: 基于最佳模型（通常是树模型）内部指标计算的特征重要性排序列表和条形图。
    *   **解释**: 显示了模型认为哪些特征对其预测结果的贡献最大。分数越高，表示特征越重要。**注意**: 对于非树模型，此文件可能不会生成。

*   **`<model_name>_shap_summary.png`**:
    *   **内容**: SHAP 摘要图（通常是蜂群图）。
    *   **解释**:
        *   每一行代表一个特征。
        *   每个点代表一个样本的该特征的 SHAP 值。
        *   点在 X 轴的位置表示该特征对该样本预测值的影响大小和方向（正/负）。
        *   点的颜色通常表示该特征本身的数值大小（高值/低值）。
        *   特征按其平均绝对 SHAP 值（总体影响）从上到下排序。

*   **`<model_name>_shap_feature_importance.png`**:
    *   **内容**: 基于平均绝对 SHAP 值计算的特征重要性条形图。
    *   **解释**: 提供了另一种基于 SHAP 的特征重要性排名视角，通常与 SHAP 摘要图中的排序一致。

*   **`<model_name>_shap_dependence_<feature>.png` (可选)**:
    *   **内容**: SHAP 依赖图，展示单个特征的数值如何影响其 SHAP 值。
    *   **解释**:
        *   X 轴是特征的实际值。
        *   Y 轴是该特征对应的 SHAP 值。
        *   每个点代表一个样本。
        *   点的颜色通常表示与之交互最强的另一个特征的值。
        *   此图有助于理解特征与预测目标之间的非线性关系以及潜在的特征交互效应。
