# 晶粒度分布分析与可视化 (中文说明)

本文件夹包含一系列脚本，旨在对微观结构的拓扑特征进行深入分析和可视化，特别是关注晶粒度分布（即晶粒的邻居数量 `neighbor_count`）及其与其他晶粒属性的关系。这些分析有助于理解机器学习模型识别出的重要特征背后的材料学意义。

## 先决条件

这些脚本依赖于主特征提取流程 (`run_feature_extraction.py`) 生成的包含每个晶粒详细信息的文件 (`*_details.csv`)。请确保在运行特征提取时启用了 `--save_grain_details` 标志：

```bash
python ../run_feature_extraction.py --mask_dir <你的掩码目录> --output_csv <聚合特征输出文件.csv> --save_grain_details --details_output_dir <你的详细数据目录>
```

## 脚本说明

每个脚本的详细说明可以在其对应的 Markdown 文件中找到：

*   **[`visualize_network.py`](visualize_network.md)** (英文文档): 可视化叠加在微观结构图像上的晶粒连接网络。网络中的节点（晶粒）可以根据选定的特征（如度、面积）进行着色和调整大小。
*   **[`plot_degree_vs_property.py`](plot_degree_vs_property.md)** (英文文档): 创建散点图（或其他类型，如 hexbin、kde）来探索晶粒度（例如 `neighbor_count`, `delaunay_degree_*`）与其他几何或形状属性（如面积、长宽比）之间的关系。
*   **[`visualize_spatial_coloring.py`](visualize_spatial_coloring.md)** (英文文档): 生成一张图像，其中每个晶粒根据特定特征（如度、面积、方向）的值进行着色，以揭示空间模式或聚类。
*   **[`plot_degree_histogram.py`](plot_degree_histogram.md)** (新增, 英文文档): 为单个样本绘制选定度特征（邻居数量）的分布直方图或条形图。
*   **[`compare_degree_distributions.py`](compare_degree_distributions.md)** (新增, 英文文档): 比较多个样本或样本组之间，针对选定度特征的度分布（使用 KDE、箱线图等）。
*   **[`visualize_degree_outliers.py`](visualize_degree_outliers.md)** (新增, 英文文档): 在分割图上高亮显示具有异常高或低度数的晶粒。

## 分析流程与结果解读

1.  **运行特征提取（包含详细信息）**: 为您想要分析的样本生成 `*_details.csv` 文件。
2.  **识别重要特征**: 利用 `run_model_analysis.py` 的结果（特征重要性、SHAP 图）来确定哪些 `grain_degree_dist_*` 或其他拓扑/几何特征对预测目标属性影响最大。
3.  **可视化网络 (`visualize_network.py`)**:
    *   选择具有代表性的样本（例如，目标属性值高/低的样本）。
    *   运行脚本，使用重要特征（如 `neighbor_count`, `area`）对着色/调整节点大小。
    *   **解读**: 观察网络拓扑结构的差异。性能好的样本是否显示出更均匀的网络？高度数节点是否聚集在一起？节点大小是否与度数在视觉上相关？
4.  **分析度数与属性的关系 (`plot_degree_vs_property.py`)**:
    *   为代表性样本运行脚本，绘制 `neighbor_count` 与其他可能相关的特征（如 `area`, `aspect_ratio`）的关系图。
    *   **解读**: 寻找趋势。小晶粒是否倾向于具有更高的度数（基于特定阈值）？形状和连通性之间是否存在关系？这种关系在性能高/低的样本之间有何不同？
5.  **可视化空间模式 (`visualize_spatial_coloring.py`)**:
    *   运行脚本，使用重要的度特征（例如 `delaunay_degree_adaptive_1r1std`）或其他相关特征对晶粒进行着色。
    *   **解读**: 寻找高/低度数晶粒的空间聚类。具有特定连通性模式的晶粒是否位于某些边界或缺陷附近（如果在原始图像中可见）？
6.  **分析度分布 (`plot_degree_histogram.py`)**:
    *   为代表性样本生成重要度特征（例如物理 `neighbor_count`、自适应 Delaunay 度）的直方图。
    *   **解读**: 描述分布的形状（例如单峰、双峰、偏态）。最常见的度数是多少？分布有多宽？
7.  **比较分布 (`compare_degree_distributions.py`)**:
    *   针对最相关的度特征，比较高性能和低性能样本组之间的度分布（使用 KDE 或箱线图）。
    *   **解读**: 组间在分布形状、均值或方差上是否存在显著差异？这为连通性模式的作用提供了有力证据。
8.  **识别异常值 (`visualize_degree_outliers.py`)**:
    *   基于选定的度特征，高亮显示具有异常低或高连通性的晶粒。
    *   **解读**: 检查这些异常晶粒的位置和特征。它们是否与特定的相、缺陷或边界类型相关？它们是否形成簇？
9.  **综合分析**: 将所有图表的可视化见解与 SHAP 分析（尤其是依赖图）的定量结果相结合，以更深入地理解特定的拓扑排列（通过度分布及相关特征捕捉）如何影响材料的性能。这种理解可以指导工程师设计具有所需特性的微观结构，以实现性能改进（材料逆向设计）。
