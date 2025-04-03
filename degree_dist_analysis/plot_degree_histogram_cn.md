# `plot_degree_histogram.py` 中文文档

## 目的

该脚本为单个样本绘制选定晶粒度特征（例如 `neighbor_count`, `delaunay_degree_fixed_50`, `delaunay_degree_adaptive_1r1std`）的分布图。它有助于可视化根据特定邻域定义，具有不同邻居数量的晶粒的频率。

## 先决条件

*   包含样本中每个晶粒特征的 `_details.csv` 文件，该文件需包含用于绘图的度特征列。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

## 使用方法

```bash
python plot_degree_histogram.py --details_csv <详细数据文件路径> \
                                --output_path <输出图像保存路径.png> \
                                [--degree_column <度特征列名>] \
                                [--bin_width <宽度>] \
                                [--plot_type <类型>]
```

## 参数说明

*   `--details_csv` (必需): 输入的包含每个晶粒特征的 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出图表图像（例如 `.png`）的保存路径。
*   `--degree_column` (可选): `_details.csv` 文件中包含要绘制的度数据的列名。默认值：`'neighbor_count'`。您可以指定其他计算出的度，如 `'delaunay_degree_fixed_50'` 或 `'delaunay_degree_adaptive_1r1std'`。
*   `--bin_width` (可选): 当 `plot_type` 为 `'hist'` 时使用的直方图箱宽度。默认值：`1`。
*   `--plot_type` (可选): 要生成的图表类型。可选值：
    *   `'hist'` (默认): 生成直方图，显示落入各个度数区间的晶粒数量。
    *   `'bar'`: 生成条形图，显示每个离散度值的精确计数。

## 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图表显示了样本中所选晶粒度特征的分布。

## 解读与材料学意义

*   **分布形状**: 观察分布的整体形状。是单峰、双峰还是多峰？是对称、左偏还是右偏？这反映了晶粒典型的连通性环境。
*   **峰值位置**: 确定最频繁的度数值。这表明了最常见的晶粒局部排列方式。
*   **分布宽度/离散度**: 评估度数的范围或标准差。窄分布表明更均匀的连通性模式，而宽分布则表示局部环境的异质性更大。
*   **比较 (手动)**: 为同一样本的不同度定义（例如，物理邻接 vs. 自适应 Delaunay 度）生成直方图，观察邻域定义如何影响感知的连通性分布。为不同样本的相同度定义生成直方图，以直观比较它们的分布（尽管 `compare_degree_distributions.py` 更适合正式比较）。

该图表提供了基于局部晶粒连通性的样本拓扑结构的基本表征。
