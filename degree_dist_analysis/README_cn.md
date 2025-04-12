# 晶粒度分布分析与可视化 (中文说明)

本文件夹包含一系列脚本，旨在对微观结构的拓扑特征进行深入分析和可视化，特别是关注晶粒度分布（即晶粒的邻居数量 `neighbor_count`）及其与其他晶粒属性的关系。这些分析有助于理解机器学习模型识别出的重要特征背后的材料学意义。

## 先决条件

这些脚本通常依赖于主特征提取流程 (`run_feature_extraction.py`) 生成的包含每个晶粒详细信息的文件 (`*_details.csv`)。请确保在运行特征提取时启用了 `--save_grain_details` 标志：

```bash
python ../run_feature_extraction.py --mask_dir <你的掩码目录> --output_csv <聚合特征输出文件.csv> --save_grain_details --details_output_dir <你的详细数据目录>
```

## 脚本说明

### `batch_all_degree_analysis.py`

#### 目的

该脚本是对`batch_degree_analysis.py`的增强版本，用于批量执行所有晶粒度分析脚本，并且支持所有绘图方式。主要功能包括：

- 自动匹配每个样本的mask文件和details文件
- 并行执行所有分析脚本
- 对支持多种绘图方式的脚本，执行所有绘图方式
- 将绘图方式添加到输出文件名中
- 将结果按脚本名称分类保存
- 统一使用`delaunay_degree_adaptive_2r0_5std`作为默认度数特征
- 特别处理`compare_degree_distributions.py`，因为它分析的是文件夹而不是单个样品

#### 先决条件

- 多个`.mask`文件（样本分割结果）
- 对应的`_details.csv`文件（包含晶粒特征）
- Python 3.6+环境
- 所有分析脚本位于同一目录下

#### 用法

```bash
python batch_all_degree_analysis.py --mask_dir <mask文件目录> \
                                  --details_dir <details文件目录> \
                                  --output_dir <输出目录> \
                                  --max_workers <并行任务数> \
                                  --skip_compare
```

#### 参数说明

- `--mask_dir`：包含`.mask`文件的目录路径（必需）
- `--details_dir`：包含`_details.csv`文件的目录路径（必需）
- `--output_dir`：输出结果目录，默认为`results\batch_all_degree_analysis`
- `--max_workers`：最大并行任务数，默认为4
- `--skip_compare`：跳过比较分布分析（可选标志）

#### 输出

脚本会在指定的输出目录下创建以下子目录结构：

```
<output_dir>/
  ├── plot_degree_histogram/
  │   ├── sample1_hist_hist.png
  │   ├── sample1_hist_bar.png
  │   ├── sample2_hist_hist.png
  │   └── sample2_hist_bar.png
  ├── plot_degree_vs_property/
  │   ├── sample1_vs_property_scatter.png
  │   ├── sample1_vs_property_hexbin.png
  │   ├── sample1_vs_property_kde.png
  │   └── ...
  ├── visualize_network/
  │   ├── sample1_network.png
  │   └── sample2_network.png
  ├── visualize_spatial_coloring/
  │   ├── sample1_spatial.png
  │   └── sample2_spatial.png
  ├── visualize_degree_outliers/
  │   ├── sample1_outliers.png
  │   └── sample2_outliers.png
  └── compare_degree_distributions/
      ├── comparison_compare_kde.png
      ├── comparison_compare_box.png
      ├── comparison_compare_violin.png
      └── comparison_compare_hist.png
```

#### 注意事项

1. 脚本会依次处理每个样本，调用所有分析脚本（包括多种绘图方式），最后处理比较分布分析（除非使用--skip_compare跳过）
2. 日志文件`batch_all_degree_analysis.log`会记录所有执行过程
3. 确保mask文件和details文件命名一致（如`sample1.mask`对应`*sample1_details.csv`）
4. 并行处理数量应根据CPU核心数合理设置
5. 输出目录结构会按分析脚本自动创建
6. 所有绘图结果会添加分析类型和绘图方式后缀（如`_hist_hist`, `_hist_bar`, `_network`等）
7. 默认度数特征已更改为`delaunay_degree_adaptive_2r0_5std`，以更好地反映基于Delaunay图的晶粒连通性分析

#### 示例

```bash
python batch_all_degree_analysis.py --mask_dir input/masks \
                                  --details_dir input/details \
                                  --output_dir results \
                                  --max_workers 8
```

这将处理`input/masks`目录中的所有`.mask`文件，使用`input/details`目录中的对应`_details.csv`文件，并将结果保存在`results`目录下。

### `batch_compare_features.py`

#### 目的

该脚本通过为指定的特征列表重复调用 `compare_degree_distributions.py` 来实现批量处理。它为每个特征生成所有四种比较图类型（KDE、Box、Violin、Histogram），并将它们保存到指定的输出文件夹中。这对于根据 Hc20 性能快速比较所有样本中多个晶粒特征的分布非常有用。

#### 先决条件

*   `compare_degree_distributions.py` 脚本位于同一目录中。
*   指定的详细信息文件夹中包含多个 `_details.csv` 文件，这些文件包含要绘制的特征。
*   `data/database-250211.csv` 文件包含按样本 ID 索引的 'Hc20' 值。

#### 用法

```bash
python batch_compare_features.py --details_folder <详细数据文件夹路径> \
                                 --output_base_folder <输出目录路径> \
                                 [--features <特征1> <特征2> ...]
```

#### 参数说明

*   `--details_folder` (必需): 包含 `_details.csv` 文件的目录路径（例如 `results/per_grain_details`）。
*   `--output_base_folder` (必需): 保存输出比较图的基础路径（例如 `results/batch_compare_features`）。如果目录不存在，脚本将创建它。
*   `--features` (可选): 要绘制的 `_details.csv` 文件中的特征列名称列表。如果未提供，则使用包含常见几何和拓扑特征的默认列表：`['area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'circularity', 'delaunay_degree_adaptive_2r0_5std', 'neighbor_count']`。

#### 输出

*   多个图像文件（例如 `.png`）保存在指定的 `--output_base_folder` 中。
*   每个文件根据模式命名：`{feature_name}_{plot_type}.png`（例如 `area_kde.png`, `delaunay_degree_adaptive_2r0_5std_box.png`）。
*   每个图表比较指定特征在所有样本中的分布，并按 Hc20 值排序。

### `compare_degree_distributions.py`

#### 目的

该脚本用于比较多个样本或预定义样本组之间的晶粒度分布。它读取多个样本的每个晶粒的详细信息文件 (`_details.csv`)，提取指定的度特征（例如 `neighbor_count`, `delaunay_degree_adaptive_1r1std`），并生成图表（如 KDE 图、箱线图、小提琴图或分面直方图）以直观地比较这些分布。

这对于识别不同实验条件、材料批次或性能组之间晶粒连通性模式的系统性差异非常有用。

#### 先决条件

*   多个 `_details.csv` 文件，每个待比较的样本一个，位于指定的文件夹内。这些文件必须包含用于比较的度特征列。它们由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python compare_degree_distributions.py --details_folder <包含详细数据文件的文件夹路径> \
                                       --output_path <比较图保存路径.png> \
                                       [--file_pattern <glob模式>] \
                                       [--degree_column <度特征列名>] \
                                       [--plot_type <类型>] \
                                       [--labels <标签1> <标签2> ...]
```

#### 参数说明

*   `--details_folder` (必需): 包含待比较样本的 `_details.csv` 文件的目录路径。
*   `--output_path` (必需): 输出比较图图像（例如 `.png`）的保存路径。
*   `--file_pattern` (可选): 用于在 `--details_folder` 中查找相关 `_details.csv` 文件的 glob 模式。默认值：`'*_details.csv'`。
*   `--degree_column` (可选): `_details.csv` 文件中包含要比较的度数据的列名。默认值：`'neighbor_count'`。
*   `--plot_type` (可选): 用于比较的图表类型。可选值：
    *   `'kde'` (默认): 核密度估计图，叠加显示以供比较。通过平滑数据展示分布形状和峰值，适合比较不同组的整体分布形态和峰位差异。关键信息：峰值位置(众数)、峰的数量(模态)、分布宽度(离散度)、不同度数值下组间的相对密度。
    *   `'box'`: 每个样本/组的箱线图，显示中位数、四分位数和潜在异常值。箱体显示第25-75百分位数，中线为中位数，须线延伸至1.5倍IQR范围内的极值，之外的点为异常值。适合快速比较各组的中心趋势(中位数)和离散程度(四分位距IQR)。关键信息：中位数位置、IQR大小、异常值分布。
    *   `'violin'`: 小提琴图，结合了箱线图和KDE图的特点。宽度表示数据密度，内部通常包含箱线图元素。适合同时比较分布形状、密度和中心趋势。关键信息：结合了箱线图和KDE图的所有关键信息，特别适合展示多模态分布。
    *   `'hist'`: 分面直方图，将数据范围划分为区间(bins)并显示每个区间内的频数，每个组别在单独的面板中展示。适合详细比较特定度数范围内的频率差异。关键信息：各区间内的频数、整体分布形状(离散形式)、识别数据缺口或特定值簇。
*   `--labels` (可选): 为找到的每个样本/文件分配的自定义标签列表（按文件被找到的顺序，通常是字母顺序）。如果未提供，则从文件名的开头（第一个下划线之前）自动生成标签。注意：当未提供labels参数时，脚本会自动从data/database-250211.csv文件中读取Hc20值，并按Hc20值从低到高对样品进行排序显示。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图表直观地比较了分析中包含的不同样本之间所选度特征的分布。

#### 解读与材料学意义

*   **分布比较 (KDE/Hist)**: 比较样本/组之间分布的形状、峰值位置和宽度。某个组是否始终显示出更高的平均度数？某个组的分布是更宽还是更窄？是否存在多个峰值，表明存在不同的晶粒连通性群体？
*   **集中趋势与离散度 (Box/Violin)**: 比较各组之间的中位度数（箱线图的中线）、四分位距（箱体高度）和整体范围（须）。在典型连通性或连通性变异性方面是否存在显著差异？
*   **统计显著性**: 虽然此脚本提供可视化比较，但观察到的差异可能需要进行正式的统计检验（例如，对按样本分组的度数据进行 t 检验、ANOVA 或 Kruskal-Wallis 检验）以确认其显著性（如果需要）。
*   **与性能/条件的联系**: 脚本会自动从data/database-250211.csv文件中读取Hc20值，并按Hc20值从低到高对样品进行排序显示。这有助于直观地分析晶粒拓扑特征与材料性能(Hc20)之间的关系。将观察到的度分布差异与样本/组相关的已知材料性能（例如 Hc20）或加工条件的差异联系起来。这有助于建立涉及晶粒拓扑的结构-性能或加工-结构关系。

该脚本有助于对晶粒连通性模式进行比较分析，这是理解微观结构拓扑如何变化并可能影响材料行为的关键一步。

#### 图表类型详解

##### 核密度估计图 (KDE)
- **原理**: 通过平滑数据展示分布形状和峰值
- **解读要点**: 
  - 峰值位置(众数)反映最常出现的度数
  - 峰的数量(模态)指示是否存在多个不同的晶粒连通性群体
  - 分布宽度反映离散程度
  - 不同组间的曲线重叠程度显示相似性
- **适用场景**: 比较整体分布形态和峰位差异

##### 箱线图 (Box Plot)
- **原理**: 显示中位数、四分位数和潜在异常值
- **解读要点**:
  - 箱体显示第25-75百分位数(IQR)
  - 中线为中位数
  - 须线延伸至1.5倍IQR范围内的极值
  - 之外的点为异常值
- **适用场景**: 快速比较各组的中心趋势和离散程度

##### 小提琴图 (Violin Plot)
- **原理**: 结合箱线图和KDE图的特点
- **解读要点**:
  - 宽度表示数据密度
  - 内部通常包含箱线图元素
  - 特别适合展示多模态分布
- **适用场景**: 同时比较分布形状、密度和中心趋势

##### 分面直方图 (Faceted Histogram)
- **原理**: 将数据划分为区间(bins)并显示频数
- **解读要点**:
  - 各区间内的频数
  - 整体分布形状(离散形式)
  - 识别数据缺口或特定值簇
- **适用场景**: 详细比较特定度数范围内的频率差异

### `plot_degree_histogram.py`

#### 目的

该脚本为单个样本绘制选定晶粒度特征（例如 `neighbor_count`, `delaunay_degree_fixed_50`, `delaunay_degree_adaptive_1r1std`）的分布图。它有助于可视化根据特定邻域定义，具有不同邻居数量的晶粒的频率。

#### 先决条件

*   包含样本中每个晶粒特征的 `_details.csv` 文件，该文件需包含用于绘图的度特征列。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python plot_degree_histogram.py --details_csv <详细数据文件路径> \
                                --output_path <输出图像保存路径.png> \
                                [--degree_column <度特征列名>] \
                                [--bin_width <宽度>] \
                                [--plot_type <类型>]
```

#### 参数说明

*   `--details_csv` (必需): 输入的包含每个晶粒特征的 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出图表图像（例如 `.png`）的保存路径。
*   `--degree_column` (可选): `_details.csv` 文件中包含要绘制的度数据的列名。默认值：`'neighbor_count'`。您可以指定其他计算出的度，如 `'delaunay_degree_fixed_50'` 或 `'delaunay_degree_adaptive_1r1std'`。
*   `--bin_width` (可选): 当 `plot_type` 为 `'hist'` 时使用的直方图箱宽度。默认值：`1`。
*   `--plot_type` (可选): 要生成的图表类型。可选值：
    *   `'hist'` (默认): 生成直方图，显示落入各个度数区间的晶粒数量。
    *   `'bar'`: 生成条形图，显示每个离散度值的精确计数。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图表显示了样本中所选晶粒度特征的分布。

#### 解读与材料学意义

*   **分布形状**: 观察分布的整体形状。是单峰、双峰还是多峰？是对称、左偏还是右偏？这反映了晶粒典型的连通性环境。
*   **峰值位置**: 确定最频繁的度数值。这表明了最常见的晶粒局部排列方式。
*   **分布宽度/离散度**: 评估度数的范围或标准差。窄分布表明更均匀的连通性模式，而宽分布则表示局部环境的异质性更大。
*   **比较 (手动)**: 为同一样本的不同度定义（例如，物理邻接 vs. 自适应 Delaunay 度）生成直方图，观察邻域定义如何影响感知的连通性分布。为不同样本的相同度定义生成直方图，以直观比较它们的分布（尽管 `compare_degree_distributions.py` 更适合正式比较）。

该图表提供了基于局部晶粒连通性的样本拓扑结构的基本表征。

### `plot_degree_vs_property.py`

#### 目的

该脚本用于生成图表，探索晶粒的拓扑度（即其邻居数量，通常是 `neighbor_count`）与该晶粒的其他几何或形状属性（例如 `area`, `aspect_ratio`, `compactness`）之间的关系。

理解这些关系有助于揭示特定类型的晶粒（例如小晶粒、细长晶粒）在微观结构中是否倾向于具有更高或更低的连通性。

#### 先决条件

*   包含样本中每个晶粒特征的 `_details.csv` 文件，该文件需包含 `neighbor_count`（或指定的度特征列）和所选的属性特征列。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python plot_degree_vs_property.py --details_csv <详细数据文件路径> \
                                  --output_path <输出图像保存路径.png> \
                                  [--property <属性特征名称>] \
                                  [--degree <度特征名称>] \
                                  [--plot_type <图表类型>]
```

#### 参数说明

*   `--details_csv` (必需): 输入的包含每个晶粒特征的 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出图表图像（例如 `.png`）的保存路径。
*   `--property` (可选): 要绘制在 x 轴上的特征列名称。默认值：`'area'` (面积)。
*   `--degree` (可选): 代表晶粒度数、要绘制在 y 轴上的特征列名称。默认值：`'delaunay_degree_adaptive_2r0_5std'` (邻居数量)。
*   `--plot_type` (可选): 要生成的图表类型。可选值：
    *   `'scatter'` (默认): 显示单个数据点。适合观察原始分布，但在数据点很多时可能出现过度绘制问题。
    *   `'hexbin'`: 将点聚合到六边形箱中，并根据箱内点的密度进行着色。适合大型数据集以可视化密度模式。
    *   `'kde'`: 核密度估计图。显示数据点的估计概率密度。适合可视化分布的整体形状和峰值。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图表显示了所选晶粒属性（x 轴）和晶粒度数（y 轴）之间的关系。

#### 解读与材料学意义

*   **趋势分析**: 观察图中的总体趋势。度数是否随着属性值的变化而增加或减少？例如，绘制度数与面积的关系图可能显示小晶粒通常比较大晶粒拥有更多还是更少的邻居。
*   **分布形状 (KDE/Hexbin)**: 观察数据点集中的区域。是否存在特别常见的属性值和度数组合？
*   **异常值**: 识别任何具有异常属性和度数组合的晶粒。
*   **比较**: 为具有不同整体性能（例如，高/低 Hc20）或在不同条件下处理的样本生成这些图。比较其趋势。晶粒尺寸/形状与连通性之间的关系在不同样本之间是否存在显著差异？
*   **与理论联系**: 将观察到的趋势与理论预期或已知现象联系起来。例如，在密集堆积的结构中，预计较小的晶粒平均邻居数量会更高（接近堆积密度的相关值）。与此的偏差可能表明存在特定的结构特征或非理想堆积。

通过检查这些图表，您可以定量地理解局部连通性（度）如何与微观结构中其他基本晶粒特征相关联。

### `visualize_degree_outliers.py`

#### 目的

该脚本生成一张分割后的微观结构图像，并特别高亮显示那些度数（基于选定特征的邻居数量）落在指定范围之外的晶粒。这有助于快速识别具有异常高或低连通性的晶粒。

#### 先决条件

*   包含样本分割结果的 `.mask` 文件。
*   包含每个晶粒特征（包括待分析的度特征列）的相应 `_details.csv` 文件。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python visualize_degree_outliers.py --mask_path <掩码文件路径> \
                                    --details_csv <详细数据文件路径> \
                                    --output_path <输出图像保存路径.png> \
                                    [--degree_column <度特征列名>] \
                                    [--min_degree <下界>] \
                                    [--max_degree <上界>]
```
**注意：** 必须提供 `--min_degree` 或 `--max_degree` 中的至少一个。

#### 参数说明

*   `--mask_path` (必需): 输入样本的 `.mask` 文件路径。
*   `--details_csv` (必需): 包含每个晶粒特征的相应 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出可视化图像（例如 `.png`）的保存路径。
*   `--degree_column` (可选): `_details.csv` 文件中包含度数据的列名。默认值：`'neighbor_count'`。
*   `--min_degree` (可选): 整数值。度数**小于**此值的晶粒将被高亮显示为异常值。
*   `--max_degree` (可选): 整数值。度数**大于**此值的晶粒将被高亮显示为异常值。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图像显示了分割后的晶粒：
    *   度数落在指定范围之外（`< min_degree` 或 `> max_degree`）的晶粒以独特的颜色（默认：红色）显示。
    *   度数在指定范围内的晶粒（或者如果只提供了一个边界，则那些不满足异常条件的晶粒）以中性色（默认：灰色）显示。
    *   背景通常为白色。
    *   图例指示了异常和正常度数晶粒的颜色。

#### 解读与材料学意义

*   **识别极端情况**: 快速定位具有极低连通性（可能孤立或靠近边缘/孔隙）或极高连通性（可能位于密集堆积的簇中，或者如果分割不完美则可能代表伪影）的晶粒。
*   **空间位置**: 观察这些异常晶粒在微观结构中的位置。低度数晶粒是否主要靠近样品边界？高度数晶粒是否聚集在特定区域？
*   **与其他特征的相关性**: 通过将此可视化与 `visualize_spatial_coloring.py` 生成的图（按面积、形状等着色）进行比较，可以研究异常连通性是否与其他晶粒特征相关。
*   **与加工/性能的联系**: 如果在具有特定性能或加工历史的样本中持续观察到某些类型的异常值（例如，大量低度数晶粒），则表明这些极端连通性状态与整体材料行为或形成机制之间存在联系。

这种可视化提供了一个聚焦于表现出异常局部连通性的晶粒的视图，这些晶粒可能是理解结构异质性或失效起始的关键点。

### `visualize_network.py`

#### 目的

该脚本用于可视化单个样本的晶粒连通性网络。它读取分割掩码数据和相应的每个晶粒的特征详细信息，构建一个网络图（通常基于质心的 Delaunay 三角剖分，并按距离过滤），然后将此网络叠加在原始 SEM 图像（如果提供）或分割后的晶粒边界上。网络中的节点代表单个晶粒，可以根据为每个晶粒提取的特定特征（如邻居数量、面积等）进行着色和调整大小。

这种可视化有助于理解微观结构中晶粒的空间排列和连通性模式。

#### 先决条件

*   包含样本分割结果的 `.mask` 文件。
*   包含每个晶粒特征（包括质心以及用于着色/调整大小的特征）的相应 `_details.csv` 文件。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python visualize_network.py --mask_path <掩码文件路径> \
                            --details_csv <详细数据文件路径> \
                            --output_path <输出图像保存路径.png> \
                            [--image_path <原始tif图像路径>] \
                            [--distance_threshold <距离阈值>] \
                            [--color_feature <颜色特征名称>] \
                            [--size_feature <尺寸特征名称>] \
                            [--min_size <最小尺寸>] \
                            [--max_size <最大尺寸>] \
                            [--cmap <颜色映射名称>]
```

#### 参数说明

*   `--mask_path` (必需): 输入样本的 `.mask` 文件路径。
*   `--details_csv` (必需): 包含每个晶粒特征的相应 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出可视化图像（例如 `.png`）的保存路径。
*   `--image_path` (可选): 原始图像文件（例如 `.tif`）的路径。如果提供，网络将叠加在此图像上。否则，将叠加在分割后的晶粒边界上。
*   `--distance_threshold` (可选): 在 Delaunay 图中，两个晶粒质心之间被视为连接的最大距离（像素）。默认值：`50`。调整此值可以揭示不同空间尺度下的连通性。
*   `--color_feature` (可选): `_details.csv` 文件中用于给节点（晶粒）着色的列名。默认值：`'delaunay_degree_adaptive_2r0_5std'` (邻居数量)。
*   `--size_feature` (可选): `_details.csv` 文件中用于调整节点（晶粒）大小的列名。默认值：`'area'` (面积)。节点大小将在 `--min_size` 和 `--max_size` 之间线性缩放。
*   `--min_size` (可选): 图中节点的最小尺寸。默认值：`10`。
*   `--max_size` (可选): 图中节点的最大尺寸。默认值：`500`。
*   `--cmap` (可选): 用于节点着色的 Matplotlib 颜色映射 (colormap) 名称。默认值：`'viridis'`。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图像显示了晶粒网络：
    *   **背景**: 原始 SEM 图像或分割后的晶粒边界。
    *   **节点**: 代表晶粒质心。其颜色和大小分别由 `--color_feature` 和 `--size_feature` 参数确定。
    *   **边**: 代表相邻晶粒之间的连接（基于 Delaunay 图和距离阈值）。

#### 解读与材料学意义

*   **连通性**: 观察边的整体密度和模式。晶粒是高度互连的，还是存在孤立区域？
*   **节点着色 (例如，按 `neighbor_count`)**: 识别具有高或低局部连通性的区域。高度数晶粒是否聚集在一起？连通性与晶粒位置（例如，靠近样品边缘）有何关系？
*   **节点大小 (例如，按 `area`)**: 直观评估晶粒尺寸和连通性之间的关系。大晶粒是否倾向于拥有较少的邻居？
*   **比较**: 为具有不同性能（例如，高/低 Hc20）的样本生成这些图。比较它们的网络结构。性能更好的样本是否表现出更均匀的网络、特定的聚类模式，或者尺寸、形状和连通性之间存在不同的关系？
*   **与缺陷的关联**: 如果叠加在原始图像上，将网络特征（如低度数节点或断开的连接）与可见的缺陷或相界关联起来。

通过分析这些网络可视化图，您可以深入了解晶粒的拓扑排列（超越简单的平均尺寸或形状）如何可能影响材料性能。

### `visualize_spatial_coloring.py`

#### 目的

该脚本生成一张分割后的微观结构图像，其中每个单独的晶粒根据与其关联的特定特征的值（例如其邻居数量、面积、方向、长宽比）进行着色。

这种类型的可视化对于识别微观结构中与所选特征相关的空间模式、聚类或梯度非常有用。

#### 先决条件

*   包含样本分割结果的 `.mask` 文件。
*   包含每个晶粒特征（包括用于着色的特征）的相应 `_details.csv` 文件。此文件由 `run_feature_extraction.py` 使用 `--save_grain_details` 标志生成。

#### 使用方法

```bash
python visualize_spatial_coloring.py --mask_path <掩码文件路径> \
                                     --details_csv <详细数据文件路径> \
                                     --output_path <输出图像保存路径.png> \
                                     [--color_feature <颜色特征名称>] \
                                     [--cmap <颜色映射名称>]
```

#### 参数说明

*   `--mask_path` (必需): 输入样本的 `.mask` 文件路径。
*   `--details_csv` (必需): 包含每个晶粒特征的相应 `_details.csv` 文件路径。
*   `--output_path` (必需): 输出可视化图像（例如 `.png`）的保存路径。
*   `--color_feature` (可选): `_details.csv` 文件中用于给晶粒着色的列名。默认值：`'neighbor_count'` (邻居数量)。
*   `--cmap` (可选): 用于着色的 Matplotlib 颜色映射 (colormap) 名称。默认值：`'viridis'`。

#### 输出

*   一个图像文件（例如 `.png`），保存在指定的 `--output_path`。该图像显示了分割后的晶粒，每个晶粒根据其在指定 `--color_feature` 上的值进行着色。图中包含一个颜色条，用于指示颜色和特征值之间的映射关系。背景（非晶粒区域）通常显示为白色或其他中性色。

#### 解读与材料学意义

*   **空间聚类**: 寻找具有相似特征值（例如，高 `neighbor_count`、大 `area`、特定 `orientation`）的晶粒聚集的区域。微观结构在此特征方面是否表现出异质性？
*   **梯度**: 观察样本中特征值是否存在梯度（例如，靠近一个边缘的晶粒较小，靠近另一个边缘的晶粒较大）。
*   **与位置的相关性**: 具有特定特征值的晶粒是否优先位于某些微观结构元素附近（例如，样品边缘、大孔隙、特定相，如果可识别）？
*   **各向异性**: 如果按 `orientation` 着色，可视化是否揭示了材料内部的择优取向或织构？
*   **比较**: 对同一样本使用不同特征生成这些图，观察空间模式如何关联。比较不同样本（例如，高/低 Hc20）的相同特征图，以识别空间组织上的系统性差异。

这种可视化提供了定量晶粒特征的直接空间映射，有助于弥合统计特征值与材料内部实际物理排列之间的差距。

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
