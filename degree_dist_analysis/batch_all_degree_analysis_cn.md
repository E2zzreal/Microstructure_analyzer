# `batch_all_degree_analysis.py` 中文文档

## 目的

该脚本是对`batch_degree_analysis.py`的增强版本，用于批量执行所有晶粒度分析脚本，并且支持所有绘图方式。主要功能包括：

- 自动匹配每个样本的mask文件和details文件
- 并行执行所有分析脚本
- 对支持多种绘图方式的脚本，执行所有绘图方式
- 将绘图方式添加到输出文件名中
- 将结果按脚本名称分类保存
- 统一使用`delaunay_degree_adaptive_2r0_5std`作为默认度数特征
- 特别处理`compare_degree_distributions.py`，因为它分析的是文件夹而不是单个样品

## 先决条件

- 多个`.mask`文件（样本分割结果）
- 对应的`_details.csv`文件（包含晶粒特征）
- Python 3.6+环境
- 所有分析脚本位于同一目录下

## 用法

```bash
python batch_all_degree_analysis.py --mask_dir <mask文件目录> \
                                  --details_dir <details文件目录> \
                                  --output_dir <输出目录> \
                                  --max_workers <并行任务数> \
                                  --skip_compare
```

### 参数说明

- `--mask_dir`：包含`.mask`文件的目录路径（必需）
- `--details_dir`：包含`_details.csv`文件的目录路径（必需）
- `--output_dir`：输出结果目录，默认为`results`
- `--max_workers`：最大并行任务数，默认为4
- `--skip_compare`：跳过比较分布分析（可选标志）

## 输出

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

## 注意事项

1. 脚本会依次处理每个样本，调用所有分析脚本（包括多种绘图方式），最后处理比较分布分析（除非使用--skip_compare跳过）
2. 日志文件`batch_all_degree_analysis.log`会记录所有执行过程
2. 确保mask文件和details文件命名一致（如`sample1.mask`对应`*sample1_details.csv`）
3. 并行处理数量应根据CPU核心数合理设置
4. 输出目录结构会按分析脚本自动创建
5. 所有绘图结果会添加分析类型和绘图方式后缀（如`_hist_hist`, `_hist_bar`, `_network`等）
6. 默认度数特征已更改为`delaunay_degree_adaptive_2r0_5std`，以更好地反映基于Delaunay图的晶粒连通性分析

## 示例

```bash
python batch_all_degree_analysis.py --mask_dir input/masks \
                                  --details_dir input/details \
                                  --output_dir results \
                                  --max_workers 8
```

这将处理`input/masks`目录中的所有`.mask`文件，使用`input/details`目录中的对应`_details.csv`文件，并将结果保存在`results`目录下。