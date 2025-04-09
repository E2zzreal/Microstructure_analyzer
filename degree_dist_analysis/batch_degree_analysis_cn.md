# `batch_degree_analysis.py` 中文文档

## 目的

该脚本用于批量执行晶粒度分析脚本，自动处理多个样本并生成所有分析结果。主要功能包括：

- 自动匹配每个样本的mask文件和details文件
- 并行执行所有分析脚本
- 将结果按脚本名称分类保存
- 统一使用`delaunay_degree_adaptive_2r0_5std`作为默认度数特征

## 先决条件

- 多个`.mask`文件（样本分割结果）
- 对应的`_details.csv`文件（包含晶粒特征）
- Python 3.6+环境
- 所有分析脚本位于同一目录下

## 使用方法

```bash
python batch_degree_analysis.py --mask_dir <mask文件目录> \
                                --details_dir <details文件目录> \
                                [--output_dir <输出目录>] \
                                [--max_workers <并行数>]
```

## 参数说明

*   `--mask_dir` (必需): 包含`.mask`文件的目录路径
*   `--details_dir` (必需): 包含`_details.csv`文件的目录路径
*   `--output_dir` (可选): 输出结果目录。默认值：`'results'`
*   `--max_workers` (可选): 最大并行任务数。默认值：`4`

## 输出结构

脚本会在输出目录下为每个分析脚本创建子目录，结构如下：

```
results/
├── plot_degree_histogram/
│   ├── sample1_hist.png
│   └── sample2_hist.png
├── plot_degree_vs_property/
│   ├── sample1_vs_property.png
│   └── sample2_vs_property.png
├── visualize_network/
│   ├── sample1_network.png
│   └── sample2_network.png
└── ...其他脚本输出目录
```

## 默认分析参数

所有分析脚本统一使用以下默认参数：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| 度数特征 | delaunay_degree_adaptive_2r0_5std | 基于Delaunay图的自适应度数 |
| 异常值范围 | min_degree=2, max_degree=15 | 可视化异常连通性晶粒 |
| 网络距离阈值 | 100像素 | 定义晶粒连接的最大距离 |
| 属性对比 | area | 度数vs面积关系 |

## 注意事项

1. 日志文件`batch_degree_analysis.log`会记录所有执行过程
2. 确保mask文件和details文件命名一致（如`sample1.mask`对应`*sample1_details.csv`）
3. 并行处理数量应根据CPU核心数合理设置
4. 输出目录结构会按分析脚本自动创建
5. 所有绘图结果会添加分析类型后缀（如`_hist`, `_network`等）

## 示例工作流

1. 准备数据：
   ```bash
   python ../run_feature_extraction.py --mask_dir input/masks \
                                      --output_csv features.csv \
                                      --save_grain_details \
                                      --details_output_dir output/details
   ```

2. 批量分析：
   ```bash
   python batch_degree_analysis.py --mask_dir input/masks \
                                  --details_dir output/details \
                                  --output_dir analysis_results \
                                  --max_workers 8
   ```

3. 检查结果：
   - 查看`analysis_results`目录下的各类分析结果
   - 检查`batch_degree_analysis.log`了解执行情况
