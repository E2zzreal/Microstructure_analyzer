import os
import glob
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='batch_all_degree_analysis.log'
)

# 分析脚本配置
SCRIPT_CONFIG = {
    'plot_degree_histogram.py': {
        'output_suffix': '_hist',
        'default_args': {
            '--degree_column': 'delaunay_degree_adaptive_2r0_5std',
        },
        'plot_types': {
            'hist': '_hist',
            'bar': '_bar'
        }
    },
    'plot_degree_vs_property.py': {
        'output_suffix': '_vs_property',
        'default_args': {
            '--degree': 'delaunay_degree_adaptive_2r0_5std',
            '--property': 'area'
        },
        'plot_types': {
            'scatter': '_scatter',
            'hexbin': '_hexbin',
            'kde': '_kde'
        }
    },
    'visualize_network.py': {
        'output_suffix': '_network',
        'default_args': {
            '--color_feature': 'delaunay_degree_adaptive_2r0_5std',
            '--distance_threshold': '100'
        }
    },
    'visualize_spatial_coloring.py': {
        'output_suffix': '_spatial',
        'default_args': {
            '--color_feature': 'delaunay_degree_adaptive_2r0_5std'
        }
    },
    'visualize_degree_outliers.py': {
        'output_suffix': '_outliers',
        'default_args': {
            '--degree_column': 'delaunay_degree_adaptive_2r0_5std',
            '--min_degree': '2',
            '--max_degree': '15'
        }
    },
    'compare_degree_distributions.py': {
        'output_suffix': '_compare',
        'default_args': {
            '--degree_column': 'delaunay_degree_adaptive_2r0_5std',
        },
        'plot_types': {
            'kde': '_kde',
            'box': '_box',
            'violin': '_violin',
            'hist': '_hist'
        },
        'is_folder_analysis': True  # 标记这个脚本是分析文件夹而不是单个样品
    }
}

def find_details_file(mask_path, details_dir):
    """根据mask路径查找对应的details文件"""
    # 提取基本文件名（不含路径和扩展名）
    base_name = os.path.basename(mask_path)
    sample_id_from_mask = os.path.splitext(base_name)[0]
    parent_dir = os.path.basename(os.path.dirname(mask_path))

    # 优先尝试更精确的模式，包含父目录名
    # 形如 masks/any_dir/1_0001.mask 对应 *any_dir_1_0001_details.csv
    if parent_dir:
        precise_pattern = os.path.join(details_dir, f'*{parent_dir}_{sample_id_from_mask}_details.csv')
        precise_matches = glob.glob(precise_pattern)
        if precise_matches:
            # 确保只返回一个最可能的匹配（如果存在多个，可能需要更复杂的逻辑）
            return precise_matches[0]

    # 如果精确模式未找到，再尝试较宽泛的模式 *{sample_id}_details.csv
    # 这可能在 details 文件名不包含父目录名时有用，但也可能导致错误匹配
    broad_pattern = os.path.join(details_dir, f'*{sample_id_from_mask}_details.csv')
    broad_matches = glob.glob(broad_pattern)
    if broad_matches:
        # 警告：这种匹配可能不准确，如果存在多个同名文件
        logging.warning(f"Found details file using broad pattern for {mask_path}: {broad_matches[0]}. Verify correctness.")
        return broad_matches[0]

    # 如果两种模式都找不到，返回None
    return None

def run_analysis_script(script_name, args):
    """执行单个分析脚本"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = ['python', script_path] + args
    logging.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Success: {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {script_name}\nStderr: {e.stderr}\nStdout: {e.stdout}") # Log both stderr and stdout
        return False

def process_sample(mask_path, details_dir, output_root, max_workers=4):
    """处理单个样本"""
    details_path = find_details_file(mask_path, details_dir)
    if not details_path:
        logging.warning(f"No details file found for mask: {mask_path}")
        return

    # Correctly extract sample_id including parent directory if applicable
    # e.g., data/masks/23/1_0001.mask -> 23_1_0001
    base_name_no_ext = os.path.splitext(os.path.basename(mask_path))[0]
    parent_dir_name = os.path.basename(os.path.dirname(mask_path))
    # Check if the parent directory is the same as the input mask_dir root
    # If mask is directly in mask_dir, parent_dir_name will be the last part of mask_dir path
    # A more robust check might compare the full dirname with the input mask_dir
    # For simplicity, assume if parent_dir_name is not empty and likely a sample identifier, prepend it.
    # This logic assumes mask files are either directly in mask_dir or one level down (e.g., mask_dir/sample_id/image.mask)
    # Adjust if the structure is deeper or different.
    input_mask_dir_basename = os.path.basename(os.path.abspath(os.path.join(mask_path, "..", ".."))) # Get the assumed root mask dir name
    if parent_dir_name and parent_dir_name != input_mask_dir_basename:
         sample_id = f"{parent_dir_name}_{base_name_no_ext}"
    else:
         sample_id = base_name_no_ext # Mask file is directly in mask_dir

    logging.info(f"Derived sample_id for output: {sample_id}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for script_name, config in SCRIPT_CONFIG.items():
            # 跳过文件夹分析脚本（如compare_degree_distributions.py）
            if config.get('is_folder_analysis', False):
                continue
                
            output_dir = os.path.join(output_root, os.path.splitext(script_name)[0])
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查脚本是否支持多种绘图方式
            if 'plot_types' in config:
                # 对每种绘图方式执行一次脚本
                for plot_type, type_suffix in config['plot_types'].items():
                    output_path = os.path.join(
                        output_dir,
                        f"{sample_id}{config['output_suffix']}{type_suffix}.png"
                    )
                    
                    # Base arguments for all scripts
                    args = [
                        '--details_csv', details_path,
                        '--output_path', output_path,
                    ]
                    # Add --mask_path only if the script needs it (visualize scripts)
                    if script_name in ['visualize_network.py', 'visualize_spatial_coloring.py', 'visualize_degree_outliers.py']:
                         args.extend(['--mask_path', mask_path])
                    
                    # Add plot_type if applicable
                    args.extend(['--plot_type', plot_type])

                    # 添加脚本特定参数
                    for arg, value in config['default_args'].items():
                        args.extend([arg, str(value)])
                    
                    futures.append(executor.submit(run_analysis_script, script_name, args))
            else:
                # 不支持多种绘图方式的脚本只执行一次
                output_path = os.path.join(
                    output_dir,
                    f"{sample_id}{config['output_suffix']}.png"
                )
                
                 # Base arguments for all scripts
                args = [
                    '--details_csv', details_path,
                    '--output_path', output_path
                ]
                # Add --mask_path only if the script needs it (visualize scripts)
                if script_name in ['visualize_network.py', 'visualize_spatial_coloring.py', 'visualize_degree_outliers.py']:
                     args.extend(['--mask_path', mask_path])

                # 添加脚本特定参数
                for arg, value in config['default_args'].items():
                    args.extend([arg, str(value)])
                
                futures.append(executor.submit(run_analysis_script, script_name, args))
        
        # 等待所有任务完成
        for future in futures:
            future.result()

def process_compare_distributions(details_dir, output_root):
    """处理比较分布脚本（compare_degree_distributions.py）"""
    script_name = 'compare_degree_distributions.py'
    config = SCRIPT_CONFIG[script_name]
    
    output_dir = os.path.join(output_root, os.path.splitext(script_name)[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # 对每种绘图方式执行一次脚本
    for plot_type, type_suffix in config['plot_types'].items():
        output_path = os.path.join(
            output_dir,
            f"comparison{config['output_suffix']}{type_suffix}.png"
        )
        
        args = [
            '--details_folder', details_dir,
            '--output_path', output_path,
            '--plot_type', plot_type
        ]
        
        # 添加脚本特定参数
        for arg, value in config['default_args'].items():
            args.extend([arg, str(value)])
        
        run_analysis_script(script_name, args)

def main():
    parser = argparse.ArgumentParser(
        description='批量执行所有晶粒度分析脚本，包括所有绘图方式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mask_dir', required=True,
                       help='包含.mask文件的目录路径')
    parser.add_argument('--details_dir', required=True,
                       help='包含_details.csv文件的目录路径')
    parser.add_argument('--output_dir', default=r'results\batch_all_degree_analysis',
                       help='输出结果目录')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大并行任务数')
    parser.add_argument('--skip_compare', action='store_true',
                       help='跳过比较分布分析')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有mask文件 (递归搜索)
    mask_files = glob.glob(os.path.join(args.mask_dir, '**/*.mask'), recursive=True)
    if not mask_files:
        logging.error(f"No .mask files found recursively in {args.mask_dir}")
        return
    print(len(mask_files))
    for mask_path in mask_files:
        logging.info(f"Processing: {mask_path}")
        process_sample(
            mask_path,
            args.details_dir,
            args.output_dir,
            args.max_workers
        )
    
    # 处理比较分布分析（如果未跳过）
    if not args.skip_compare:
        logging.info("Processing compare_degree_distributions.py")
        process_compare_distributions(args.details_dir, args.output_dir)

if __name__ == '__main__':
    main()
