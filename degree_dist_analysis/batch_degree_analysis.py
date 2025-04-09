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
    filename='batch_degree_analysis.log'
)

# 分析脚本配置
SCRIPT_CONFIG = {
    'plot_degree_histogram.py': {
        'output_suffix': '_hist',
        'default_args': {
            '--degree_column': 'delaunay_degree_adaptive_2r0_5std',
            '--plot_type': 'hist'
        }
    },
    'plot_degree_vs_property.py': {
        'output_suffix': '_vs_property',
        'default_args': {
            '--degree': 'delaunay_degree_adaptive_2r0_5std',
            '--property': 'area'
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
            '--degree_column': 'delaunay_degree_adaptive_2r0_5std'
        }
    }
}

def find_details_file(mask_path, details_dir):
    """根据mask路径查找对应的details文件"""
    base_name = os.path.basename(mask_path)
    sample_id = os.path.splitext(base_name)[0]
    pattern = os.path.join(details_dir, f'*{sample_id}_details.csv')
    matches = glob.glob(pattern)
    return matches[0] if matches else None

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
        logging.error(f"Failed: {script_name}\nError: {e.stderr}")
        return False

def process_sample(mask_path, details_dir, output_root, max_workers=4):
    """处理单个样本"""
    details_path = find_details_file(mask_path, details_dir)
    if not details_path:
        logging.warning(f"No details file found for mask: {mask_path}")
        return

    sample_id = os.path.splitext(os.path.basename(mask_path))[0]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for script_name, config in SCRIPT_CONFIG.items():
            output_dir = os.path.join(output_root, os.path.splitext(script_name)[0])
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(
                output_dir,
                f"{sample_id}{config['output_suffix']}.png"
            )
            
            args = [
                '--mask_path', mask_path,
                '--details_csv', details_path,
                '--output_path', output_path
            ]
            
            # 添加脚本特定参数
            for arg, value in config['default_args'].items():
                args.extend([arg, str(value)])
            
            futures.append(executor.submit(run_analysis_script, script_name, args))
        
        # 等待所有任务完成
        for future in futures:
            future.result()

def main():
    parser = argparse.ArgumentParser(
        description='批量执行晶粒度分析脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mask_dir', required=True,
                       help='包含.mask文件的目录路径')
    parser.add_argument('--details_dir', required=True,
                       help='包含_details.csv文件的目录路径')
    parser.add_argument('--output_dir', default='results',
                       help='输出结果目录')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大并行任务数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有mask文件
    mask_files = glob.glob(os.path.join(args.mask_dir, '*.mask'))
    if not mask_files:
        logging.error(f"No .mask files found in {args.mask_dir}")
        return
    
    for mask_path in mask_files:
        logging.info(f"Processing: {mask_path}")
        process_sample(
            mask_path,
            args.details_dir,
            args.output_dir,
            args.max_workers
        )

if __name__ == '__main__':
    main()
