import argparse
import os
import sys
import time

# Ensure the package directory is in the Python path
# This allows importing from microstructure_analyzer
# Assuming the script is run from the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.segmentation import batch_save_masks
except ImportError as e:
    print(f"Error importing segmentation module: {e}")
    print("Ensure the script is run from the project root directory containing 'microstructure_analyzer'.")
    sys.exit(1)

def run_segmentation_pipeline():
    parser = argparse.ArgumentParser(description="Run Microstructure Image Segmentation")

    parser.add_argument('--input_dir', required=True, help='Directory containing input .tif images.')
    parser.add_argument('--mask_dir', required=True, help='Directory to save .mask files.')
    parser.add_argument('--sam_checkpoint', required=True, help='Path to SAM2 model checkpoint (.pt).')
    parser.add_argument('--sam_config', required=True, help='Path to SAM2 model config (.yaml).')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for SAM2 model (default: cuda).')
    # Add other relevant parameters from batch_save_masks if needed (e.g., min_area, area_ratio)
    parser.add_argument('--min_area', type=int, default=100, help='Minimum mask area in pixels for deduplication (default: 100).')
    parser.add_argument('--area_ratio', type=float, default=None, help='Minimum mask area as ratio of image area (optional).')


    args = parser.parse_args()

    print("--- Starting Segmentation ---")
    start_time = time.time()
    print(f"Input Image Directory: {args.input_dir}")
    print(f"Output Mask Directory: {args.mask_dir}")
    print(f"SAM Checkpoint: {args.sam_checkpoint}")
    print(f"SAM Config: {args.sam_config}")
    print(f"Device: {args.device}")
    print(f"Min Area: {args.min_area}, Area Ratio: {args.area_ratio}")


    # Validate paths
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {args.sam_checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.sam_config):
        print(f"Error: SAM config not found: {args.sam_config}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.mask_dir, exist_ok=True)

    try:
        batch_save_masks(
            input_dir=args.input_dir,
            output_dir=args.mask_dir,
            sam_model_cfg=args.sam_config,
            sam_checkpoint_path=args.sam_checkpoint,
            device_str=args.device,
            min_area=args.min_area,
            area_ratio=args.area_ratio
            # Pass other parameters if added to batch_save_masks and parser
        )
    except Exception as e:
        print(f"\n--- Segmentation Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- Segmentation Complete ---")
    print(f"Masks saved to: {args.mask_dir}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_segmentation_pipeline()
