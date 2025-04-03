import argparse
import os
import sys
import time
import pandas as pd

# Ensure the package directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.feature_extraction import batch_calculate_features
except ImportError as e:
    print(f"Error importing feature_extraction module: {e}")
    print("Ensure the script is run from the project root directory containing 'microstructure_analyzer'.")
    sys.exit(1)

def run_feature_extraction_pipeline():
    parser = argparse.ArgumentParser(description="Run Microstructure Feature Extraction")

    # Changed required=True to required=False and added r prefix to default path
    parser.add_argument('--mask_dir', required=False, default = r'D:\TTRS\1-mag\2-ZH\0-ImageProcessing\2-Data\2-SEM\corrosed\20250228\masks', help='Directory containing input .mask files.')
    # Changed required=True to required=False and added r prefix to default path
    parser.add_argument('--output_csv', required=False, default=r'results\all_features.csv', help='Path to save the aggregate extracted features CSV file.')
    # New arguments for saving per-grain details
    parser.add_argument('--save_grain_details', action='store_true', help='If set, save detailed features for each grain to separate CSV files.')
    parser.add_argument('--details_output_dir', default=r'results\per_grain_details', help='Directory to save the per-grain detail CSV files (used only if --save_grain_details is set).')

    args = parser.parse_args()

    print("--- Starting Feature Extraction ---")
    start_time = time.time()
    print(f"Input Mask Directory: {args.mask_dir}")
    print(f"Output Aggregate Features CSV: {args.output_csv}")
    if args.save_grain_details:
        print(f"Saving Per-Grain Details: Yes")
        print(f"Details Output Directory: {args.details_output_dir}")
    else:
        print(f"Saving Per-Grain Details: No")


    # Validate paths
    if not os.path.isdir(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        sys.exit(1)
    if not os.listdir(args.mask_dir):
         print(f"Warning: Mask directory '{args.mask_dir}' is empty.")
         # Decide if this is an error or just means no work to do
         # sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir: # Ensure output_dir is not empty (e.g., if filename is in root)
        os.makedirs(output_dir, exist_ok=True)
    if args.save_grain_details and args.details_output_dir:
         os.makedirs(args.details_output_dir, exist_ok=True)


    try:
        # Pass the new arguments to batch_calculate_features
        features_df = batch_calculate_features(
            args.mask_dir,
            save_details=args.save_grain_details,
            details_folder=args.details_output_dir
        )

        if not features_df.empty:
            features_df.to_csv(args.output_csv)
            print(f"\nAggregate features successfully extracted and saved to: {args.output_csv}")
            print(f"Aggregate DataFrame shape: {features_df.shape}")
        else:
            print("\nWarning: No features were extracted (possibly no valid .mask files found or processed).")

    except Exception as e:
        print(f"\n--- Feature Extraction Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- Feature Extraction Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_feature_extraction_pipeline()
