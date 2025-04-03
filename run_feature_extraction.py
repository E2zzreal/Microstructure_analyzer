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

    parser.add_argument('--mask_dir', required=True, help='Directory containing input .mask files.')
    parser.add_argument('--output_csv', required=True, help='Path to save the extracted features CSV file.')

    args = parser.parse_args()

    print("--- Starting Feature Extraction ---")
    start_time = time.time()
    print(f"Input Mask Directory: {args.mask_dir}")
    print(f"Output Features CSV: {args.output_csv}")

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

    try:
        features_df = batch_calculate_features(args.mask_dir)

        if not features_df.empty:
            features_df.to_csv(args.output_csv)
            print(f"\nFeatures successfully extracted and saved to: {args.output_csv}")
            print(f"DataFrame shape: {features_df.shape}")
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
