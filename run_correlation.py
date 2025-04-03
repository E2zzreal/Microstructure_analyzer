import argparse
import os
import sys
import time
import pandas as pd

# Ensure the package directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.correlation import load_data, merge_data, calculate_correlation
except ImportError as e:
    print(f"Error importing correlation module: {e}")
    print("Ensure the script is run from the project root directory containing 'microstructure_analyzer'.")
    sys.exit(1)

def run_correlation_pipeline():
    parser = argparse.ArgumentParser(description="Run Correlation Analysis between Microstructure Features and External Data")

    parser.add_argument('--input_features_csv', required=True, help='Path to the final filtered features CSV file.')
    parser.add_argument('--external_data_csv', required=True, help='Path to the external data (properties, process, etc.) CSV file.')
    parser.add_argument('--output_txt', required=True, help='Path to save the correlation analysis results text file.')

    # Define target column groups as arguments (optional, could also be hardcoded or read from config)
    # Example: --targets_props Br20 Hc20 Hc147 Hr20 Hr147 --targets_pps C F J L ...
    # For simplicity, we'll use the hardcoded groups from the notebook for now.

    args = parser.parse_args()

    print("--- Starting Correlation Analysis ---")
    start_time = time.time()
    print(f"Input Features CSV: {args.input_features_csv}")
    print(f"External Data CSV: {args.external_data_csv}")
    print(f"Output Results TXT: {args.output_txt}")

    # Validate paths
    if not os.path.exists(args.input_features_csv):
        print(f"Error: Input features CSV file not found: {args.input_features_csv}")
        sys.exit(1)
    if not os.path.exists(args.external_data_csv):
        print(f"Error: External data CSV file not found: {args.external_data_csv}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Load data
        features_df = load_data(args.input_features_csv, index_col=0) # Assumes index is sample ID
        external_df = load_data(args.external_data_csv, index_col=0) # Assumes index is sample ID

        if features_df.empty:
             print("Error: Loaded features data is empty.")
             sys.exit(1)
        if external_df.empty:
             print("Error: Loaded external data is empty.")
             sys.exit(1)

        # Merge data
        merged_data = merge_data(features_df, external_df)

        if merged_data.empty:
            print("Error: Merged data is empty (no common samples found). Cannot perform correlation.")
            sys.exit(1)

        print(f"\nMerged data shape for correlation: {merged_data.shape}")

        # Define feature and target columns based on notebook
        feature_cols = features_df.columns.tolist() # Use all columns from the filtered features file
        # Define target groups (hardcoded based on notebook)
        props_cols = ['Br20','Hc20','Hc147','Hr20','Hr147']
        pps_cols = ['C','F','J','L']
        effs_cols = ['Neff','alpha']
        pcs_cols = ['PC5-X10000A']
        target_groups = {'props': props_cols, 'pps': pps_cols, 'effs': effs_cols, 'pcs': pcs_cols}

        correlation_results = {}
        print("\nCalculating correlations...")

        with open(args.output_txt, 'w') as f:
            for group_name, target_cols in target_groups.items():
                f.write(f"--- Correlation: Features vs. {group_name.upper()} ---\n")
                print(f" Correlating with {group_name.upper()}...")

                # Ensure target columns exist in merged data
                valid_targets = [col for col in target_cols if col in merged_data.columns]
                if not valid_targets:
                     warning_msg = f"Warning: None of the target columns for group '{group_name}' found in merged data. Skipping."
                     print(warning_msg)
                     f.write(warning_msg + "\n\n")
                     continue

                # Calculate correlation
                corr_df = calculate_correlation(merged_data, feature_cols, valid_targets)

                if not corr_df.empty:
                    # Sort by the first target column in the group for consistent display (optional)
                    sort_col = valid_targets[0]
                    sorted_corr_df = corr_df.sort_values(by=sort_col, ascending=False)
                    print(f"Correlation results for {group_name} (sorted by {sort_col}):")
                    print(sorted_corr_df)
                    f.write(sorted_corr_df.to_string())
                    f.write("\n\n")
                    correlation_results[group_name] = sorted_corr_df # Store sorted results if needed elsewhere
                else:
                    error_msg = f"Could not calculate correlation for group '{group_name}'."
                    print(error_msg)
                    f.write(error_msg + "\n\n")

        print(f"\nCorrelation results saved to: {args.output_txt}")


    except Exception as e:
        print(f"\n--- Correlation Analysis Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- Correlation Analysis Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_correlation_pipeline()
