import argparse
import os
import sys
import time
import pandas as pd

# Ensure the package directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.analysis import (
        prepare_analysis_dataframe, calculate_icc, perform_anova,
        filter_features_icc_anova, filter_features_variance,
        filter_features_correlation
    )
except ImportError as e:
    print(f"Error importing analysis module: {e}")
    print("Ensure the script is run from the project root directory containing 'microstructure_analyzer'.")
    sys.exit(1)

def run_analysis_pipeline():
    parser = argparse.ArgumentParser(description="Run Microstructure Feature Analysis and Filtering")

    parser.add_argument('--input_csv', required=False, default=r'results\all_features.csv',help='Path to the raw extracted features CSV file.')
    parser.add_argument('--final_output_csv', required=False, default=r'results\filterd_features.csv', help='Path to save the final filtered features (mean per sample) CSV file.')
    parser.add_argument('--filtered_icc_anova_output', help='(Optional) Path to save intermediate ICC/ANOVA filter results CSV.')

    # Filtering parameters
    parser.add_argument('--icc_threshold', type=float, default=0.75, help='ICC threshold for feature filtering (default: 0.75).')
    parser.add_argument('--corr_threshold', type=float, default=0.95, help='Correlation threshold for feature filtering (default: 0.95).')
    parser.add_argument('--max_raters', type=int, default=8, help='Max raters/measurements per sample for ICC (default: 8).')
    parser.add_argument('--anova_alpha', type=float, default=0.05, help='Significance level for ANOVA (default: 0.05).')
    parser.add_argument('--variance_threshold', type=float, default=0.0, help='Variance threshold for feature filtering (default: 0.0).')

    args = parser.parse_args()

    print("--- Starting Feature Analysis and Filtering ---")
    start_time = time.time()
    print(f"Input Raw Features CSV: {args.input_csv}")
    print(f"Output Final Features CSV: {args.final_output_csv}")
    if args.filtered_icc_anova_output:
        print(f"Output ICC/ANOVA Results: {args.filtered_icc_anova_output}")
    print(f"Parameters: ICC>={args.icc_threshold}, ANOVA_alpha={args.anova_alpha}, MaxRaters={args.max_raters}, VarThresh={args.variance_threshold}, Corr<{args.corr_threshold}")


    # Validate paths
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}")
        sys.exit(1)

    # Create output directories if they don't exist
    final_output_dir = os.path.dirname(args.final_output_csv)
    if final_output_dir:
        os.makedirs(final_output_dir, exist_ok=True)
    if args.filtered_icc_anova_output:
        icc_anova_output_dir = os.path.dirname(args.filtered_icc_anova_output)
        if icc_anova_output_dir:
            os.makedirs(icc_anova_output_dir, exist_ok=True)

    try:
        # Load raw features
        raw_features_df = pd.read_csv(args.input_csv, index_col=0) # Assuming index is sample ID from extraction step
        if raw_features_df.empty:
             print("Error: Input features file is empty.")
             sys.exit(1)

        # Prepare data for ICC/ANOVA
        analysis_df = prepare_analysis_dataframe(raw_features_df)
        if analysis_df.empty:
            raise ValueError("Failed to prepare DataFrame for analysis (e.g., no numeric features).")

        all_feature_names = analysis_df.columns[2:].tolist() # Exclude Sample, Rater

        # --- ICC/ANOVA Filtering ---
        print("\nRunning ICC/ANOVA filtering...")
        icc_df = calculate_icc(analysis_df, all_feature_names, max_raters=args.max_raters)
        anova_df = perform_anova(analysis_df, all_feature_names, alpha=args.anova_alpha)
        icc_anova_results = filter_features_icc_anova(icc_df, anova_df, icc_threshold=args.icc_threshold)

        if args.filtered_icc_anova_output and not icc_anova_results.empty:
            icc_anova_results.to_csv(args.filtered_icc_anova_output)
            print(f"ICC/ANOVA filter results saved to: {args.filtered_icc_anova_output}")

        if not icc_anova_results.empty:
            good_features_icc_anova = icc_anova_results[icc_anova_results['Good_Feature']]['Feature'].tolist()
            print(f"Features passing ICC & ANOVA ({len(good_features_icc_anova)}): {good_features_icc_anova}")
        else:
            print("Warning: ICC/ANOVA filtering produced no results. Proceeding with all features for subsequent steps.")
            good_features_icc_anova = all_feature_names # Use all features if filtering fails

        if not good_features_icc_anova:
            print("Error: No features passed the ICC/ANOVA filter. Stopping analysis.")
            sys.exit(1)

        # --- Variance/Correlation Filtering (on mean features) ---
        print("\nRunning Variance/Correlation filtering...")
        # Calculate mean features per sample for the features that passed ICC/ANOVA
        mean_features_df = analysis_df.groupby('Sample')[good_features_icc_anova].mean()

        # Variance Threshold
        features_pass_variance = filter_features_variance(mean_features_df, threshold=args.variance_threshold)
        if not features_pass_variance:
            print("Error: No features passed the variance threshold filter. Stopping analysis.")
            sys.exit(1)
        print(f"Features passing Variance Threshold ({len(features_pass_variance)}).")


        # Correlation Threshold
        final_feature_list = filter_features_correlation(mean_features_df[features_pass_variance], threshold=args.corr_threshold)
        if not final_feature_list:
            print("Error: No features passed the correlation filter. Stopping analysis.")
            sys.exit(1)
        print(f"Final selected features after Correlation Filter ({len(final_feature_list)}).")


        # --- Save Final Features ---
        final_features_df_mean = mean_features_df[final_feature_list]
        final_features_df_mean.to_csv(args.final_output_csv)
        print(f"\nFinal filtered features (mean per sample) saved to: {args.final_output_csv}")
        print(f"Final DataFrame shape: {final_features_df_mean.shape}")


    except Exception as e:
        print(f"\n--- Feature Analysis Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- Feature Analysis Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_analysis_pipeline()
