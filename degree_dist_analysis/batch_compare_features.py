import argparse
import os
import sys

# Add project root to path to allow importing compare_degree_distributions
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Assuming compare_degree_distributions.py is in the same directory
    from .compare_degree_distributions import compare_degree_distributions
except ImportError:
    print("Error: Could not import compare_degree_distributions function.")
    print("Ensure batch_compare_features.py and compare_degree_distributions.py are in the same directory ('degree_dist_analysis').")
    sys.exit(1)

def batch_compare(details_folder, output_base_folder, features_to_plot):
    """
    Generates comparison plots for multiple features and plot types.

    Args:
        details_folder (str): Path to the folder containing _details.csv files.
        output_base_folder (str): Path to the base folder where plots will be saved.
        features_to_plot (list): List of feature column names to plot.
    """
    plot_types = ['kde', 'box', 'violin', 'hist']

    # Ensure the output base directory exists
    os.makedirs(output_base_folder, exist_ok=True)
    print(f"Output directory set to: {output_base_folder}")

    print(f"\nStarting batch comparison for {len(features_to_plot)} features and {len(plot_types)} plot types...")

    for feature in features_to_plot:
        print(f"\n--- Processing Feature: {feature} ---")
        for plot_type in plot_types:
            output_filename = f"{feature}_{plot_type}.png"
            output_path = os.path.join(output_base_folder, output_filename)
            print(f"  Generating {plot_type} plot for '{feature}' -> {output_path}")

            try:
                # Call the imported function
                compare_degree_distributions(
                    details_folder=details_folder,
                    output_path=output_path,
                    degree_column=feature, # Use the current feature from the list
                    plot_type=plot_type
                )
                print(f"  Successfully saved: {output_path}")
            except FileNotFoundError as fnf_err:
                 print(f"  Error processing {feature} ({plot_type}): Input file/folder not found - {fnf_err}")
            except KeyError as key_err:
                 print(f"  Error processing {feature} ({plot_type}): Feature '{key_err}' likely not found in some CSV files. Skipping this plot.")
            except Exception as e:
                print(f"  Error generating plot for {feature} ({plot_type}): {e}")
                # Optionally continue to the next plot or stop
                # continue

    print("\nBatch comparison finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate comparison plots for multiple features.")
    parser.add_argument('--details_folder', required=True, help='Path to the folder containing _details.csv files (e.g., results/per_grain_details).')
    parser.add_argument('--output_base_folder', required=True, help='Base path to save the output comparison plots (e.g., results/batch_compare_features).')
    # Optional: Allow specifying features via command line, otherwise use a default list
    parser.add_argument('--features', nargs='+', default=[
                                                            'area', 'perimeter',
                                                            'aspect_ratio', 'circularity',
                                                            'delaunay_degree_adaptive_2r0_5std', # Example degree feature
                                                            'neighbor_count' # Another common one
                                                            ],
                        help='List of feature column names from details CSV to plot.')

    args = parser.parse_args()

    # Use the provided or default list of features
    features_list = args.features
    print(f"Features to plot: {features_list}")

    batch_compare(
        details_folder=args.details_folder,
        output_base_folder=args.output_base_folder,
        features_to_plot=features_list
    )
