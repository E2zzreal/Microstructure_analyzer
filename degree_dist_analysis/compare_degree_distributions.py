import argparse
import os
import sys
import warnings
import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

def compare_degree_distributions(details_folder, output_path, file_pattern='*_details.csv',
                                 degree_column='neighbor_count', plot_type='kde',
                                 sample_labels=None):
    """
    Compares grain degree distributions across multiple samples or groups.

    Args:
        details_folder (str): Path to the folder containing _details.csv files.
        output_path (str): Path to save the output comparison plot image.
        file_pattern (str): Glob pattern to find the details files within the folder.
                            Defaults to '*_details.csv'.
        degree_column (str): Name of the column containing the degree information.
                             Defaults to 'neighbor_count'.
        plot_type (str): Type of comparison plot ('kde', 'box', 'violin', 'hist').
                         Defaults to 'kde'.
        sample_labels (list, optional): List of labels corresponding to each file found.
                                        If None, uses filenames (or parts) as labels.
                                        Defaults to None.
    """
    print(f"Searching for details files in: {details_folder} using pattern: {file_pattern}")
    search_path = os.path.join(details_folder, file_pattern)
    details_files = sorted(glob.glob(search_path))

    if not details_files:
        print(f"Error: No details files found matching pattern '{file_pattern}' in '{details_folder}'."); sys.exit(1)
    print(f"Found {len(details_files)} details files.")

    all_data = []
    processed_labels = []

    for i, file_path in enumerate(details_files):
        print(f"  Loading: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            if df.empty or degree_column not in df.columns:
                print(f"Warning: Skipping file {file_path} due to missing data or degree column '{degree_column}'.")
                continue

            degree_data = df[degree_column].dropna()
            if degree_data.empty:
                 print(f"Warning: Skipping file {file_path} due to no valid degree data.")
                 continue

            # Determine label
            if sample_labels and i < len(sample_labels):
                label = sample_labels[i]
            else:
                # Extract label from filename (e.g., 'sampleID_imagename_details.csv' -> 'sampleID')
                base_name = os.path.basename(file_path)
                label_match = re.match(r"([^_]+)", base_name) # Assumes sample ID is before first underscore
                label = label_match.group(1) if label_match else f"Sample_{i+1}"
            processed_labels.append(label)

            # Store data with labels
            temp_df = pd.DataFrame({degree_column: degree_data, 'Sample': label})
            all_data.append(temp_df)

        except Exception as e:
            print(f"Warning: Failed to load or process file {file_path}: {e}")

    if not all_data:
        print("Error: No valid data loaded from any file. Cannot generate comparison plot.")
        sys.exit(1)

    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Plotting ---
    print(f"Generating comparison plot (type: {plot_type}) for degree column: {degree_column}...")
    # 设置一个合理的最大宽度（例如40英寸）以避免尺寸过大
    max_width_inches = 40
    calculated_width = max(8, len(processed_labels) * 1.5)
    figure_width = min(calculated_width, max_width_inches)
    plt.figure(figsize=(figure_width, 6))

    try:
        if plot_type == 'kde':
            sns.kdeplot(data=combined_df, x=degree_column, hue='Sample', fill=True, common_norm=False)
            plt.title(f'Kernel Density Estimate of Grain Degrees ({degree_column}) by Sample')
        elif plot_type == 'box':
            sns.boxplot(data=combined_df, x='Sample', y=degree_column)
            plt.title(f'Box Plot of Grain Degrees ({degree_column}) by Sample')
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'violin':
            sns.violinplot(data=combined_df, x='Sample', y=degree_column)
            plt.title(f'Violin Plot of Grain Degrees ({degree_column}) by Sample')
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'hist':
            # Histogram comparison can be tricky, using FacetGrid or overlapping histograms
            g = sns.FacetGrid(combined_df, col="Sample", col_wrap=min(4, len(processed_labels)), sharey=True, sharex=True)
            g.map(sns.histplot, degree_column, discrete=True) # Use discrete=True for integer degrees
            g.fig.suptitle(f'Histograms of Grain Degrees ({degree_column}) by Sample', y=1.02)
        else:
            print(f"Warning: Unknown plot type '{plot_type}'. Defaulting to KDE plot.")
            sns.kdeplot(data=combined_df, x=degree_column, hue='Sample', fill=True, common_norm=False)
            plt.title(f'Kernel Density Estimate of Grain Degrees ({degree_column}) by Sample')

        plt.xlabel(degree_column)
        if plot_type not in ['hist']: # FacetGrid handles its own labels
             plt.ylabel('Density' if plot_type == 'kde' else degree_column)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.98] if plot_type=='hist' else None) # Adjust layout for FacetGrid title

        # Save output
        print(f"Saving plot to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Done.")

    except Exception as e:
        print(f"Error during plotting: {e}")
        plt.close() # Ensure plot is closed even on error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare grain degree distributions across samples.")
    parser.add_argument('--details_folder', required=True, help='Path to the folder containing _details.csv files.')
    parser.add_argument('--output_path', required=True, help='Path to save the output comparison plot image.')
    parser.add_argument('--file_pattern', default='*_details.csv', help="Glob pattern for finding details files (default: '*_details.csv').")
    parser.add_argument('--degree_column', default='neighbor_count', help="Column name for degree data (default: 'neighbor_count').")
    parser.add_argument('--plot_type', default='kde', choices=['kde', 'box', 'violin', 'hist'], help="Type of comparison plot (default: 'kde').")
    parser.add_argument('--labels', nargs='+', help='(Optional) List of labels for the samples, in the order files are found.')

    args = parser.parse_args()

    # Need to import re here if not imported globally
    import re

    compare_degree_distributions(
        details_folder=args.details_folder,
        output_path=args.output_path,
        file_pattern=args.file_pattern,
        degree_column=args.degree_column,
        plot_type=args.plot_type,
        sample_labels=args.labels
    )
