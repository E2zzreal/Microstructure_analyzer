import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")

def plot_degree_histogram(details_csv_path, output_path,
                          degree_column='neighbor_count', bin_width=1,
                          plot_type='hist'):
    """
    Plots the histogram or bar chart of grain degrees for a single sample.

    Args:
        details_csv_path (str): Path to the _details.csv file containing per-grain data.
        output_path (str): Path to save the output plot image.
        degree_column (str): Name of the column containing the degree information
                             (e.g., 'neighbor_count', 'delaunay_degree_fixed_50').
                             Defaults to 'neighbor_count'.
        bin_width (int): Width of bins for the histogram. Defaults to 1.
        plot_type (str): Type of plot ('hist' for histogram, 'bar' for count bar chart).
                         Defaults to 'hist'.
    """
    print(f"Loading grain details: {details_csv_path}")
    if not os.path.exists(details_csv_path):
        print(f"Error: Details CSV file not found: {details_csv_path}"); sys.exit(1)
    details_df = pd.read_csv(details_csv_path)
    if details_df.empty: print(f"Error: Details CSV is empty: {details_csv_path}"); sys.exit(1)

    # Check if required degree column exists
    if degree_column not in details_df.columns:
        print(f"Error: Degree column '{degree_column}' not found in details CSV."); sys.exit(1)

    degree_data = details_df[degree_column].dropna()
    if degree_data.empty:
        print(f"Warning: No valid data found for degree column '{degree_column}'. Cannot generate plot.")
        return

    # --- Plotting ---
    print(f"Generating {plot_type} plot for degree column: {degree_column}...")
    plt.figure(figsize=(8, 6))

    try:
        if plot_type == 'hist':
            # Calculate bins based on data range and bin_width
            min_degree = int(np.floor(degree_data.min()))
            max_degree = int(np.ceil(degree_data.max()))
            bins = np.arange(min_degree, max_degree + bin_width, bin_width)
            sns.histplot(degree_data, bins=bins, kde=False, stat='count') # Use stat='count' for frequency
            plt.xlabel(f'{degree_column} (Bin Width = {bin_width})')
            plt.ylabel('Grain Count')
            plt.title(f'Histogram of Grain Degrees ({degree_column})')

        elif plot_type == 'bar':
            # Count occurrences of each degree value
            degree_counts = degree_data.astype(int).value_counts().sort_index()
            sns.barplot(x=degree_counts.index, y=degree_counts.values, color='skyblue')
            plt.xlabel(f'{degree_column}')
            plt.ylabel('Grain Count')
            plt.title(f'Distribution of Grain Degrees ({degree_column})')

        else:
            print(f"Warning: Unknown plot type '{plot_type}'. Defaulting to histogram.")
            min_degree = int(np.floor(degree_data.min()))
            max_degree = int(np.ceil(degree_data.max()))
            bins = np.arange(min_degree, max_degree + bin_width, bin_width)
            sns.histplot(degree_data, bins=bins, kde=False, stat='count')
            plt.xlabel(f'{degree_column} (Bin Width = {bin_width})')
            plt.ylabel('Grain Count')
            plt.title(f'Histogram of Grain Degrees ({degree_column})')


        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

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
    parser = argparse.ArgumentParser(description="Plot histogram of grain degrees for a single sample.")
    parser.add_argument('--details_csv', required=True, help='Path to the input _details.csv file.')
    parser.add_argument('--output_path', required=True, help='Path to save the output plot image.')
    parser.add_argument('--degree_column', default='neighbor_count', help="Column name for degree data (default: 'neighbor_count').")
    parser.add_argument('--bin_width', type=int, default=1, help="Width of bins for histogram plot (default: 1).")
    parser.add_argument('--plot_type', default='hist', choices=['hist', 'bar'], help="Type of plot ('hist' or 'bar') (default: 'hist').")


    args = parser.parse_args()

    plot_degree_histogram(
        details_csv_path=args.details_csv,
        output_path=args.output_path,
        degree_column=args.degree_column,
        bin_width=args.bin_width,
        plot_type=args.plot_type
    )
