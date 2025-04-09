import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

def plot_degree_vs_property(details_csv_path, output_path,
                            property_feature='area', degree_feature='neighbor_count',
                            plot_type='scatter'):
    """
    Plots the relationship between grain degree and another grain property.

    Args:
        details_csv_path (str): Path to the _details.csv file containing per-grain data.
        output_path (str): Path to save the output plot image.
        property_feature (str): Feature name for the x-axis (e.g., 'area', 'aspect_ratio'). Defaults to 'area'.
        degree_feature (str): Feature name for the y-axis (usually 'neighbor_count'). Defaults to 'neighbor_count'.
        plot_type (str): Type of plot ('scatter', 'hexbin', 'kde'). Defaults to 'scatter'.
    """
    print(f"Loading grain details: {details_csv_path}")
    if not os.path.exists(details_csv_path):
        print(f"Error: Details CSV file not found: {details_csv_path}"); sys.exit(1)
    details_df = pd.read_csv(details_csv_path)
    if details_df.empty: print(f"Error: Details CSV is empty: {details_csv_path}"); sys.exit(1)

    # Check if required feature columns exist
    if property_feature not in details_df.columns:
        print(f"Error: Property feature '{property_feature}' not found in details CSV."); sys.exit(1)
    if degree_feature not in details_df.columns:
        print(f"Error: Degree feature '{degree_feature}' not found in details CSV."); sys.exit(1)

    # --- Plotting ---
    print(f"Generating {plot_type} plot: {degree_feature} vs {property_feature}...")
    plt.figure(figsize=(8, 6))

    x_data = details_df[property_feature]
    y_data = details_df[degree_feature]

    try:
        if plot_type == 'scatter':
            sns.scatterplot(x=x_data, y=y_data, alpha=0.5)
        elif plot_type == 'hexbin':
            # Hexbin requires numerical data, ensure types are correct
            if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
                 plt.hexbin(x_data, y_data, gridsize=30, cmap='viridis', mincnt=1) # mincnt=1 shows bins with >=1 point
                 plt.colorbar(label='Count in bin')
            else:
                 print(f"Warning: Hexbin plot requires numeric data. Columns '{property_feature}' or '{degree_feature}' might not be numeric. Falling back to scatter.")
                 sns.scatterplot(x=x_data, y=y_data, alpha=0.5)
        elif plot_type == 'kde':
            sns.kdeplot(x=x_data, y=y_data, cmap="viridis", fill=True)
        else:
            print(f"Warning: Unknown plot type '{plot_type}'. Defaulting to scatter plot.")
            sns.scatterplot(x=x_data, y=y_data, alpha=0.5)

        plt.xlabel(property_feature)
        plt.ylabel(degree_feature)
        plt.title(f'Grain {degree_feature} vs. {property_feature}')
        plt.grid(True, alpha=0.3)
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
    parser = argparse.ArgumentParser(description="Plot relationship between grain degree and another property.")
    parser.add_argument('--details_csv', required=True, help='Path to the input _details.csv file.')
    parser.add_argument('--output_path', required=True, help='Path to save the output plot image.')
    parser.add_argument('--property', default='area', help="Feature column for x-axis (default: 'area').")
    parser.add_argument('--degree', default='delaunay_degree_adaptive_2r0_5std', help="Feature column for y-axis (default: 'delaunay_degree_adaptive_2r0_5std').")
    parser.add_argument('--plot_type', default='scatter', choices=['scatter', 'hexbin', 'kde'], help="Type of plot (default: 'scatter').")

    args = parser.parse_args()

    plot_degree_vs_property(
        details_csv_path=args.details_csv,
        output_path=args.output_path,
        property_feature=args.property,
        degree_feature=args.degree,
        plot_type=args.plot_type
    )
