import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io, measure, segmentation, color

# Add project root to path to allow importing microstructure_analyzer
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.segmentation import load_masks
    from microstructure_analyzer.feature_extraction import generate_labels
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure microstructure_analyzer package is correctly installed or accessible.")
    sys.exit(1)

warnings.filterwarnings("ignore")

def visualize_degree_outliers(mask_path, details_csv_path, output_path,
                              degree_column='neighbor_count',
                              min_degree=None, max_degree=None,
                              outlier_color=(1, 0, 0), # Red
                              normal_color=(0.7, 0.7, 0.7), # Gray
                              background_color=(1, 1, 1)): # White
    """
    Visualizes segmented grains, highlighting those with outlier degrees.

    Args:
        mask_path (str): Path to the .mask file.
        details_csv_path (str): Path to the corresponding _details.csv file.
        output_path (str): Path to save the output visualization image.
        degree_column (str): Name of the column containing the degree information.
                             Defaults to 'neighbor_count'.
        min_degree (int, optional): Grains with degree less than this value are outliers.
                                    Defaults to None (no lower bound check).
        max_degree (int, optional): Grains with degree greater than this value are outliers.
                                    Defaults to None (no upper bound check).
        outlier_color (tuple): RGB tuple (0-1 range) for highlighting outlier grains.
                               Defaults to red (1, 0, 0).
        normal_color (tuple): RGB tuple (0-1 range) for non-outlier grains.
                              Defaults to gray (0.7, 0.7, 0.7).
        background_color (tuple): RGB tuple (0-1 range) for the background.
                                  Defaults to white (1, 1, 1).
    """
    if min_degree is None and max_degree is None:
        print("Error: At least one of --min_degree or --max_degree must be provided.")
        sys.exit(1)

    print(f"Loading mask: {mask_path}")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found: {mask_path}"); sys.exit(1)
    masks = load_masks(mask_path)
    if not masks: print(f"Error: No masks loaded from {mask_path}"); sys.exit(1)

    print(f"Loading grain details: {details_csv_path}")
    if not os.path.exists(details_csv_path):
        print(f"Error: Details CSV file not found: {details_csv_path}"); sys.exit(1)
    details_df = pd.read_csv(details_csv_path)
    if details_df.empty: print(f"Error: Details CSV is empty: {details_csv_path}"); sys.exit(1)

    # Check if required degree column exists
    if degree_column not in details_df.columns:
        print(f"Error: Degree column '{degree_column}' not found in details CSV."); sys.exit(1)

    # Generate labels from masks
    print("Generating labels...")
    labels = generate_labels(masks)
    if labels is None: print("Error: Failed to generate labels."); sys.exit(1)

    # --- Create Colored Image based on Outliers ---
    print(f"Highlighting outlier grains based on column: {degree_column}")
    if min_degree is not None: print(f"  Lower bound (exclusive): {min_degree}")
    if max_degree is not None: print(f"  Upper bound (exclusive): {max_degree}")

    # Create mapping from label to degree value
    label_to_degree = details_df.set_index('label')[degree_column].to_dict()

    # Create an RGB image initialized with background color
    colored_image = np.full(labels.shape + (3,), background_color, dtype=float)

    # Iterate through unique grain labels
    unique_labels = np.unique(labels[labels > 0])
    outlier_count = 0
    for label_val in unique_labels:
        degree = label_to_degree.get(label_val, np.nan)
        is_outlier = False
        if not pd.isna(degree):
            if min_degree is not None and degree < min_degree:
                is_outlier = True
            if max_degree is not None and degree > max_degree:
                is_outlier = True
        else:
            print(f"Warning: No degree value found for label {label_val}.")
            # Treat missing degree as non-outlier or outlier? Defaulting to non-outlier.

        # Assign color based on whether it's an outlier
        grain_mask = (labels == label_val)
        if is_outlier:
            colored_image[grain_mask] = outlier_color
            outlier_count += 1
        else:
            colored_image[grain_mask] = normal_color

    print(f"Found {outlier_count} outlier grains.")

    # --- Plotting ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(colored_image) # Display the RGB image

    # Create custom legend handles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=outlier_color, edgecolor='black', label='Outlier Degree'),
                       Patch(facecolor=normal_color, edgecolor='black', label='Normal Degree')]
    ax.legend(handles=legend_elements, loc='lower right')

    title = f'Grain Degree Outliers ({degree_column}'
    if min_degree is not None: title += f' < {min_degree}'
    if min_degree is not None and max_degree is not None: title += ' or'
    if max_degree is not None: title += f' > {max_degree}'
    title += ')'
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Save output
    print(f"Saving visualization to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize grains with outlier degrees.")
    parser.add_argument('--mask_path', required=True, help='Path to the input .mask file.')
    parser.add_argument('--details_csv', required=True, help='Path to the corresponding _details.csv file.')
    parser.add_argument('--output_path', required=True, help='Path to save the output visualization image.')
    parser.add_argument('--degree_column', default='neighbor_count', help="Column name for degree data (default: 'neighbor_count').")
    parser.add_argument('--min_degree', type=int, help='Highlight grains with degree LESS THAN this value.')
    parser.add_argument('--max_degree', type=int, help='Highlight grains with degree GREATER THAN this value.')

    args = parser.parse_args()

    if args.min_degree is None and args.max_degree is None:
         parser.error("At least one of --min_degree or --max_degree must be specified.")


    visualize_degree_outliers(
        mask_path=args.mask_path,
        details_csv_path=args.details_csv,
        output_path=args.output_path,
        degree_column=args.degree_column,
        min_degree=args.min_degree,
        max_degree=args.max_degree
    )
