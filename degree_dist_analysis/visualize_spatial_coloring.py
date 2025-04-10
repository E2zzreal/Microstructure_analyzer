import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Import colors module
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

def visualize_spatial_coloring(mask_path, details_csv_path, output_path,
                               color_feature='neighbor_count', cmap='viridis',
                               background_color=(1, 1, 1)):
    """
    Visualizes the segmented grains colored by a specific feature value.

    Args:
        mask_path (str): Path to the .mask file.
        details_csv_path (str): Path to the corresponding _details.csv file.
        output_path (str): Path to save the output visualization image.
        color_feature (str): Feature name from details_csv to use for coloring grains.
                             Defaults to 'neighbor_count'.
        cmap (str): Colormap name for coloring. Defaults to 'viridis'.
        background_color (tuple): RGB tuple (0-1 range) for the background color.
                                  Defaults to white (1, 1, 1).
    """
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

    # Check if required feature column exists
    if color_feature not in details_df.columns:
        print(f"Error: Color feature '{color_feature}' not found in details CSV."); sys.exit(1)

    # Generate labels from masks
    print("Generating labels...")
    labels = generate_labels(masks)
    if labels is None: print("Error: Failed to generate labels."); sys.exit(1)

    # --- Create Colored Label Image ---
    print(f"Coloring grains by feature: {color_feature}...")
    # Create mapping from label to feature value
    label_to_value = details_df.set_index('label')[color_feature].to_dict()

    # Get unique labels present in the image (excluding background 0)
    unique_labels = np.unique(labels[labels > 0])

    # Create an array to store the feature value for each pixel label
    # Initialize with a value that maps to the low end of the colormap or NaN
    min_feature_val = details_df[color_feature].min()
    value_map_array = np.full(labels.shape, min_feature_val, dtype=float) # Use float for potential NaNs

    for label_val in unique_labels:
        feature_val = label_to_value.get(label_val, np.nan) # Get value, default to NaN if label missing in details
        if pd.isna(feature_val):
             print(f"Warning: No feature value found for label {label_val}. It will use the minimum color.")
             feature_val = min_feature_val # Assign min value if missing
        value_map_array[labels == label_val] = feature_val

    # Handle background (label 0) - assign NaN or a value outside the feature range
    value_map_array[labels == 0] = np.nan # Use NaN for background

    # --- Determine Colormap and Normalization ---
    valid_values = value_map_array[~np.isnan(value_map_array)]
    is_discrete = False
    norm = None
    cmap_obj = None
    unique_vals = []
    n_colors = 0
    boundaries = None

    if len(valid_values) > 0:
        # Check if values are integer-like
        if np.all(valid_values == valid_values.astype(int)):
            unique_vals = np.sort(np.unique(valid_values.astype(int)))
            n_colors = len(unique_vals)
            if n_colors > 1:
                is_discrete = True
                print(f"Feature '{color_feature}' detected as discrete with {n_colors} unique values.")
                # Choose discrete colormap
                if n_colors <= 10:
                    cmap_obj = plt.get_cmap('tab10', n_colors)
                elif n_colors <= 20:
                    cmap_obj = plt.get_cmap('tab20', n_colors)
                else:
                    print(f"Warning: Too many unique values ({n_colors}) for standard discrete colormaps. Using continuous '{cmap}'.")
                    is_discrete = False # Revert to continuous
            else:
                 print(f"Feature '{color_feature}' has only one unique value. Using single color from '{cmap}'.")
                 is_discrete = False # Treat as continuous for simplicity

        if is_discrete:
            # Create boundaries for discrete coloring
            boundaries = np.concatenate(([unique_vals[0] - 0.5], unique_vals[:-1] + 0.5, [unique_vals[-1] + 0.5]))
            norm = mcolors.BoundaryNorm(boundaries, cmap_obj.N)
        else:
            # Continuous feature or fallback
            if not is_discrete: # Only print if not already warned about too many colors
                 print(f"Feature '{color_feature}' treated as continuous.")
            cmap_obj = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=np.min(valid_values), vmax=np.max(valid_values))

    else: # Handle case where all values might be NaN (e.g., no valid labels)
        print("Warning: No valid feature values found for coloring.")
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=1) # Default norm

    # --- Plotting ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the value map using imshow with the chosen colormap and norm
    # Set background pixels (NaNs) to the specified background color
    im = ax.imshow(value_map_array, cmap=cmap_obj, norm=norm)
    im.cmap.set_bad(color=background_color) # Set color for NaN values

    # Add colorbar
    if len(valid_values) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([]) # Pass empty array

        if is_discrete and n_colors > 1:
            # Discrete colorbar
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7, boundaries=boundaries, ticks=unique_vals)
            # Ensure ticks are integers
            try:
                 cbar.set_ticks(unique_vals.astype(int))
                 cbar.set_ticklabels(unique_vals.astype(int))
            except TypeError:
                 cbar.set_ticks(unique_vals) # Fallback
                 cbar.set_ticklabels(unique_vals)
        else:
            # Continuous colorbar (or single value)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7)

        cbar.set_label(f'Feature Value: {color_feature}')
    else:
        print("Skipping colorbar as no valid data was found.")

    ax.set_title(f'Grains Colored by {color_feature}')
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
    parser = argparse.ArgumentParser(description="Visualize segmented grains colored by a feature.")
    parser.add_argument('--mask_path', required=True, help='Path to the input .mask file.')
    parser.add_argument('--details_csv', required=True, help='Path to the corresponding _details.csv file.')
    parser.add_argument('--output_path', required=True, help='Path to save the output visualization image.')
    parser.add_argument('--color_feature', default='neighbor_count', help="Feature column from details CSV for coloring (default: 'neighbor_count').")
    parser.add_argument('--cmap', default='viridis', help='Colormap for coloring (default: viridis).')

    args = parser.parse_args()

    visualize_spatial_coloring(
        mask_path=args.mask_path,
        details_csv_path=args.details_csv,
        output_path=args.output_path,
        color_feature=args.color_feature,
        cmap=args.cmap
    )
