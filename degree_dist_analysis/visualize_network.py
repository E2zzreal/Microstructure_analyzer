import argparse
import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Import colors module
import networkx as nx
import numpy as np
import pandas as pd
from skimage import io, measure, segmentation

# Add project root to path to allow importing microstructure_analyzer
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from microstructure_analyzer.segmentation import load_masks
    from microstructure_analyzer.feature_extraction import generate_labels
    from microstructure_analyzer.topology import TopologyAnalyzer
    from microstructure_analyzer.feature_utils import build_delaunay_graph # Assuming this is the primary graph method used
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure microstructure_analyzer package is correctly installed or accessible.")
    sys.exit(1)

warnings.filterwarnings("ignore")

def visualize_grain_network(mask_path, details_csv_path, output_path,
                            image_path=None, distance_threshold=50,
                            node_color_feature='neighbor_count', node_size_feature='area',
                            min_node_size=10, max_node_size=500, cmap='viridis'):
    """
    Visualizes the grain network overlaid on the segmentation or original image.

    Args:
        mask_path (str): Path to the .mask file.
        details_csv_path (str): Path to the corresponding _details.csv file.
        output_path (str): Path to save the output visualization image.
        image_path (str, optional): Path to the original image (.tif) for background.
                                    If None, uses the segmentation labels as background. Defaults to None.
        distance_threshold (float): Max distance for edges in the Delaunay graph. Defaults to 50.
        node_color_feature (str): Feature name from details_csv to use for node coloring. Defaults to 'neighbor_count'.
        node_size_feature (str): Feature name from details_csv to use for node sizing. Defaults to 'area'.
        min_node_size (int): Minimum node size for visualization. Defaults to 10.
        max_node_size (int): Maximum node size for visualization. Defaults to 500.
        cmap (str): Colormap name for node coloring. Defaults to 'viridis'.
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

    # Check if required feature columns exist
    if node_color_feature not in details_df.columns:
        print(f"Error: Node color feature '{node_color_feature}' not found in details CSV."); sys.exit(1)
    if node_size_feature not in details_df.columns:
        print(f"Error: Node size feature '{node_size_feature}' not found in details CSV."); sys.exit(1)

    # Generate labels from masks
    print("Generating labels...")
    labels = generate_labels(masks)
    if labels is None: print("Error: Failed to generate labels."); sys.exit(1)

    # --- Prepare Background Image ---
    if image_path and os.path.exists(image_path):
        print(f"Loading background image: {image_path}")
        try:
            background_img = io.imread(image_path)
            # Basic preprocessing similar to segmentation (optional, adjust as needed)
            background_img = background_img[20:]
            if background_img.shape[-1] == 4: background_img = background_img[:, :, :3]
            background_img = background_img[:labels.shape[0], :labels.shape[1], :] # Ensure shape matches labels
            background_img = background_img.astype(np.uint8)
            img_display = background_img
        except Exception as e:
            print(f"Warning: Could not load or process background image '{image_path}': {e}. Using segmentation boundaries.")
            img_display = segmentation.find_boundaries(labels, mode='thick')
    else:
        print("Using segmentation boundaries as background.")
        img_display = segmentation.find_boundaries(labels, mode='thick')

    # --- Build Network Graph ---
    print(f"Building grain network (threshold={distance_threshold})...")
    # Use centroids from details_df, ensure correct order/mapping
    centroids_map = details_df.set_index('label')[['centroid_x', 'centroid_y']].to_dict('index')
    centroids_list = [(centroids_map[lbl]['centroid_x'], centroids_map[lbl]['centroid_y']) for lbl in details_df['label']]
    label_to_index = {lbl: i for i, lbl in enumerate(details_df['label'])} # Map label to index in centroids_list

    if len(centroids_list) < 3:
        print("Warning: Less than 3 grains, cannot build Delaunay graph.")
        G = nx.Graph()
        # Add nodes manually if needed for visualization
        for lbl, pos in centroids_map.items(): G.add_node(lbl, pos=pos)
    else:
        # Build graph using utils function
        G_indexed = build_delaunay_graph(np.array(centroids_list), distance_threshold=distance_threshold)
        # Convert indexed graph back to label-based graph
        G = nx.Graph()
        index_to_label_map = {i: lbl for lbl, i in label_to_index.items()}
        for node_idx, data in G_indexed.nodes(data=True):
            label = index_to_label_map[node_idx]
            G.add_node(label, pos=data['pos'])
        for u_idx, v_idx, data in G_indexed.edges(data=True):
            u_label = index_to_label_map[u_idx]
            v_label = index_to_label_map[v_idx]
            G.add_edge(u_label, v_label, **data)


    # --- Prepare Node Attributes for Visualization ---
    node_positions = {lbl: (data['pos'][0], data['pos'][1]) for lbl, data in G.nodes(data=True)} # Ensure (x, y) format
    node_colors = []
    node_sizes = []
    labels_in_graph = list(G.nodes())

    # Get feature values, handling potential missing labels in graph vs details_df
    color_values = details_df.set_index('label').reindex(labels_in_graph)[node_color_feature].fillna(0)
    size_values = details_df.set_index('label').reindex(labels_in_graph)[node_size_feature].fillna(1)

    # Normalize sizes
    min_val, max_val = size_values.min(), size_values.max()
    if max_val > min_val:
        node_sizes = min_node_size + (size_values - min_val) / (max_val - min_val) * (max_node_size - min_node_size)
    else:
        node_sizes = [ (min_node_size + max_node_size) / 2 ] * len(labels_in_graph) # Assign average size if all values are same

    # --- Determine Colormap and Normalization ---
    is_discrete = False
    if pd.api.types.is_integer_dtype(color_values) or \
       (pd.api.types.is_float_dtype(color_values) and (color_values.fillna(-1) == color_values.fillna(-1).astype(int)).all()):
        # Treat as discrete if integer or float that are all whole numbers
        unique_vals = np.sort(color_values.dropna().unique().astype(int))
        n_colors = len(unique_vals)
        if n_colors > 1:
            is_discrete = True
            print(f"Feature '{node_color_feature}' detected as discrete with {n_colors} unique values.")
            # Choose discrete colormap
            if n_colors <= 10:
                cmap_obj = plt.get_cmap('tab10', n_colors)
            elif n_colors <= 20:
                cmap_obj = plt.get_cmap('tab20', n_colors)
            else:
                print(f"Warning: Too many unique values ({n_colors}) for standard discrete colormaps. Using continuous '{cmap}'.")
                is_discrete = False # Revert to continuous
                cmap_obj = plt.get_cmap(cmap)
                norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())

            if is_discrete:
                # Create boundaries for discrete coloring
                boundaries = np.concatenate(([unique_vals[0] - 0.5], unique_vals[:-1] + 0.5, [unique_vals[-1] + 0.5]))
                norm = mcolors.BoundaryNorm(boundaries, cmap_obj.N)
        else:
             # Only one unique value, use continuous map but with single color
             print(f"Feature '{node_color_feature}' has only one unique value. Using single color from '{cmap}'.")
             cmap_obj = plt.get_cmap(cmap)
             norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())

    else:
        # Continuous feature
        print(f"Feature '{node_color_feature}' treated as continuous.")
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())

    # Map values to colors *before* drawing nodes
    # Create a ScalarMappable to handle the mapping
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    rgba_colors = sm.to_rgba(color_values) # Map the actual feature values to RGBA

    # --- Plotting ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display background
    ax.imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)

    # Draw network
    nx.draw_networkx_edges(G, pos=node_positions, ax=ax, edge_color='gray', alpha=0.6)
    # Pass the pre-calculated RGBA colors directly to node_color, remove cmap and norm
    nodes = nx.draw_networkx_nodes(G, pos=node_positions, ax=ax, nodelist=labels_in_graph,
                                   node_size=node_sizes, node_color=rgba_colors, alpha=0.8) # Use rgba_colors here

    # Add colorbar - Use the same ScalarMappable created earlier
    if nodes:
        # sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm) # Already created above
        sm.set_array([]) # Pass empty array for ScalarMappable

        if is_discrete and n_colors > 1:
            # Discrete colorbar
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, boundaries=boundaries, ticks=unique_vals)
            # Ensure ticks are integers if possible
            try:
                 cbar.set_ticks(unique_vals.astype(int))
                 cbar.set_ticklabels(unique_vals.astype(int))
            except TypeError:
                 cbar.set_ticks(unique_vals) # Fallback if conversion fails
                 cbar.set_ticklabels(unique_vals)

        else:
            # Continuous colorbar (or single value)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)

        cbar.set_label(f'Node Color: {node_color_feature}')
    else:
        print("Warning: No nodes drawn.")


    ax.set_title(f'Grain Network (Nodes colored by {node_color_feature}, sized by {node_size_feature})')
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
    parser = argparse.ArgumentParser(description="Visualize Grain Network")
    parser.add_argument('--mask_path', required=True, help='Path to the input .mask file.')
    parser.add_argument('--details_csv', required=True, help='Path to the corresponding _details.csv file.')
    parser.add_argument('--output_path', required=True, help='Path to save the output visualization image.')
    parser.add_argument('--image_path', help='(Optional) Path to the original image (.tif) for background.')
    parser.add_argument('--distance_threshold', type=float, default=100, help='Max distance for edges in the Delaunay graph (default: 50).')
    parser.add_argument('--color_feature', default='delaunay_degree_adaptive_2r0_5std', help="Feature column from details CSV for node color (default: 'neighbor_count').")
    parser.add_argument('--size_feature', default='area', help="Feature column from details CSV for node size (default: 'area').")
    parser.add_argument('--min_size', type=int, default=10, help='Minimum node size for visualization (default: 10).')
    parser.add_argument('--max_size', type=int, default=500, help='Maximum node size for visualization (default: 500).')
    parser.add_argument('--cmap', default='viridis', help='Colormap for node coloring (default: viridis).')

    args = parser.parse_args()

    visualize_grain_network(
        mask_path=args.mask_path,
        details_csv_path=args.details_csv,
        output_path=args.output_path,
        image_path=args.image_path,
        distance_threshold=args.distance_threshold,
        node_color_feature=args.color_feature,
        node_size_feature=args.size_feature,
        min_node_size=args.min_size,
        max_node_size=args.max_size,
        cmap=args.cmap
    )
