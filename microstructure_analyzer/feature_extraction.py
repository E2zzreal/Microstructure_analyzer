import os
import re
import numpy as np
import pandas as pd
from skimage import measure, morphology, segmentation, feature
from scipy import ndimage
from collections import defaultdict
import warnings

# Internal imports (relative paths within the package)
from .segmentation import load_masks
from .topology import TopologyAnalyzer
from .feature_utils import (
    self_similarity_dimension, count_skeleton_branches, calculate_orientation,
    calculate_grain_boundary_properties, calculate_nearest_neighbor_distances,
    calculate_orientation_statistics, calculate_fractal_dimension_on_boundaries,
    calculate_persistence_homology, calculate_pairwise_distances,
    calculate_grain_network_features, calculate_perimeter_area_ratio,
    calculate_compactness, calculate_rectangularity, calculate_convex_defect,
    calculate_tortuosity, calculate_curvature_std, calculate_fourier_energy_ratio,
    calculate_curvature # Needed for shape stats
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Ignore warnings from skimage/scipy about future changes etc.


def generate_labels(masks):
    """
    Generates a labeled image from a list of non-overlapping masks.

    Args:
        masks (list): List of mask dictionaries, where each dictionary has a
                      'segmentation' key with a boolean numpy array. Masks
                      are assumed to be deduplicated (non-overlapping).

    Returns:
        np.ndarray: A 2D integer array where each unique mask region is assigned
                    a unique positive integer label. Background remains 0.
                    Returns None if masks list is empty or invalid.
    """
    if not masks or 'segmentation' not in masks[0]:
        print("Warning: Empty or invalid mask list provided to generate_labels.")
        return None

    # Get shape from the first mask
    image_shape = masks[0]['segmentation'].shape
    labels = np.zeros(image_shape, dtype=np.int32)

    # Assign unique label for each mask
    for idx, mask_dict in enumerate(masks, start=1):
        mask_array = mask_dict['segmentation']
        if mask_array.shape != image_shape:
             print(f"Warning: Mask shape mismatch {mask_array.shape} vs {image_shape} in generate_labels. Skipping mask.")
             continue
        # Assign label only where the mask is True and no label is assigned yet
        # (Shouldn't be necessary if masks are truly deduplicated, but safe)
        labels[mask_array & (labels == 0)] = idx

    return labels


def generate_grain_boundaries(labels, background_area_threshold=100, thin_wall_threshold=3, junction_dilation_radius=3, thin_wall_min_size=10, junction_min_size=50, junction_expansion_radius=5):
    """
    Identifies and labels different types of grain boundaries (thin wall vs. junction).

    Args:
        labels (np.ndarray): Labeled image of grains (output of generate_labels).
        background_area_threshold (int): Minimum area for a background region to be considered.
        thin_wall_threshold (int): Distance threshold (pixels) to classify thin walls.
        junction_dilation_radius (int): Radius for dilating junction points.
        thin_wall_min_size (int): Minimum size (pixels) for thin wall regions after filtering.
        junction_min_size (int): Minimum size (pixels) for junction regions after filtering.
        junction_expansion_radius (int): Radius for expanding junction regions before final filtering.

    Returns:
        tuple: (filtered_background_labels, thin_wall_labels, junction_labels)
               - filtered_background_labels: Labeled image of valid background regions.
               - thin_wall_labels: Labeled image of thin wall boundary regions.
               - junction_labels: Labeled image of junction boundary regions.
               Returns (None, None, None) if input labels are invalid.
    """
    if labels is None or labels.ndim != 2:
        print("Warning: Invalid labels input to generate_grain_boundaries.")
        return None, None, None

    # 1. Identify valid background regions (area >= threshold)
    background_mask = (labels == 0)
    bg_labels_raw = measure.label(background_mask, connectivity=2) # 8-connectivity
    bg_regions = measure.regionprops(bg_labels_raw)
    valid_bg_labels = [region.label for region in bg_regions if region.area >= background_area_threshold]
    filtered_background_mask = np.isin(bg_labels_raw, valid_bg_labels)
    filtered_background_labels = measure.label(filtered_background_mask, connectivity=2) # Re-label filtered background

    if not np.any(filtered_background_mask):
         print("Warning: No significant background regions found.")
         # Return empty label maps of the correct shape
         empty_labels = np.zeros_like(labels, dtype=np.int32)
         return filtered_background_labels, empty_labels.copy(), empty_labels.copy()


    # 2. Skeletonize the valid background (potential boundary network)
    skeleton = morphology.skeletonize(filtered_background_mask)

    # 3. Detect skeleton junction points (pixels with >= 3 neighbors in skeleton)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    # Junction points are skeleton pixels with neighbor count >= 4 (including self)
    junction_points = skeleton & (neighbor_count >= 4)

    # 4. Dilate junction points to define junction regions
    junction_dilated = morphology.binary_dilation(junction_points, morphology.disk(junction_dilation_radius))

    # 5. Use distance transform on background mask to estimate width
    distance_map = ndimage.distance_transform_edt(filtered_background_mask)

    # 6. Classify based on distance and proximity to junctions
    # Thin wall candidates: narrow regions (low distance value)
    thin_wall_candidate = distance_map <= thin_wall_threshold
    # Junction candidates: wider regions OR regions near dilated junctions
    junction_candidate = (~thin_wall_candidate) | junction_dilated

    # Apply classification only within the valid background mask
    thin_wall_mask = filtered_background_mask & thin_wall_candidate & ~junction_dilated # Narrow and not near junction
    junction_mask = filtered_background_mask & junction_candidate # Wide OR near junction

    # 7. Post-processing
    # Remove small objects
    thin_wall_mask = morphology.remove_small_objects(thin_wall_mask, min_size=thin_wall_min_size)
    junction_mask = morphology.remove_small_objects(junction_mask, min_size=junction_min_size)

    # Expand junctions slightly and remove overlap from thin walls
    expanded_junction = morphology.binary_dilation(junction_mask, morphology.disk(junction_expansion_radius))
    thin_wall_mask = thin_wall_mask & ~expanded_junction

    # Fill holes within thin wall regions (optional, based on notebook)
    thin_wall_filled = ndimage.binary_fill_holes(thin_wall_mask)
    thin_wall_mask = thin_wall_mask | thin_wall_filled # Combine original and filled

    # Final small object removal
    thin_wall_mask = morphology.remove_small_objects(thin_wall_mask, min_size=thin_wall_min_size)
    junction_mask = morphology.remove_small_objects(junction_mask, min_size=junction_min_size) # Re-apply to junction mask too

    # Final labeling
    thin_wall_labels = measure.label(thin_wall_mask, connectivity=2)
    junction_labels = measure.label(junction_mask, connectivity=2)

    return filtered_background_labels, thin_wall_labels, junction_labels


# ======================== Single Region Feature Calculation ========================

def geometric_features(region):
    """
    Calculates geometric features for a single region.

    Args:
        region (skimage.measure._regionprops.RegionProperties): The region object.

    Returns:
        dict: Dictionary of geometric features.
    """
    features = {}
    area = region.area
    perimeter = region.perimeter

    # Basic Area/Size
    features['area'] = area
    features['equivalent_diameter'] = region.equivalent_diameter
    try:
        # Feret diameter calculation can sometimes fail for complex shapes
        features['feret_diameter_max'] = region.feret_diameter_max
    except Exception:
        features['feret_diameter_max'] = 0.0 # Default value on failure

    # Shape Features
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    features['aspect_ratio'] = major_axis / (minor_axis + 1e-6) # Avoid division by zero
    features['compactness'] = (4 * np.pi * area) / (perimeter**2 + 1e-6) if perimeter > 0 else 0
    features['elongation'] = 1 - (minor_axis / (major_axis + 1e-6)) if major_axis > 0 else 0
    features['rectangularity'] = area / region.area_bbox if region.area_bbox > 0 else 0
    features['convexity'] = area / region.area_convex if region.area_convex > 0 else 0

    # Boundary Features
    features['perimeter'] = perimeter
    features['fractal_dimension_boxcount'] = self_similarity_dimension(region) # Use helper

    # Skeleton Features
    features['skeleton_branches'] = 0
    features['skeleton_length_ratio'] = 0.0
    if area > 4: # Need minimal area for skeletonization
        try:
            # Ensure region.image is 2D boolean
            region_image_bool = region.image.astype(bool)
            if region_image_bool.ndim == 2:
                skeleton = morphology.skeletonize(region_image_bool)
                if np.any(skeleton):
                    features['skeleton_branches'] = count_skeleton_branches(skeleton) # Use helper
                    skeleton_length = np.sum(skeleton)
                    features['skeleton_length_ratio'] = skeleton_length / area if area > 0 else 0
        except Exception as e:
            # print(f"Warning: Skeleton analysis failed for region {region.label}: {e}")
            pass # Keep defaults

    # Orientation Feature
    features['orientation'] = calculate_orientation(region) # Use helper

    # Moments
    # Hu moments are invariant to translation, scale, rotation
    features['hu_moments'] = region.moments_hu # Returns array of 7 Hu moments

    return features


def advanced_features(region, neighbors):
    """
    Calculates advanced or composite features for a single region,
    considering its neighboring regions.

    Args:
        region (skimage.measure._regionprops.RegionProperties): The region object.
        neighbors (list): List of region property objects for the neighbors.

    Returns:
        dict: Dictionary of advanced features.
    """
    features = {}
    area = region.area
    perimeter = region.perimeter

    # Composite Shape Features
    features['slenderness'] = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
    # Shape complexity is same as 1/compactness
    # features['shape_complexity'] = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0

    # Local Environment Features
    neighbor_areas = [n.area for n in neighbors if n.area > 0]
    if len(neighbor_areas) > 0:
        mean_neighbor_area = np.mean(neighbor_areas)
        features['neighbor_area_mean'] = mean_neighbor_area
        features['neighbor_area_std'] = np.std(neighbor_areas)
        # Relative size variation
        features['neighbor_size_variation'] = np.std(neighbor_areas) / (mean_neighbor_area + 1e-6)
    else:
        features['neighbor_area_mean'] = 0.0
        features['neighbor_area_std'] = 0.0
        features['neighbor_size_variation'] = 0.0

    return features


# ======================== Per-Region Feature Calculation ========================

def calculate_per_region_features(labels, region_type_prefix):
    """
    Calculates detailed features for each individual region in a labeled image.

    Args:
        labels (np.ndarray): Labeled image (e.g., grains).
        region_type_prefix (str): Prefix related to the region type (e.g., "grain_").

    Returns:
        list: A list of dictionaries. Each dictionary contains features for one region,
              including 'label', 'centroid_y', 'centroid_x', and calculated features.
              Returns empty list if no regions.
    """
    # Removed coordinates='rc' argument for compatibility with older skimage versions or environment issues
    regions = measure.regionprops(labels)
    if not regions:
        return []

    # Analyze topology once for all regions in this label map
    try:
        topo_analyzer = TopologyAnalyzer(labels)
    except Exception as e:
        print(f"Warning: TopologyAnalyzer failed for {region_type_prefix}: {e}. Topological features will be missing.")
        topo_analyzer = None # Proceed without topology if it fails

    all_region_details = []
    for region in regions:
        label = region.label
        region_details = {'label': label} # Start with the label

        # Add centroid
        centroid_y, centroid_x = region.centroid
        region_details['centroid_y'] = centroid_y
        region_details['centroid_x'] = centroid_x

        # Get neighbors for this region
        if topo_analyzer:
            neighbor_indices = np.where(topo_analyzer.adjacency_matrix[label] > 0)[0]
            neighbors = [r for r in regions if r.label in neighbor_indices and r.label != label]
        else:
            neighbors = [] # Empty list if topology failed

        # Calculate features for the single region
        try:
            geo_feat = geometric_features(region)
            adv_feat = advanced_features(region, neighbors)
            topo_feat = topo_analyzer.get_grain_topology_features(label) if topo_analyzer else {}

            # Combine features, handling potential overlaps if necessary
            combined = {**geo_feat, **adv_feat, **topo_feat}

            # Add combined features to the region's details dictionary
            # Handle array-like features (e.g., Hu moments) by expanding them
            for key, value in combined.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    for idx, elem in enumerate(value):
                        try:
                            region_details[f"{key}_{idx}"] = float(elem)
                        except (ValueError, TypeError):
                            region_details[f"{key}_{idx}"] = 0.0 # Default value
                else:
                    try:
                        region_details[key] = float(value) # Converts bools to 0.0/1.0
                    except (ValueError, TypeError):
                        region_details[key] = 0.0 # Default value

        except Exception as e:
            print(f"Warning: Feature calculation failed for region {label} ({region_type_prefix}): {e}")
            # Add placeholder values or skip features on error? Add placeholders for now.
            # This part could be more robust by listing expected features and filling NaNs.

        all_region_details.append(region_details)

    return all_region_details


# ======================== Feature Aggregation and Statistics ========================

def calculate_region_features_stats(labels, region_type_prefix):
    """
    Calculates statistics (mean, std) of geometric, topological, and advanced
    features for all regions defined in a labeled image.

    Args:
        labels (np.ndarray): Labeled image (e.g., thin walls, junctions, or grains).
        region_type_prefix (str): Prefix to add to feature names (e.g., "thinw_", "junc_", "grain_").

    Returns:
        dict: Dictionary where keys are feature names (e.g., "thinw_area_mean")
              and values are the calculated statistics. Returns empty dict if no regions.
    """
    # Get per-region features first
    all_region_details = calculate_per_region_features(labels, region_type_prefix)
    if not all_region_details:
        return {}

    # Aggregate features into lists for statistical calculation
    # Exclude 'label', 'centroid_y', 'centroid_x' from stats calculation
    feature_values = defaultdict(list)
    excluded_keys = {'label', 'centroid_y', 'centroid_x'}
    for region_detail_dict in all_region_details:
        for key, value in region_detail_dict.items():
            if key not in excluded_keys:
                 # Assuming values are already float/numeric from calculate_per_region_features
                 feature_values[key].append(value)

    # Calculate statistics (mean, std) for each feature
    final_stats = {}
    for key, values_list in feature_values.items():
        arr = np.array(values_list)
        # Clean array (handle potential NaNs/Infs introduced by calculations)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mean_val = np.mean(arr) if len(arr) > 0 else 0.0
        std_val = np.std(arr) if len(arr) > 0 else 0.0

        final_stats[f"{region_type_prefix}{key}_mean"] = mean_val
        final_stats[f"{region_type_prefix}{key}_std"] = std_val

    return final_stats


def calculate_global_features(labels):
    """
    Calculates global features describing the overall microstructure based on grain labels.

    Args:
        labels (np.ndarray): Labeled image of grains.

    Returns:
        dict: Dictionary of global features. Returns empty dict if no regions.
    """
    regions = measure.regionprops(labels)
    if not regions:
        return {}

    features = {}
    areas = [r.area for r in regions]

    # 1. Basic Grain Statistics
    features['num_grains'] = len(regions)
    features['area_mean'] = np.mean(areas) if areas else 0.0
    features['area_std'] = np.std(areas) if areas else 0.0

    # 2. Boundary and Curvature (using helper)
    total_boundary_length, mean_curvature = calculate_grain_boundary_properties(labels)
    features['total_boundary_length'] = total_boundary_length
    features['mean_boundary_curvature'] = mean_curvature

    # 3. Spatial Distribution (using helpers)
    nn_dist_mean, nn_dist_std = calculate_nearest_neighbor_distances(labels)
    features['nn_dist_mean'] = nn_dist_mean
    features['nn_dist_std'] = nn_dist_std

    pairwise_dist_mean, pairwise_dist_std = calculate_pairwise_distances(labels)
    features['pairwise_dist_mean'] = pairwise_dist_mean
    features['pairwise_dist_std'] = pairwise_dist_std

    # 4. Orientation Statistics (using helper)
    orientation_mean, orientation_std = calculate_orientation_statistics(labels)
    features['orientation_mean'] = orientation_mean
    features['orientation_std'] = orientation_std

    # 5. Fractal Dimension of Boundaries (using helper)
    features['boundary_fractal_dim'] = calculate_fractal_dimension_on_boundaries(labels)

    # 6. Persistence Homology (using helper)
    # Consider making max_dist_thresh a parameter or adaptive
    persistence_0_mean, persistence_1_mean = calculate_persistence_homology(labels, max_dist_thresh=100)
    features['persistence_H0_mean'] = persistence_0_mean
    features['persistence_H1_mean'] = persistence_1_mean

    # 7. Grain Network Features (using helper for different thresholds)
    for threshold in [20, 50, 100, 150]:
        network_features = calculate_grain_network_features(labels, distance_threshold=threshold)
        # Add prefix to avoid name collisions if needed, or just update
        features.update(network_features) # Keys already include threshold

    # 8. Statistics of Per-Grain Shape Features (using helpers)
    shape_feature_lists = defaultdict(list)
    for region in regions:
        # Calculate features requiring boundary coordinates
        boundary_coords = None
        contours = measure.find_contours(region.image, 0.5)
        if contours:
            # Use longest contour, convert to global coordinates
            contour_local = contours[0]
            boundary_coords = contour_local + region.bbox[:2]

        shape_feature_lists['perimeter_area_ratio'].append(calculate_perimeter_area_ratio(region))
        shape_feature_lists['compactness'].append(calculate_compactness(region))
        shape_feature_lists['rectangularity'].append(calculate_rectangularity(region))
        shape_feature_lists['convex_defect'].append(calculate_convex_defect(region))
        if boundary_coords is not None:
            shape_feature_lists['tortuosity'].append(calculate_tortuosity(boundary_coords))
            shape_feature_lists['curvature_std'].append(calculate_curvature_std(boundary_coords))
            shape_feature_lists['fourier_energy'].append(calculate_fourier_energy_ratio(boundary_coords))
        else: # Append default if boundary couldn't be found
            shape_feature_lists['tortuosity'].append(0.0)
            shape_feature_lists['curvature_std'].append(0.0)
            shape_feature_lists['fourier_energy'].append(0.0)

    # Calculate mean and std for each shape feature list
    for key, values_list in shape_feature_lists.items():
        arr = np.array(values_list)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) # Clean data
        features[f'shape_{key}_mean'] = np.mean(arr) if len(arr) > 0 else 0.0
        features[f'shape_{key}_std'] = np.std(arr) if len(arr) > 0 else 0.0

    # Add prefix to all global features
    prefixed_features = {f"global_{k}": v for k, v in features.items()}

    return prefixed_features


# ======================== Main Calculation Workflow ========================

def calculate_features_for_maskfile(mask_file_path, save_details=False, details_folder=None):
    """
    Calculates all aggregate features for a single .mask file and optionally saves
    per-grain details to a separate CSV file.

    Args:
        mask_file_path (str): Path to the .mask file.
        save_details (bool): If True, save per-grain details to a CSV file. Defaults to False.
        details_folder (str, optional): Directory to save the per-grain details CSV.
                                        If None, saves in the same directory as the mask file.
                                        Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing all calculated aggregate features for this file,
                      with the index set based on the filename pattern.
                      Returns an empty DataFrame if processing fails.
    """
    try:
        print(f"Calculating features for: {mask_file_path}")
        # 1. Load masks
        dedup_masks = load_masks(mask_file_path)
        if not dedup_masks:
            print(f"  No masks loaded from {mask_file_path}. Skipping.")
            return pd.DataFrame()

        # 2. Generate grain labels
        grain_labels = generate_labels(dedup_masks)
        if grain_labels is None:
             print(f"  Failed to generate grain labels for {mask_file_path}. Skipping.")
             return pd.DataFrame()

        # 3. Generate boundary labels
        _, thin_wall_labels, junction_labels = generate_grain_boundaries(grain_labels)
        if thin_wall_labels is None or junction_labels is None:
             print(f"  Failed to generate boundary labels for {mask_file_path}. Skipping.")
             return pd.DataFrame()

        # --- Optional: Calculate and Save Per-Grain Details ---
        if save_details:
            print(f"  Calculating per-grain details...")
            grain_details_list = calculate_per_region_features(grain_labels, region_type_prefix="grain_")
            if grain_details_list:
                grain_details_df = pd.DataFrame(grain_details_list)
                # Determine save path for details file, including sample ID in filename
                base_name = os.path.splitext(os.path.basename(mask_file_path))[0]
                # Get the parent directory name as the sample identifier
                sample_id = os.path.basename(os.path.dirname(mask_file_path))
                # Combine sample_id and base_name for a unique filename
                details_filename = f"{sample_id}_{base_name}_details.csv"
                if details_folder:
                    os.makedirs(details_folder, exist_ok=True)
                    details_save_path = os.path.join(details_folder, details_filename)
                else:
                    details_save_path = os.path.join(os.path.dirname(mask_file_path), details_filename)

                try:
                    grain_details_df.to_csv(details_save_path, index=False)
                    print(f"  Per-grain details saved to: {details_save_path}")
                except Exception as e:
                    print(f"  Warning: Failed to save per-grain details: {e}")
            else:
                print("  No per-grain details generated.")


        # 4. Calculate Aggregate Features for each category
        print(f"  Calculating aggregate features...")
        # Global features based on grains
        global_grain_features = calculate_global_features(grain_labels)

        # Statistics of features for grain regions
        grain_region_stats = calculate_region_features_stats(grain_labels, region_type_prefix="grain_")

        # Statistics of features for thin wall regions
        thinw_region_stats = calculate_region_features_stats(thin_wall_labels, region_type_prefix="thinw_")

        # Statistics of features for junction regions
        junction_region_stats = calculate_region_features_stats(junction_labels, region_type_prefix="junc_")

        # 5. Combine all aggregate features into a single dictionary
        all_features_dict = {
            **global_grain_features,
            **grain_region_stats,
            **thinw_region_stats,
            **junction_region_stats
        }

        # 6. Create Aggregate Features DataFrame and set index based on filename
        # Extract index name (e.g., sample ID) from filename
        index_name_match = re.match(r"^([^_]+)", os.path.basename(mask_file_path))
        if index_name_match:
            index_name = index_name_match.group(1)
        else:
            index_name = os.path.splitext(os.path.basename(mask_file_path))[0] # Fallback to full name without ext

        features_df = pd.DataFrame(all_features_dict, index=[index_name])

        print(f"  Finished feature calculation for: {mask_file_path}")
        return features_df

    except FileNotFoundError:
        print(f"Error: Mask file not found: {mask_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error calculating features for {mask_file_path}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return pd.DataFrame()


def batch_calculate_features(mask_folder_path, save_details=False, details_folder=None):
    """
    Processes a batch of .mask files in a folder, calculates aggregate features for each,
    optionally saves per-grain details, and returns a combined DataFrame of aggregate features.

    Args:
        mask_folder_path (str): Path to the folder containing .mask files
                                (can have subdirectories).
        save_details (bool): If True, save per-grain details for each mask file. Defaults to False.
        details_folder (str, optional): Directory to save the per-grain details CSVs.
                                        If None, saves next to the mask files. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing aggregate features for all processed files.
                      Index is set based on the first-level subdirectory name
                      or a default name ('root') if masks are in the top level.
    """
    all_features_list = []
    print(f"Starting batch feature calculation in: {mask_folder_path}")

    for root, dirs, files in os.walk(mask_folder_path):
        for file in files:
            if file.endswith('.mask'):
                mask_path = os.path.join(root, file)

                # Calculate aggregate features and optionally save details
                features_df = calculate_features_for_maskfile(mask_path, save_details=save_details, details_folder=details_folder)

                if not features_df.empty:
                    # Determine the sample identifier (first-level directory or filename base)
                    # Use filename base as sample ID if no subdirectories, or first subdir name
                    relative_path = os.path.relpath(mask_path, mask_folder_path)
                    path_parts = relative_path.split(os.path.sep)

                    if len(path_parts) > 1: # Has subdirectories
                         sample_id = path_parts[0]
                    else: # File is in the root mask_folder_path
                         # Use filename without extension as ID, maybe remove suffix like '_0001'
                         base_name = os.path.splitext(path_parts[0])[0]
                         id_match = re.match(r"(.+?)(?:_\d+)?$", base_name) # Try to capture base name before potential _number suffix
                         sample_id = id_match.group(1) if id_match else base_name

                    # Set the index for the aggregate features DataFrame
                    features_df.index = [sample_id]
                    all_features_list.append(features_df)

    if not all_features_list:
        print("No mask files processed successfully.")
        return pd.DataFrame()

    # Concatenate all results
    final_features_df = pd.concat(all_features_list, axis=0)
    print(f"Batch feature calculation complete. Total samples processed: {len(final_features_df)}")
    return final_features_df


# Example usage (if run as a script)
if __name__ == '__main__':
    # Example paths - ADJUST AS NEEDED
    # These paths likely need to be provided via command-line arguments in run_feature_extraction.py
    EXAMPLE_MASK_FOLDER = r'path/to/your/masks'
    EXAMPLE_OUTPUT_CSV = 'results/all_features.csv'
    EXAMPLE_DETAILS_FOLDER = 'results/per_grain_details' # Example folder for details

    print("Running feature extraction module example...")
    print(f"Input mask folder: {EXAMPLE_MASK_FOLDER}")
    print(f"Output aggregate CSV: {EXAMPLE_OUTPUT_CSV}")
    print(f"Output details folder: {EXAMPLE_DETAILS_FOLDER}")

    if not os.path.isdir(EXAMPLE_MASK_FOLDER):
        print(f"Error: Example mask folder '{EXAMPLE_MASK_FOLDER}' not found. Cannot run example. Please adjust the path.")
    else:
        try:
            # Example: Run batch processing and save details
            all_features_results = batch_calculate_features(
                EXAMPLE_MASK_FOLDER,
                save_details=True,
                details_folder=EXAMPLE_DETAILS_FOLDER
            )

            if not all_features_results.empty:
                # Save the aggregate results
                os.makedirs(os.path.dirname(EXAMPLE_OUTPUT_CSV), exist_ok=True)
                all_features_results.to_csv(OUTPUT_CSV)
                print(f"Successfully saved features to {OUTPUT_CSV}")
                print("\nFeature DataFrame head:")
                print(all_features_results.head())
                print(f"\nTotal features calculated: {len(all_features_results.columns)}")

        except Exception as main_e:
            print(f"An error occurred during the example run: {main_e}")

    print("Feature extraction module example finished.")
