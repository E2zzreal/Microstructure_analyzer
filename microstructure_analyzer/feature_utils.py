import numpy as np
from skimage import measure, morphology, draw, feature, segmentation
from scipy import ndimage
from scipy.spatial import distance, Delaunay, Voronoi, cKDTree
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from ripser import Rips
from collections import defaultdict
import warnings

# Ignore specific warnings if necessary, or configure logging
warnings.filterwarnings("ignore", category=RuntimeWarning) # Example: Ignore runtime warnings like division by zero

# ======================== Geometric Feature Helpers ========================

def self_similarity_dimension(region, num_scales=10):
    """
    Calculates fractal dimension using a box-counting method on the region's image.

    Args:
        region (skimage.measure._regionprops.RegionProperties): The region object.
        num_scales (int): Number of scales to use for box counting.

    Returns:
        float: The calculated fractal dimension. Returns 0.0 if calculation fails.
    """
    try:
        img = region.image.astype(bool) # Use boolean image for box counting
        if img.size == 0:
            return 0.0
        max_dim = max(img.shape)
        if max_dim <= 1: # Cannot compute for single pixel or line
             return 0.0

        # Generate scales logarithmically from 1 up to max_dim
        scales = np.logspace(0, np.log10(max_dim), num_scales, base=10.0)
        scales = np.unique(np.round(scales).astype(int))
        scales = scales[scales > 0] # Ensure scales are positive

        if len(scales) < 2: # Need at least two scales for fitting
            return 0.0

        counts = []
        valid_scales = []

        for scale in scales:
            if scale > min(img.shape): # Scale cannot be larger than the smallest dimension
                continue

            # Calculate number of boxes needed to cover the shape
            # Pad image to be divisible by scale
            padded_shape = (np.ceil(img.shape[0] / scale) * scale, np.ceil(img.shape[1] / scale) * scale)
            padded_img = np.pad(img,
                                ((0, int(padded_shape[0] - img.shape[0])),
                                 (0, int(padded_shape[1] - img.shape[1]))),
                                mode='constant', constant_values=0)

            # Use block_reduce to count boxes containing part of the object
            block_shape = (scale, scale)
            reduced = measure.block_reduce(padded_img, block_shape, np.any)
            count = np.sum(reduced)

            if count > 0: # Only consider scales where the object is detected
                counts.append(count)
                valid_scales.append(scale)

        if len(counts) < 2:
            return 0.0 # Not enough data points to fit a line

        # Fit log(count) vs log(1/scale)
        # Using log(scale) directly and taking negative slope is equivalent
        coeffs = np.polyfit(np.log(valid_scales), np.log(counts), 1)
        fractal_dim = -coeffs[0] # Fractal dimension is the negative slope
        return fractal_dim

    except Exception as e:
        # print(f"Warning: Fractal dimension calculation failed for region {region.label}: {e}")
        return 0.0


def count_skeleton_branches(skeleton):
    """
    Counts branch points in a skeletonized image.
    A branch point is a skeleton pixel with 3 or more neighbors.

    Args:
        skeleton (np.ndarray): A boolean skeleton image.

    Returns:
        int: The number of branch points.
    """
    if not np.any(skeleton):
        return 0
    if skeleton.ndim != 2 or skeleton.dtype != bool:
        skeleton = np.atleast_2d(skeleton).astype(bool)

    # Use a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1], # Don't count the center pixel itself
                       [1, 1, 1]], dtype=np.uint8)
    # Convolve to get neighbor count for each pixel
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

    # Branch points are skeleton pixels with more than 2 neighbors
    branch_points = (skeleton & (neighbor_count > 2))
    return np.sum(branch_points)


def calculate_orientation(region):
    """
    Calculates the orientation of the region using PCA.

    Args:
        region (skimage.measure._regionprops.RegionProperties): The region object.

    Returns:
        float: Orientation angle in radians, or 0.0 if calculation fails.
    """
    coords = region.coords
    if coords.shape[0] < 2: # Need at least 2 points for PCA
        return 0.0

    try:
        # Center the coordinates
        coords_centered = coords - coords.mean(axis=0)
        # Perform PCA
        pca = PCA(n_components=2) # Calculate both components
        pca.fit(coords_centered)
        # The orientation is the angle of the first principal component vector
        major_vector = pca.components_[0]
        orientation = np.arctan2(major_vector[1], major_vector[0]) # Angle with respect to x-axis
        return orientation
    except Exception as e:
        # print(f"Warning: Orientation calculation failed for region {region.label}: {e}")
        return 0.0

# ======================== Grain Property Helpers ========================

def calculate_curvature(coords):
    """
    Calculates the curvature for each point on a 2D contour.
    Uses gradient method for numerical differentiation.

    Args:
        coords (np.ndarray): Array of contour coordinates (N, 2), typically (y, x).

    Returns:
        np.ndarray: Array of curvature values for each point (N,). Returns zeros if input is too small.
    """
    if coords.shape[0] < 3: # Need at least 3 points to estimate curvature reliably
        return np.zeros(coords.shape[0])

    try:
        # Ensure float type for calculations
        coords = coords.astype(float)

        # Calculate first derivatives (velocity components)
        dx = np.gradient(coords[:, 1]) # Gradient of x-coordinates
        dy = np.gradient(coords[:, 0]) # Gradient of y-coordinates

        # Calculate second derivatives (acceleration components)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # Calculate curvature using the formula: K = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        denominator = (dx**2 + dy**2)**1.5
        # Avoid division by zero for stationary points or straight lines
        denominator[denominator < 1e-9] = 1e-9 # Replace near-zero with small epsilon

        curvature = np.abs(dx * d2y - dy * d2x) / denominator

        # Handle potential NaNs or Infs resulting from edge cases or numerical issues
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

        return curvature

    except Exception as e:
        # print(f"Warning: Curvature calculation failed: {e}")
        return np.zeros(coords.shape[0])


def calculate_grain_boundary_properties(labels):
    """
    Calculates total boundary length and mean curvature for all grains.

    Args:
        labels (np.ndarray): Labeled image where each grain has a unique integer ID > 0.

    Returns:
        tuple: (total_boundary_length, mean_boundary_curvature)
    """
    total_boundary_length = 0
    all_curvatures = []

    # Find boundaries between labeled regions (mode='inner' excludes background boundaries)
    boundaries = segmentation.find_boundaries(labels, mode='inner', background=0)
    # Label the boundary pixels with the label of the grain they belong to
    # Note: A boundary pixel might be adjacent to multiple labels. find_boundaries assigns it somewhat arbitrarily.
    # A more robust approach might involve iterating through regions and their specific boundaries.

    regions = measure.regionprops(labels)
    boundary_pixels_labeled = labels * boundaries # Pixels on boundary get their grain label

    for region in regions:
        label = region.label
        # Find boundary pixels associated with this specific region
        region_boundary_indices = np.where(boundary_pixels_labeled == label)

        if len(region_boundary_indices[0]) > 0:
            # Length is approximated by the number of boundary pixels for this region
            # This might double-count boundaries shared between two grains if summed naively.
            # A better approach for total length is sum(boundaries).
            # For per-grain length, this count is reasonable.
            boundary_length = len(region_boundary_indices[0])
            # total_boundary_length += boundary_length # This would overestimate total length

            # Calculate curvature for this region's boundary segment
            if boundary_length >= 3:
                # Need to order the boundary pixels to form a contour
                # This is non-trivial. Using region.coords on the boundary mask might work if connected.
                # Alternative: Use find_contours on the region's mask.
                contours = measure.find_contours(region.image, 0.5) # Find contour on the region's binary mask
                if contours:
                    # Use the longest contour, convert to global coordinates
                    contour = contours[0] + region.bbox[:2]
                    curvatures = calculate_curvature(contour)
                    all_curvatures.extend(curvatures)

    # Calculate total boundary length directly from the boundary mask
    total_boundary_length = np.sum(boundaries)

    mean_curvature = np.mean(all_curvatures) if all_curvatures else 0.0

    return total_boundary_length, mean_curvature


def calculate_nearest_neighbor_distances(labels):
    """
    Calculates statistics of nearest neighbor distances between grain centroids.

    Args:
        labels (np.ndarray): Labeled image.

    Returns:
        tuple: (mean_nn_distance, std_nn_distance)
    """
    regions = measure.regionprops(labels)
    if len(regions) < 2:
        return 0.0, 0.0

    centroids = np.array([r.centroid for r in regions])

    try:
        # Use sklearn's NearestNeighbors (often faster than cKDTree for lower dimensions)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)

        # distances[:, 0] is distance to self (0), distances[:, 1] is distance to nearest neighbor
        nn_distances = distances[:, 1]

        nn_dist_mean = np.mean(nn_distances)
        nn_dist_std = np.std(nn_distances)

        return nn_dist_mean, nn_dist_std

    except Exception as e:
        # print(f"Warning: Nearest neighbor calculation failed: {e}")
        return 0.0, 0.0


def calculate_orientation_statistics(labels):
    """
    Calculates statistics of grain orientations.

    Args:
        labels (np.ndarray): Labeled image.

    Returns:
        tuple: (mean_orientation, std_orientation) in radians.
    """
    regions = measure.regionprops(labels)
    if not regions:
        return 0.0, 0.0

    orientations = [r.orientation for r in regions]
    # Handle potential issues with orientation calculation in regionprops if needed

    orientation_mean = np.mean(orientations)
    orientation_std = np.std(orientations)

    return orientation_mean, orientation_std


def box_counting_fractal_dimension(binary_image):
    """
    Calculates fractal dimension using box counting on a binary image.
    (Simplified version from the notebook, potentially less robust than self_similarity_dimension)

    Args:
        binary_image (np.ndarray): Boolean array where True represents the object.

    Returns:
        float: Fractal dimension, or 0.0 if calculation fails.
    """
    if not np.any(binary_image):
        return 0.0

    # Limit box sizes to reasonable range, e.g., powers of 2 up to half the smallest dimension
    min_dim = min(binary_image.shape)
    if min_dim < 2: return 0.0
    max_exponent = int(np.log2(min_dim / 2))
    if max_exponent < 1: return 0.0
    box_sizes = 2 ** np.arange(1, max_exponent + 1)

    counts = []
    valid_scales = []

    for size in box_sizes:
        # Calculate number of boxes needed
        rows = int(np.ceil(binary_image.shape[0] / size))
        cols = int(np.ceil(binary_image.shape[1] / size))
        count = 0
        for i in range(rows):
            for j in range(cols):
                # Extract box
                box = binary_image[i*size:(i+1)*size, j*size:(j+1)*size]
                # Count if any part of the object is in the box
                if np.any(box):
                    count += 1
        if count > 0:
            counts.append(count)
            valid_scales.append(size)

    if len(counts) < 2:
        return 0.0 # Need at least 2 points to fit

    # Fit log(count) vs log(1/size) or log(size)
    try:
        coeffs = np.polyfit(np.log(valid_scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]
        return fractal_dim
    except Exception as e:
        # print(f"Warning: Box counting fit failed: {e}")
        return 0.0


def calculate_fractal_dimension_on_boundaries(labels):
    """
    Calculates fractal dimension specifically on the grain boundaries.

    Args:
        labels (np.ndarray): Labeled image.

    Returns:
        float: Fractal dimension of the boundaries.
    """
    boundaries = segmentation.find_boundaries(labels, mode='inner', background=0)
    return box_counting_fractal_dimension(boundaries)


def calculate_persistence_homology(labels, max_dist_thresh=100):
    """
    Calculates persistence homology features (mean persistence for H0 and H1).

    Args:
        labels (np.ndarray): Labeled image.
        max_dist_thresh (float): Maximum distance threshold for Rips complex construction.

    Returns:
        tuple: (mean_H0_persistence, mean_H1_persistence)
    """
    regions = measure.regionprops(labels)
    if len(regions) < 2:
        return 0.0, 0.0

    centroids = np.array([r.centroid for r in regions])

    try:
        rips = Rips(maxdim=1, thresh=max_dist_thresh, verbose=False) # Set verbose=False
        diagrams = rips.fit_transform(centroids)

        # H0 features (connected components persistence)
        persistence_0 = diagrams[0]
        # Filter out infinite persistence (usually one component lasts forever)
        finite_persistence_0 = persistence_0[persistence_0[:, 1] != np.inf]
        lifespans_0 = finite_persistence_0[:, 1] - finite_persistence_0[:, 0]
        persistence_0_mean = np.mean(lifespans_0) if len(lifespans_0) > 0 else 0.0

        # H1 features (loop persistence)
        persistence_1 = diagrams[1]
        finite_persistence_1 = persistence_1[persistence_1[:, 1] != np.inf] # Should not happen for H1 with thresh
        lifespans_1 = finite_persistence_1[:, 1] - finite_persistence_1[:, 0]
        persistence_1_mean = np.mean(lifespans_1) if len(lifespans_1) > 0 else 0.0

        return persistence_0_mean, persistence_1_mean

    except Exception as e:
        # print(f"Warning: Persistence homology calculation failed: {e}")
        return 0.0, 0.0


def calculate_pairwise_distances(labels):
    """
    Calculates statistics of pairwise distances between all grain centroids.

    Args:
        labels (np.ndarray): Labeled image.

    Returns:
        tuple: (mean_pairwise_distance, std_pairwise_distance)
    """
    regions = measure.regionprops(labels)
    if len(regions) < 2:
        return 0.0, 0.0

    centroids = np.array([r.centroid for r in regions])

    try:
        # Calculate all pairwise distances
        distances = pdist(centroids, metric='euclidean')
        if len(distances) == 0:
             return 0.0, 0.0

        pairwise_dist_mean = np.mean(distances)
        pairwise_dist_std = np.std(distances)

        return pairwise_dist_mean, pairwise_dist_std
    except Exception as e:
        # print(f"Warning: Pairwise distance calculation failed: {e}")
        return 0.0, 0.0


def build_delaunay_graph(centroids, distance_threshold=None):
    """Builds a graph based on Delaunay triangulation of centroids."""
    if len(centroids) < 3:
        return nx.Graph() # Need at least 3 points for triangulation

    G = nx.Graph()
    try:
        tri = Delaunay(centroids)

        # Add nodes (use index as node ID)
        for i in range(len(centroids)):
            G.add_node(i, pos=centroids[i]) # Store position for potential visualization

        # Add edges from Delaunay triangulation
        for simplex in tri.simplices:
            # Each simplex is a triangle [idx1, idx2, idx3]
            for i in range(3):
                u = simplex[i]
                v = simplex[(i + 1) % 3] # Connect vertices of the triangle
                # Ensure edge doesn't already exist (shouldn't happen with simple graphs)
                if not G.has_edge(u, v):
                    distance = np.linalg.norm(centroids[u] - centroids[v])
                    # Optionally filter edges longer than a threshold
                    if distance_threshold is None or distance <= distance_threshold:
                        G.add_edge(u, v, weight=distance)
        return G

    except Exception as e:
        # print(f"Warning: Delaunay graph construction failed: {e}")
        return nx.Graph() # Return empty graph on failure


def calculate_grain_network_features(labels, distance_threshold=50, threshold_name=None):
    """
    Calculates graph-based features from the grain network (Delaunay).

    Args:
        labels (np.ndarray): Labeled image.
        distance_threshold (float): Max distance for edges in Delaunay graph.
        threshold_name (str, optional): A name for the threshold used (e.g., 'fixed_50', 'adaptive_1r0.5std').
                                        If provided, used in the output feature keys. Defaults to None.

    Returns:
        dict: Dictionary containing network features like degree distribution counts,
              average clustering coefficient, and small-world-ness flag. Keys will include
              the threshold_name or distance_threshold value as a suffix.
    """
    features = {}
    # Determine the suffix for feature keys
    if threshold_name:
        suffix = threshold_name
    else:
        # Use the numerical threshold, ensuring it's a valid string representation
        suffix = str(distance_threshold).replace('.', '_') # Replace dots if float

    regions = measure.regionprops(labels)
    if len(regions) < 3: # Need enough grains for network analysis
        # Return default values for all expected features using the suffix
        for d in range(9): features[f'degree_dist_{d}_{suffix}'] = 0
        features[f'avg_clustering_{suffix}'] = 0.0
        features[f'is_small_world_{suffix}'] = False
        return features

    # Use region centroids (y, x) but Delaunay expects (x, y)
    centroids = np.array([r.centroid[::-1] for r in regions]) # Convert to (x, y)
    G = build_delaunay_graph(centroids, distance_threshold)

    if G.number_of_nodes() == 0: # Handle empty graph case
        for d in range(9): features[f'degree_dist_{d}_{suffix}'] = 0
        features[f'avg_clustering_{suffix}'] = 0.0
        features[f'is_small_world_{suffix}'] = False
        return features


    # 1. Degree Distribution (count nodes with degree 0 to 8)
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    for d in range(9): # Calculate counts for degree 0 through 8
        features[f'degree_dist_{d}_{suffix}'] = degree_values.count(d)

    # 2. Average Clustering Coefficient
    try:
        avg_clustering = nx.average_clustering(G)
    except ZeroDivisionError:
        avg_clustering = 0.0 # Handle cases with no triangles
    features[f'avg_clustering_{suffix}'] = avg_clustering

    # 3. Small-World-ness (Compare to random graph)
    # Note: Small-world calculation can be complex and sensitive.
    # Keeping the existing logic, but be aware of its limitations.
    is_small_world = False # Default
    if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        try:
            # Analyze the largest connected component for path length
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc).copy()

            if G_largest.number_of_nodes() > 1:
                # Generate reference random graph (Erdos-Renyi)
                n_nodes = G_largest.number_of_nodes()
                # Calculate probability p for GNM model based on actual edges in largest component
                m_edges = G_largest.number_of_edges()
                if n_nodes > 1:
                     p = (2 * m_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
                     p = min(max(p, 0), 1) # Ensure p is valid probability
                     random_G = nx.erdos_renyi_graph(n_nodes, p)

                     # Calculate properties for comparison (on largest component of random graph too)
                     if random_G.number_of_nodes() > 0:
                          largest_cc_random = max(nx.connected_components(random_G), key=len, default=None)
                          if largest_cc_random and len(largest_cc_random) > 1:
                               random_G_largest = random_G.subgraph(largest_cc_random).copy()
                               random_clustering = nx.average_clustering(random_G_largest)
                               random_path = nx.average_shortest_path_length(random_G_largest)

                               # Calculate properties of actual graph's largest component
                               real_clustering = nx.average_clustering(G_largest) # Use clustering of largest component
                               real_path = nx.average_shortest_path_length(G_largest)

                               # Small-world criteria (High clustering, similar path length)
                               # Check random_path > 0 to avoid division by zero
                               if random_clustering > 1e-6 and random_path > 1e-6: # Avoid trivial cases
                                    sigma = (real_clustering / random_clustering) / (real_path / random_path)
                                    # Alternative simple check: C >> Cr and L approx Lr
                                    if real_clustering > random_clustering and real_path < random_path * 2: # Heuristic check
                                         is_small_world = True
                          else: # Random graph disconnected or too small
                               is_small_world = False # Cannot compare reliably
                     else: # Random graph has no nodes
                          is_small_world = False

            else: # Largest component has only 1 node
                is_small_world = False # Cannot calculate path length or clustering

        except Exception as e:
            # print(f"Warning: Small-world calculation failed: {e}")
            is_small_world = False # Default to False on error

    features[f'is_small_world_{suffix}'] = is_small_world

    return features


# ======================== Per-Region Shape Feature Helpers ========================

def calculate_perimeter_area_ratio(region):
    """Calculates Perimeter / sqrt(Area)."""
    perimeter = region.perimeter
    area = region.area
    if area <= 0: return 0.0
    return perimeter / np.sqrt(area)

def calculate_compactness(region):
    """Calculates (4 * pi * Area) / Perimeter^2 (circularity)."""
    perimeter = region.perimeter
    area = region.area
    if perimeter <= 0: return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)

def calculate_rectangularity(region):
    """Calculates Area / BoundingBoxArea."""
    area = region.area
    minr, minc, maxr, maxc = region.bbox
    bbox_area = (maxr - minr) * (maxc - minc)
    if bbox_area <= 0: return 0.0
    return area / bbox_area

def calculate_convex_defect(region):
    """Calculates (ConvexArea - Area) / Area."""
    try:
        # convex_hull_image requires the region's image mask
        convex_hull = morphology.convex_hull_image(region.image)
        convex_area = np.sum(convex_hull)
        actual_area = region.area # Use regionprops area for consistency
        if actual_area <= 0: return 0.0
        # Ensure convex area is not smaller than actual area due to discretization
        convex_area = max(convex_area, actual_area)
        return (convex_area - actual_area) / actual_area
    except Exception as e:
        # print(f"Warning: Convex defect calculation failed for region {region.label}: {e}")
        return 0.0

def calculate_tortuosity(boundary_coords):
    """Calculates PathLength / EndToEndDistance for a boundary contour."""
    if boundary_coords.shape[0] < 2:
        return 0.0 # Or 1.0 if defined differently for single points

    try:
        # Calculate actual path length (sum of segment lengths)
        segment_lengths = np.linalg.norm(np.diff(boundary_coords, axis=0), axis=1)
        path_length = np.sum(segment_lengths)

        # Calculate straight-line distance between start and end points
        start_end_distance = np.linalg.norm(boundary_coords[0] - boundary_coords[-1])

        if start_end_distance < 1e-6: # Avoid division by zero if start/end points are identical
             # If path length is also near zero, it's a point, tortuosity is low (e.g., 1)
             # If path length is significant, it's a closed loop, tortuosity is high (inf or large number)
             return path_length / 1e-6 if path_length > 1e-6 else 1.0

        return path_length / start_end_distance
    except Exception as e:
        # print(f"Warning: Tortuosity calculation failed: {e}")
        return 0.0


def calculate_curvature_std(boundary_coords):
    """Calculates the standard deviation of curvature along a boundary."""
    if boundary_coords.shape[0] < 3:
        return 0.0
    curvatures = calculate_curvature(boundary_coords)
    return np.std(curvatures)


def calculate_fourier_energy_ratio(boundary_coords, n_harmonics=10):
    """
    Calculates the ratio of high-frequency energy to total energy using Fourier descriptors.

    Args:
        boundary_coords (np.ndarray): Ordered boundary coordinates (N, 2).
        n_harmonics (int): Number of high-frequency harmonics to consider.

    Returns:
        float: Ratio of high-frequency energy.
    """
    if boundary_coords.shape[0] < 2:
        return 0.0

    try:
        # Convert coordinates to complex numbers (x + iy)
        # Assuming input is (y, x), convert to (x + iy)
        complex_coords = boundary_coords[:, 1] + 1j * boundary_coords[:, 0]

        # Perform Fast Fourier Transform
        fft_coeffs = np.fft.fft(complex_coords)

        # Calculate energy (magnitude squared of coefficients)
        # Exclude the DC component (fft_coeffs[0]) which relates to the centroid
        energies = np.abs(fft_coeffs[1:])**2
        total_energy = np.sum(energies)

        if total_energy < 1e-9:
            return 0.0 # No energy / variation

        # Calculate high-frequency energy (last n_harmonics)
        # Need to handle symmetry of FFT for real signals if applicable, but complex coords are general.
        # Consider taking harmonics from the end of the spectrum.
        num_coeffs = len(fft_coeffs)
        # Ensure n_harmonics is not larger than available high-frequency components
        actual_harmonics = min(n_harmonics, (num_coeffs - 1) // 2) # Use up to Nyquist limit approx

        if actual_harmonics <= 0:
             return 0.0

        # Sum energy from highest frequencies (excluding DC)
        # FFT output: [DC, f1, f2, ..., f_nyquist, ..., f_-2, f_-1]
        # High frequencies are near the middle and at the end. Let's take from the end.
        high_freq_energy = np.sum(np.abs(fft_coeffs[-actual_harmonics:])**2)

        return high_freq_energy / total_energy

    except Exception as e:
        # print(f"Warning: Fourier energy ratio calculation failed: {e}")
        return 0.0
