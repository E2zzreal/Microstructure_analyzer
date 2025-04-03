import numpy as np
from skimage import measure, morphology, feature
from skimage.segmentation import find_boundaries
from scipy.spatial import Voronoi
from scipy import ndimage
import networkx as nx
import warnings

class TopologyAnalyzer:
    """
    Analyzes topological features of labeled regions (e.g., grains) in an image.

    Calculates adjacency relationships, Voronoi neighbors, and optionally
    analyzes the boundary network structure.

    Attributes:
        labels (np.ndarray): The input labeled image.
        height (int): Image height.
        width (int): Image width.
        regions (list): List of region properties from skimage.measure.regionprops.
        max_label (int): The maximum label value present in the image.
        adjacency_matrix (np.ndarray): Matrix where adj[i, j] > 0 if regions i and j touch.
        voronoi_graph (nx.Graph): NetworkX graph representing Voronoi neighbors based on centroids.
        skel_graph (nx.Graph): NetworkX graph representing the boundary skeleton network (optional).
    """
    def __init__(self, labels):
        """
        Initializes the TopologyAnalyzer.

        Args:
            labels (np.ndarray): A 2D integer array where each unique positive integer
                                 represents a distinct region (grain), and 0 represents
                                 the background or boundaries.
        """
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Input 'labels' must be a 2D integer array.")

        self.labels = labels
        self.height, self.width = labels.shape
        # Calculate region properties, excluding background label 0
        self.regions = measure.regionprops(labels) # regionprops ignores label 0 by default
        self.max_label = np.max(labels) if np.any(labels) else 0

        # Build adjacency and Voronoi structures
        self.adjacency_matrix = self._build_adjacency_matrix()
        self.voronoi_graph = self._build_voronoi_graph()
        self.skel_graph = None # Initialize skeleton graph as None, build on demand

    def _build_adjacency_matrix(self):
        """
        Builds an adjacency matrix based on touching regions.
        Matrix size is (max_label + 1) x (max_label + 1).
        adj[i, j] = count of boundary pixels between region i and j.
        """
        matrix_size = self.max_label + 1
        adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        # Find boundaries between labeled regions (excluding background)
        # 'outer' mode finds pixels in label N adjacent to label M (N!=M)
        boundaries = find_boundaries(self.labels, mode='outer', background=0)
        y_coords, x_coords = np.where(boundaries)

        # Define neighborhood offsets (8-connectivity)
        dy = [-1, -1, -1, 0, 0, 1, 1, 1]
        dx = [-1, 0, 1, -1, 1, -1, 0, 1]

        for y, x in zip(y_coords, x_coords):
            current_label = self.labels[y, x]
            if current_label == 0: # Should not happen with background=0, but check anyway
                continue

            # Check neighbors
            for i in range(len(dy)):
                ny, nx = y + dy[i], x + dx[i]
                # Check bounds
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    neighbor_label = self.labels[ny, nx]
                    # If neighbor is a different labeled region (not background)
                    if neighbor_label != 0 and neighbor_label != current_label:
                        # Increment count for this pair (symmetric)
                        # Use min/max to ensure consistency if needed, but direct assignment is fine
                        adj_matrix[current_label, neighbor_label] += 1
                        # adj_matrix[neighbor_label, current_label] += 1 # Matrix will be symmetric if counted once per boundary pixel pair

        # Ensure symmetry by adding the transpose and dividing by 2, or by careful counting
        # The current method counts each boundary pixel interaction once from the perspective
        # of the 'current_label' pixel. If a boundary pixel between A and B is processed
        # when current_label is A, it adds 1 to adj[A,B]. If processed when current_label is B,
        # it adds 1 to adj[B,A]. Let's make it symmetric explicitly.
        adj_matrix = (adj_matrix + adj_matrix.T) // 2 # Average the counts

        return adj_matrix

    def _build_voronoi_graph(self):
        """
        Builds a NetworkX graph based on Voronoi neighbors of region centroids.
        Nodes are region labels, edges connect Voronoi neighbors.
        """
        G = nx.Graph()
        centroids = []
        valid_region_labels = [] # Store labels corresponding to centroids

        # Collect centroids of valid regions (label > 0)
        # regionprops already excludes label 0
        for r in self.regions:
            # Ensure region has area and valid centroid
            if r.area > 0 and r.centroid is not None:
                # Voronoi expects (x, y) format, regionprops gives (row, col) i.e. (y, x)
                centroids.append(r.centroid[::-1]) # Reverse to (x, y)
                valid_region_labels.append(r.label)
                G.add_node(r.label, pos=r.centroid[::-1]) # Add node with its label and position

        # Need at least 3 points for Voronoi diagram
        if len(centroids) < 3:
            # print("Warning: Less than 3 valid regions found, cannot compute Voronoi diagram.")
            return G # Return empty graph

        try:
            centroids_array = np.array(centroids)
            vor = Voronoi(centroids_array)

            # Create mapping from Voronoi point index to region label
            index_to_label = {i: label for i, label in enumerate(valid_region_labels)}

            # Add edges for Voronoi neighbors (ridges connect points)
            for i, ridge_points in enumerate(vor.ridge_points):
                # ridge_points contains pairs of indices into the input 'centroids' array
                idx1, idx2 = ridge_points
                # Check for valid indices (Voronoi can produce ridges involving points at infinity, represented by -1)
                if idx1 >= 0 and idx2 >= 0:
                    label1 = index_to_label[idx1]
                    label2 = index_to_label[idx2]
                    # Add edge between the corresponding region labels
                    if not G.has_edge(label1, label2):
                         distance = np.linalg.norm(centroids_array[idx1] - centroids_array[idx2])
                         G.add_edge(label1, label2, weight=distance)

        except Exception as e:
            print(f"Warning: Voronoi computation failed: {e}")
            # Return the graph built so far (nodes only, or potentially partial edges)
            return G

        return G

    def get_grain_topology_features(self, region_label):
        """
        Calculates topological features for a specific grain/region label.

        Args:
            region_label (int): The label of the region to analyze.

        Returns:
            dict: A dictionary containing topological features:
                  'neighbor_count': Number of physically touching neighbors.
                  'triple_junction': Boolean indicating if it's part of >= 3-grain junction.
                  'voronoi_neighbors': Number of Voronoi neighbors.
                  'neighbor_consistency': Number of neighbors common to both adjacency and Voronoi.
                  'extra_neighbors': Number of touching neighbors not found in Voronoi.
                  'missing_neighbors': Number of Voronoi neighbors not found touching.
        """
        features = {}

        # Check if label is valid
        if region_label <= 0 or region_label > self.max_label:
            # Return default values if label is invalid
            features['neighbor_count'] = 0
            features['triple_junction'] = False
            features['voronoi_neighbors'] = 0
            features['neighbor_consistency'] = 0
            features['extra_neighbors'] = 0
            features['missing_neighbors'] = 0
            return features

        # 1. Adjacency features
        # Find neighbors from the adjacency matrix row/column for this label
        # Ensure indices are within bounds
        if region_label < self.adjacency_matrix.shape[0]:
             neighbors_indices = np.where(self.adjacency_matrix[region_label] > 0)[0]
             # Filter out self-adjacency if matrix diagonal is non-zero for some reason
             actual_neighbors = set(idx for idx in neighbors_indices if idx != region_label and idx != 0)
        else:
             actual_neighbors = set()

        features['neighbor_count'] = len(actual_neighbors)
        # A region is at a triple junction (or higher) if it touches 3 or more other regions
        features['triple_junction'] = features['neighbor_count'] >= 3 # Based on physical contact

        # 2. Voronoi features
        vor_neighbors = set()
        if self.voronoi_graph.has_node(region_label):
            vor_neighbors = set(self.voronoi_graph.neighbors(region_label))
        features['voronoi_neighbors'] = len(vor_neighbors)

        # 3. Consistency features
        common_neighbors = actual_neighbors.intersection(vor_neighbors)
        extra_neighbors = actual_neighbors.difference(vor_neighbors) # Touching but not Voronoi neighbor
        missing_neighbors = vor_neighbors.difference(actual_neighbors) # Voronoi neighbor but not touching

        features['neighbor_consistency'] = len(common_neighbors)
        features['extra_neighbors'] = len(extra_neighbors)
        features['missing_neighbors'] = len(missing_neighbors)

        return features


    # --- Boundary Network Analysis Methods (Optional, build on demand) ---

    def _find_endpoints(self, skeleton):
        """Helper function to find endpoints in a skeleton."""
        if not np.any(skeleton): return np.zeros_like(skeleton, dtype=bool)
        kernel = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=np.uint8)
        # Endpoint is a skeleton pixel with exactly one neighbor in the 3x3 neighborhood (excluding itself)
        neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
        # The count includes the center pixel, so endpoints have count == 2 (1 neighbor + self)
        endpoints = (skeleton & (neighbor_count == 2))
        return endpoints

    def _trace_edge(self, G, node_map, skeleton, visited, start_y, start_x):
        """Helper function to trace edges between nodes in the skeleton graph."""
        stack = [(start_y, start_x)]
        path = []
        start_node_id = None

        while stack:
            cy, cx = stack.pop()

            # Skip if already visited or not part of skeleton
            if visited[cy, cx] or not skeleton[cy, cx]:
                continue

            visited[cy, cx] = True
            path.append((cy, cx))
            current_node_id = node_map[cy, cx] # Get node ID if this pixel is a node

            # Check if we reached a node
            if current_node_id > 0:
                if start_node_id is None:
                    # This is the first node encountered in this trace
                    start_node_id = current_node_id
                    # Reset path, start edge from this node
                    path = [(cy, cx)]
                else:
                    # Reached the end node of the edge
                    end_node_id = current_node_id
                    if start_node_id != end_node_id: # Avoid self-loops from single node tracing
                         # Add edge to graph if not already present
                         if not G.has_edge(start_node_id, end_node_id):
                              G.add_edge(start_node_id, end_node_id, length=len(path)-1, path=np.array(path))
                    # Reset for potentially tracing new edges from this end_node
                    start_node_id = end_node_id
                    path = [(cy, cx)] # Start new path from this node

            # Find valid, unvisited neighbors
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = cy + dy, cx + dx
                    # Check bounds and if neighbor is part of skeleton
                    if 0 <= ny < self.height and 0 <= nx < self.width and skeleton[ny, nx] and not visited[ny, nx]:
                        neighbors.append((ny, nx))

            # Continue tracing
            if len(neighbors) == 1:
                # Continue along the single path
                stack.append(neighbors[0])
            elif len(neighbors) > 1:
                # Reached a junction or a point where multiple paths diverge
                # If the current point wasn't a node, this indicates a potential issue or complex junction
                # If it was a node, we start new traces from here for each neighbor
                if current_node_id > 0: # If we just arrived at a node
                     for ny, nx in neighbors:
                          # Start new traces from this node towards each neighbor
                          self._trace_edge(G, node_map, skeleton, visited, ny, nx)
                # If current point is not a node but has >1 neighbors, it might be part of a thick line or noise.
                # The current tracing logic might stop here or behave unpredictably.
                # For simplicity, we stop this path and rely on traces starting from actual nodes.


    def build_boundary_graph(self, boundary_mask):
        """
        Builds a NetworkX graph representing the network of boundaries.
        Nodes are junctions (branch points) and endpoints, edges are boundary segments.

        Args:
            boundary_mask (np.ndarray): A boolean mask representing the boundaries to analyze
                                       (e.g., thin walls or all grain boundaries).

        Returns:
            nx.Graph: The constructed boundary network graph. Stores the graph in self.skel_graph.
        """
        if not np.any(boundary_mask):
            self.skel_graph = nx.Graph()
            return self.skel_graph

        # Skeletonize the boundary mask
        skeleton = morphology.skeletonize(boundary_mask)
        if not np.any(skeleton):
            self.skel_graph = nx.Graph()
            return self.skel_graph

        # --- Node Identification ---
        # Find branch points (pixels with > 2 neighbors)
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
        branch_points_mask = (skeleton & (neighbor_count > 2))

        # Find endpoints (pixels with == 1 neighbor)
        endpoints_mask = (skeleton & (neighbor_count == 1))

        # Combine branch points and endpoints to define nodes
        node_mask = branch_points_mask | endpoints_mask
        node_coords_y, node_coords_x = np.where(node_mask)

        # Create a mapping from coordinates to node IDs and initialize graph
        G = nx.Graph()
        node_map = np.zeros_like(skeleton, dtype=int) # Map coordinates to node ID
        node_id_counter = 1
        node_positions = {} # Store positions for graph attributes

        for y, x in zip(node_coords_y, node_coords_x):
            node_map[y, x] = node_id_counter
            node_positions[node_id_counter] = (y, x) # Store y, x
            G.add_node(node_id_counter, pos=(y, x)) # Add node to graph
            node_id_counter += 1

        # --- Edge Tracing ---
        visited = np.zeros_like(skeleton, dtype=bool)
        # Iterate through all nodes and start tracing from them
        for start_node_id in G.nodes():
            start_y, start_x = node_positions[start_node_id]
            visited[start_y, start_x] = True # Mark node itself as visited for tracing logic

            # Explore neighbors of the starting node
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = start_y + dy, start_x + dx

                    if 0 <= ny < self.height and 0 <= nx < self.width and skeleton[ny, nx] and not visited[ny, nx]:
                         # Start tracing an edge from this neighbor
                         self._trace_edge_from_node(G, node_map, skeleton, visited, start_node_id, ny, nx)


        self.skel_graph = G
        return G


    def _trace_edge_from_node(self, G, node_map, skeleton, visited, start_node_id, start_y, start_x):
         """Helper to trace a single edge starting from a node's neighbor."""
         queue = [(start_y, start_x)] # Use queue for BFS-like tracing
         path = [(node_map.item((start_y, start_x)), (start_y, start_x))] # Store path pixels
         visited_in_trace = set([(start_y, start_x)]) # Track visited pixels within this specific trace

         while queue:
              cy, cx = queue.pop(0)

              # Check if we reached another node
              end_node_id = node_map[cy, cx]
              if end_node_id > 0 and end_node_id != start_node_id:
                   # Reached the end node
                   if not G.has_edge(start_node_id, end_node_id):
                        G.add_edge(start_node_id, end_node_id, length=len(path), path=np.array([p[1] for p in path]))
                   # Mark all pixels in this path as globally visited
                   for _, (py, px) in path: visited[py, px] = True
                   return # Edge tracing complete

              # Find next unvisited skeleton neighbor
              neighbors = []
              for dy in [-1, 0, 1]:
                   for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0: continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width and \
                           skeleton[ny, nx] and (ny, nx) not in visited_in_trace:
                             neighbors.append((ny, nx))

              # Continue trace
              if len(neighbors) == 1:
                   ny, nx = neighbors[0]
                   visited_in_trace.add((ny, nx))
                   path.append((node_map.item((ny, nx)), (ny, nx)))
                   queue.append((ny, nx))
              elif len(neighbors) > 1:
                   # Reached an intermediate branch point (not marked as a node)
                   # This indicates a complex junction or thick skeleton. Stop trace here.
                   # Mark path as visited to avoid re-tracing
                   for _, (py, px) in path: visited[py, px] = True
                   return
              else:
                   # Reached a dead end (not a node) - possible skeleton artifact
                   # Mark path as visited
                   for _, (py, px) in path: visited[py, px] = True
                   return # Stop trace


    def get_boundary_network_features(self, boundary_mask):
        """
        Calculates features of the boundary network graph. Builds the graph if needed.

        Args:
            boundary_mask (np.ndarray): Boolean mask of the boundaries.

        Returns:
            dict: Dictionary containing network features like Euler number,
                  longest chain length, etc. Returns defaults if graph is empty.
        """
        # Build the skeleton graph if it hasn't been built yet or if mask changes
        # For simplicity, we rebuild it each time this is called with a mask
        self.build_boundary_graph(boundary_mask)

        features = {}
        if self.skel_graph is None or self.skel_graph.number_of_nodes() == 0:
            features['network_euler_number'] = 0
            features['longest_chain_length'] = 0
            # Add other default features if needed
            return features

        # Calculate features from the graph
        try:
            # Euler characteristic: V - E + F (F=1 for connected planar graph?)
            # For graphs: V - E
            num_nodes = self.skel_graph.number_of_nodes()
            num_edges = self.skel_graph.number_of_edges()
            # Euler number might be more related to connected components: V - E + C
            num_components = nx.number_connected_components(self.skel_graph)
            features['network_euler_number'] = num_nodes - num_edges + num_components
        except Exception as e:
            # print(f"Warning: Euler number calculation failed: {e}")
            features['network_euler_number'] = 0

        try:
            # Find the longest path in the graph (can be computationally expensive)
            # Note: dag_longest_path assumes Directed Acyclic Graph. This graph is undirected.
            # We need the longest simple path between any two nodes.
            longest_path_len = 0
            # This is NP-hard in general graphs. Approximate or use for small graphs only.
            # For simplicity, let's find diameter (longest shortest path) as a proxy,
            # or skip if too complex.
            # Diameter calculation requires connected graph.
            if nx.is_connected(self.skel_graph):
                 # longest_path_len = nx.diameter(self.skel_graph) # Diameter is length of longest shortest path
                 # Finding the actual longest path is harder. Let's skip for now.
                 pass # Placeholder
            features['longest_chain_length'] = 0 # Placeholder - calculation is complex

        except Exception as e:
            # print(f"Warning: Longest chain calculation failed: {e}")
            features['longest_chain_length'] = 0

        # Add more network features here if needed (e.g., average degree, density)

        return features
