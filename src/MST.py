import numpy as np
import networkx as nx


# ================= MST-BASED ORDERING FUNCTIONS =================
def compute_euclidean_distance(p1, p2):
    """calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)  # linalg is used for better computational efficiency


def build_graph(points):
    """
    Build a weighted graph from a set of points. Each node represents a point (by its index) and each edge is weighted
    by the Euclidean distance between points.
    """
    n = len(points)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_euclidean_distance(points[i], points[j])
            G.add_edge(i, j, weight=dist)
    return G


def compute_mst(points):
    """Compute the minimum spanning tree (MST) for the given points"""
    G = build_graph(points)
    T = nx.minimum_spanning_tree(G, weight="weight")
    return T


def find_tree_diameter(T):
    """
    Find the longest path (diameter) of the tree T.
    This is done by:
      1. Picking an arbitrary leaf node.
      2. Running Dijkstra's algorithm from that node to find the farthest node.
      3. Running Dijkstra's algorithm from that farthest node to find the opposite node.
    The resulting path is taken as the natural ordering of the points.
    """
    # Identify all leaf nodes (nodes with degree 1)
    leaves = [node for node, degree in T.degree() if degree == 1]
    if not leaves:
        return list(T.nodes)
    start_leaf = leaves[0]
    dist_from_leaf, _ = nx.single_source_dijkstra(T, start_leaf, weight="weight")
    farthest_node = max(dist_from_leaf, key=dist_from_leaf.get)
    dist_from_far, paths_from_far = nx.single_source_dijkstra(
        T, farthest_node, weight="weight"
    )
    opposite_node = max(dist_from_far, key=dist_from_far.get)
    diameter_path = paths_from_far[opposite_node]
    return diameter_path


def sort_corners(points):
    """Sort a list of corner points based on MST-based ordering"""
    pts_array = np.array(points, dtype=float)
    T = compute_mst(pts_array)
    diameter_order = find_tree_diameter(T)
    sorted_points = pts_array[diameter_order]
    return sorted_points.astype(int).tolist()
