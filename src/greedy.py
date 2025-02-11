import numpy as np


def sort_corners(corners):
    """Sort the corners using a greedy-heuristic approach"""
    # Convert input (list of lists or array) to a NumPy array
    pts_array = np.array(corners, dtype=float)  # ensure float

    # 1. Find the index of the point with the smallest y-value
    start_index = np.argmin(pts_array[:, 1])

    # 2. Initialize visited set and path with the start index
    visited = set([start_index])
    path = [start_index]
    current = start_index

    # 3. Nearest-neighbor iteration
    while len(visited) < len(pts_array):
        # Compute distances from 'current' to all other points
        dists = np.linalg.norm(pts_array[current] - pts_array, axis=1)

        # Mark distances to visited points as infinite
        for v in visited:
            dists[v] = np.inf

        # Pick the closest unvisited point
        next_idx = np.argmin(dists)
        visited.add(next_idx)
        path.append(next_idx)
        current = next_idx

    # 4. Return the reordered points as a standard Python list
    #    If you prefer a NumPy array, you can just return pts_array[path]
    sorted_points = pts_array[path].astype(int).tolist()
    # print("sorted_points", sorted_points)
    return sorted_points
