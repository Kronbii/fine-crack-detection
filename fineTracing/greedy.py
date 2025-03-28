import numpy as np
from . import utils
from . import config


def filter_outlier(corners) -> list:
    last = corners[-1]
    blast = corners[-2]
    
    dist = utils.calculate_euclidean_distance(last, blast)
    if dist > config.greedy_dist_threshold and blast[0] > last[0]:
        corners = corners[:-1]
        
    return corners

def main(corners) -> list:
    """Sort the corners using a greedy-heuristic approach"""
    # Convert input (list of lists or array) to a NumPy array
    pts_array = np.array(corners, dtype=float)  # ensure float

    # Find the index of the point with the smallest y-value
    start_index = np.argmin(pts_array[:, 1])

    # Initialize visited set and path with the start index
    visited = set([start_index])
    path = [start_index]
    current = start_index

    # Nearest-neighbor iteration
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

    # Return the reordered points as a standard Python list
    sorted_points = pts_array[path].astype(int).tolist()
    # print("sorted_points", sorted_points)
    
    sorted_points = filter_outlier(sorted_points)
    
    return sorted_points


if __name__ == "__main__":
    main()
