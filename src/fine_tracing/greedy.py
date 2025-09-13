"""Greedy nearest-neighbor ordering."""
from __future__ import annotations
import numpy as np
from . import utils, config


def filter_outlier(corners) -> list:
    if len(corners) < 2:
        return corners
    last = corners[-1]
    blast = corners[-2]
    dist = utils.calculate_euclidean_distance(last, blast)
    if dist > config.greedy_dist_threshold and blast[0] > last[0]:
        corners = corners[:-1]
    return corners


def main(corners) -> list:
    pts_array = np.array(corners, dtype=float)
    start_index = np.argmin(pts_array[:, 1])
    visited = {start_index}
    path = [start_index]
    current = start_index
    while len(visited) < len(pts_array):
        dists = np.linalg.norm(pts_array[current] - pts_array, axis=1)
        for v in visited:
            dists[v] = np.inf
        next_idx = int(np.argmin(dists))
        visited.add(next_idx)
        path.append(next_idx)
        current = next_idx
    sorted_points = pts_array[path].astype(int).tolist()
    sorted_points = filter_outlier(sorted_points)
    return sorted_points


__all__ = ["main"]
