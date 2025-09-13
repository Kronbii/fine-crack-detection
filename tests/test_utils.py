import numpy as np
from fine_tracing import utils


def test_euclidean_distance_zero():
    assert utils.calculate_euclidean_distance((0, 0), (0, 0)) == 0


def test_process_lines_length():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    corners = [(0, 0), (3, 4)]  # distance 5
    out, length = utils.process_lines(img, corners)
    assert length == 5
    assert out.shape == img.shape
