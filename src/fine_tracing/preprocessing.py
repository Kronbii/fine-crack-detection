"""Pre-processing wrapper."""
from __future__ import annotations
import numpy as np
from . import edge


def main(image) -> np.ndarray:
    processed_img = edge.main(image)
    return processed_img


__all__ = ["main"]
