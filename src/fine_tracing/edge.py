"""Edge detection helpers."""
from __future__ import annotations
import numpy as np
import cv2


def main(img):
    gray = img.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    epsilon = 1e-5
    blur = np.clip(blur, 1, 255)
    img_log = (np.log(blur) / (np.log(1 + np.max(blur) + epsilon))) * 255
    img_log = img_log.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img_log, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening, 22, 23)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closing = cv2.threshold(closing, 1, 255, cv2.THRESH_BINARY)[1]
    return closing


__all__ = ["main"]
