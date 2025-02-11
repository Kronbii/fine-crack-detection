import cv2
import numpy as np
import os
from scipy.interpolate import splprep, splev


def GaussianBlur(image):
    """Applies Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (3, 3), 0)


def MedianBlur(image):
    """Applies median blur to the image."""
    return cv2.medianBlur(image, 3)


def bilateralFilter(image):
    """Applies bilateral filter to the image."""
    return cv2.bilateralFilter(image, 9, 75, 75)


def opening(image):
    """Applies morphological opening (erosion + dilation)."""
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=1)


def histogram_equalization(image):
    """Applies histogram equalization to the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def convert_mask_to_binary(mask):
    """Convert the mask to a binary image."""
    if mask.shape[2] == 4:  # RGBA image
        mask = mask[:, :, 3]
    elif mask.shape[2] == 3:  # RGB image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask


def find_contours(mask):
    """Find the contours in the binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def convert_corners_to_coords(corners):
    """Convert the corners to a list of coordinates."""
    return [[int(x), int(y)] for x, y in corners.squeeze()]


def sort_corners(corners):
    """Sort the corners in a clockwise order."""
    return sorted(corners, key=lambda point: point[1])


def draw_corners_on_mask(corners, corner_mask):
    """Draw the corners on the mask."""
    corner_img = corner_mask.copy()
    for x, y in corners:
        cv2.circle(corner_img, (x, y), 2, (0, 255, 255), -1)
    return corner_img


def draw_lines(img, corners):
    """Draw the lines connecting the corners on the image."""
    image = img.copy()
    for i in range(len(corners) - 1):
        cv2.line(image, tuple(corners[i]), tuple(corners[i + 1]), (255, 255, 255), 1)
    return image


def draw_inerpolated_lines(img, xi, yi):
    """Draw the interpolated lines on the image."""
    image = img.copy()
    for i in range(len(xi) - 1):
        pt1 = (int(xi[i]), int(yi[i]))
        pt2 = (int(xi[i + 1]), int(yi[i + 1]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 1)
    return image


def interpolate_points(corners):
    """Interpolate the corners using a spline interpolation."""
    old_x = [x for x, y in corners]
    old_y = [y for x, y in corners]
    # Use a spline interpolation of order 5 with no smoothing
    tck, u = splprep([old_x, old_y], s=0, k=5)
    # unew = np.linspace(0, 1, 10000)
    xi, yi = splev(u, tck)
    return xi, yi


def transform_corners(corners, xmin, ymin):
    """Transform the corners from the cropped image coordinates to the original image coordinates."""
    return [[corner[0] + xmin, corner[1] + ymin] for corner in corners]


def define_dir(*directories):
    """Create directories if they do not exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def make_mask_transparent(mask):
    """Make the black pixels of a mask transparent"""
    h, w = mask.shape
    transparent_mask = np.zeros((h, w, 4), dtype=np.uint8)
    transparent_mask[:, :, :3] = 255
    transparent_mask[:, :, 3] = mask
    return transparent_mask


def overlay_mask_on_image(
    image, transparent_mask, overlay_color=(0, 0, 255), alpha_factor=0.1
):
    """Overlay a mask on an image with a given color and transparency."""
    _, _, _, alpha_mask = cv2.split(transparent_mask)
    alpha = (alpha_mask / 255.0) * alpha_factor
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[:] = overlay_color
    overlayed_image = cv2.convertScaleAbs(
        image * (1 - alpha[..., None]) + colored_mask * alpha[..., None]
    )
    return overlayed_image
