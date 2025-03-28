import cv2
import numpy as np
import os
import json
import csv
from scipy.interpolate import splprep, splev
from . import config


def GaussianBlur(image, kernel_size=3) -> np.ndarray:
    """Applies Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def MedianBlur(image, kernel_size=3) -> np.ndarray:
    """Applies median blur to the image."""
    return cv2.medianBlur(image, kernel_size)


def bilateralFilter(image) -> np.ndarray:
    """Applies bilateral filter to the image."""
    return cv2.bilateralFilter(image, 9, 75, 75)


def opening(image) -> np.ndarray:
    """Applies morphological opening (erosion + dilation)."""
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    return cv2.dilate(erosion, kernel, iterations=1)


def histogram_equalization(image) -> np.ndarray:
    """Applies histogram equalization to the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def convert_mask_to_binary(mask) -> np.ndarray:
    """Convert the mask to a binary image."""
    if mask.shape[2] == 4:  # RGBA image
        mask = mask[:, :, 3]
    elif mask.shape[2] == 3:  # RGB image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask


def find_contours(mask) -> list:
    """Find the contours in the binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def convert_corners_to_coords(corners) -> list:
    """Convert the corners to a list of coordinates."""
    return [[int(x), int(y)] for x, y in corners.squeeze()]


def sort_corners(corners) -> list:
    """Sort the corners in a clockwise order."""
    return sorted(corners, key=lambda point: point[1])


def draw_corners_on_mask(corners, corner_mask) -> np.ndarray:
    """Draw the corners on the mask."""
    corner_img = corner_mask.copy()
    for x, y in corners:
        cv2.circle(corner_img, (x, y), 2, (0, 255, 255), -1)
    return corner_img


def process_lines(img, corners) -> np.ndarray:
    """Draw the lines connecting the corners on the image and return crack length."""
    image = img.copy()
    crack_length = 0.0
    for i in range(len(corners) - 1):
        cv2.line(image, tuple(corners[i]), tuple(corners[i + 1]), (255, 255, 255), 1)
        crack_length += calculate_euclidean_distance(corners[i], corners[i + 1])
    return image, crack_length


def process_interpolated_lines(img, xi, yi) -> np.ndarray:
    """Draw the interpolated lines on the image and return crack length."""
    crack_length = 0.0
    image = img.copy()
    for i in range(len(xi) - 1):
        pt1 = (int(xi[i]), int(yi[i]))
        pt2 = (int(xi[i + 1]), int(yi[i + 1]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 1)
        crack_length += calculate_euclidean_distance(pt1, pt2)
    return image, crack_length


def interpolate_points(corners) -> tuple:
    """Interpolate the corners using a spline interpolation."""
    old_x = [x for x, y in corners]
    old_y = [y for x, y in corners]
    # Use a spline interpolation of order 5 with no smoothing
    tck, u = splprep([old_x, old_y], s=0, k=config.interpolation_degree)
    u_new = np.linspace(0, 1, 10000)  # New uniform parameterization
    xi, yi = splev(u_new, tck)
    return xi, yi


def transform_corners(corners, xmin, ymin) -> list:
    """Transform the corners from the cropped image coordinates to the original image coordinates."""
    return [[int(corner[0] + xmin), int(corner[1] + ymin)] for corner in corners]


def define_dir(*directories) -> None:
    """Create directories if they do not exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def make_mask_transparent(mask) -> np.ndarray:
    """Make the black pixels of a mask transparent"""
    h, w = mask.shape
    transparent_mask = np.zeros((h, w, 4), dtype=np.uint8)
    transparent_mask[:, :, :3] = 255
    transparent_mask[:, :, 3] = mask
    return transparent_mask


def overlay_mask_on_image(
    image, transparent_mask, overlay_color=(0, 0, 255), alpha_factor=0.1
) -> np.ndarray:
    """Overlay a mask on an image with a given color and transparency."""
    _, _, _, alpha_mask = cv2.split(transparent_mask)
    alpha = (alpha_mask / 255.0) * alpha_factor
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[:] = overlay_color
    overlayed_image = cv2.convertScaleAbs(
        image * (1 - alpha[..., None]) + colored_mask * alpha[..., None]
    )
    return overlayed_image


def calculate_euclidean_distance(p1, p2) -> float:
    """calculate Euclidean distance between two points"""
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1 - p2)  # linalg is used for better computational efficiency


def save_lengths(crack_lengths, filename):
    """Writes crack_lengths to a CSV file."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Frame",
                "Crack",
                "Crack Length (Pixels)",
            ]
        )

        for frame, crack in crack_lengths.items():
            for crack_name, crack_length in crack.items():
                writer.writerow([frame, crack_name, crack_length])


def orb_keypoints(image):
    """Detects ORB keypoints in the image."""
    # Initialize ORB feature detector
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints and compute descriptors
    keypoints, _ = orb.detectAndCompute(image, None)

    # Get the coordinates of the keypoints
    keypoint_coords = np.array([kp.pt for kp in keypoints])

    return keypoint_coords


def write_to_json(metrics, filename) -> None:
    """Writes metrics to a JSON file."""
    with open(filename, "w") as file:
        json.dump(metrics, file, indent=4)


def write_to_csv(metrics, filename) -> None:  # TODO: make scalable
    """Writes metrics to a CSV file."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Frame",
                "Crack",
                "Precision",
                "Recall",
                "F1-score",
                "IoU",
                "Mean-Distance",
                "Max-Distance",
                "RMS-Error",
            ]
        )

        for frame, cracks in metrics.items():
            for crack_id, metric in cracks.items():
                writer.writerow(
                    [
                        frame,
                        crack_id,
                        metric["Precision"],
                        metric["Recall"],
                        metric["F1-score"],
                        metric["IoU"],
                        metric["Mean-Distance"],
                        metric["Max-Distance"],
                        metric["RMS-Error"],
                    ]
                )
