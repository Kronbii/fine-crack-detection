import cv2
import numpy as np
from . import MST
from . import greedy
from . import utils
from . import config


def shitomasi(cropped_img, cropped_mask, xmin, ymin, offset) -> tuple:
    """Process each contour using shi-tomasi and return the final image with lines and corners"""
    """ returns [[x1, y1], [x2, y2], ...] """

    # Detect corners using Shi-Tomasi method
    raw_corners = cv2.goodFeaturesToTrack(
        cropped_img,
        maxCorners=config.maxcorners,
        qualityLevel=config.minquality - offset,
        minDistance=config.mindistance,
        mask=cropped_mask,
        blockSize=config.blocksize,
    )

    if raw_corners is None:
        return None
    if (len(raw_corners) - 1) < config.interpolation_degree:
        return None

    # Convert and adjust corner coordinates
    corners = utils.convert_corners_to_coords(raw_corners)
    corners = utils.transform_corners(corners, xmin, ymin)
    return corners


def orb(cropped_img, cropped_mask, xmin, ymin) -> tuple:
    """Process each contour using shi-tomasi and return the final image with lines and corners"""
    """ returns [[x1, y1], [x2, y2], ...] """

    if cropped_img.shape[0] < 32 or cropped_img.shape[1] < 32:
        cropped_img = cv2.resize(cropped_img, (32, 32))
        print("Resized image to 32x32")

    if cropped_img is None or cropped_img.size == 0:
        print("Error: cropped_img is empty or invalid")
        return None

    # Initialize ORB feature detector
    orb = cv2.ORB_create(nfeatures=config.orb_features)

    # Detect keypoints and compute descriptors
    keypoints, desc = orb.detectAndCompute(cropped_img, cropped_mask)

    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", 1200, 800)
    cv2.imshow("gray", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", 1200, 800)
    cv2.imshow("gray", cropped_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", 1200, 800)
    cv2.imshow("gray", cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not keypoints:
        return None
    # Get the coordinates of the keypoints
    raw_corners = np.array([kp.pt for kp in keypoints])

    if raw_corners is None:
        return None
    if (len(raw_corners) - 1) < config.interpolation_degree:
        return None

    # Convert and adjust corner coordinates
    corners = [[int(num) for num in sublist] for sublist in raw_corners]
    corners = utils.transform_corners(raw_corners, xmin, ymin)
    return corners


def main(
    contour, gray_img, mask, final_img_lines, final_corner_mask, empty_mask, result_img
) -> tuple:

    x, y, w, h = cv2.boundingRect(contour)
    xmin, ymin, xmax, ymax = x, y, x + w, y + h
    # Crop the image and mask
    cropped_img = gray_img[ymin:ymax, xmin:xmax]
    cropped_mask = mask[ymin:ymax, xmin:xmax]

    flag = True
    idx = 0

    # while idx <= 0.06:
    if config.detection_method == "orb":
        corners = orb(gray_img, mask, xmin, ymin)
    elif config.detection_method == "shitomasi":
        while idx <= 0.06:
            corners = shitomasi(cropped_img, cropped_mask, contour, xmin, ymin, idx)
            # Order corners using the specified method
            if config.sort_method == "MST":
                sorted_corners = MST.main(corners)
            elif config.sort_method == "greedy":
                sorted_corners = greedy.main(corners)
            elif config.sort_method == "classic":
                sorted_corners = utils.sort_corners(corners)
            else:
                sorted_corners = utils.sort_corners(corners)
            _ = utils.calculate_euclidean_distance(
                sorted_corners[0], sorted_corners[-1]
            )
            if (
                _ / len(sorted_corners) > config.corner_distance_threshold
                or corners is None
            ):
                idx += 0.01
                continue
            else:
                break

    if corners is None:
        crack_length = 0
        return (
            final_img_lines,
            empty_mask,
            final_corner_mask,
            result_img,
            crack_length,
        )

    # Order corners using the specified method
    if config.sort_method == "MST":
        sorted_corners = MST.main(corners)
    elif config.sort_method == "greedy":
        sorted_corners = greedy.main(corners)
    elif config.sort_method == "classic":
        sorted_corners = utils.sort_corners(corners)
    else:
        sorted_corners = utils.sort_corners(corners)

    if config.tracing_method == "int":
        # Interpolate points for smooth curve drawing
        xi, yi = utils.interpolate_points(sorted_corners)
        img_with_lines, crack_length = utils.process_interpolated_lines(
            final_img_lines, xi, yi
        )
        result_img, _ = utils.process_interpolated_lines(result_img, xi, yi)

    else:
        img_with_lines, crack_length = utils.process_lines(
            final_img_lines, sorted_corners
        )
        result_img, _ = utils.process_lines(result_img, sorted_corners)

    final_corner_mask = utils.draw_corners_on_mask(corners, final_corner_mask)
    result_img = utils.draw_corners_on_mask(corners, result_img)

    empty_mask_with_lines, _ = utils.process_lines(empty_mask, sorted_corners)
    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", 1200, 800)
    cv2.imshow("gray", img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (
        img_with_lines,
        empty_mask_with_lines,
        final_corner_mask,
        result_img,
        crack_length,
    )
