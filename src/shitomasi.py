import cv2
from . import MST
from . import greedy
from . import utils
from . import config


def process_contour(
    contour, gray_img, mask, final_img_lines, final_corner_mask, empty_mask, result_img
):
    """Process each contour using shi-tomasi and return the final image with lines and corners"""
    x, y, w, h = cv2.boundingRect(contour)
    xmin, ymin, xmax, ymax = x, y, x + w, y + h

    cropped_img = gray_img[ymin:ymax, xmin:xmax]
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    processed_img = utils.opening(cropped_img)

    # Detect corners using Shi-Tomasi method
    raw_corners = cv2.goodFeaturesToTrack(
        processed_img,
        maxCorners=config.maxcorners,
        qualityLevel=config.minquality,
        minDistance=config.mindistance,
        mask=cropped_mask,
        blockSize=config.blocksize,
    )

    if raw_corners is None or len(raw_corners) <= config.interpolation_degree:
        return final_img_lines, empty_mask, final_corner_mask, result_img

    # Convert and adjust corner coordinates
    corners = utils.convert_corners_to_coords(raw_corners)
    corners = utils.transform_corners(corners, xmin, ymin)

    # Order corners using the specified method
    if config.sort_method == "MST":
        sorted_corners = MST.sort_corners(corners)
    elif config.sort_method == "greedy":
        sorted_corners = greedy.sort_corners(corners)
    elif config.sort_method == "classic":
        sorted_corners = utils.sort_corners(corners)
    else:
        sorted_corners = utils.sort_corners(corners)

    if config.tracing_method == "interpolation":
        # Interpolate points for smooth curve drawing
        xi, yi = utils.interpolate_points(sorted_corners)
        img_with_lines = utils.draw_inerpolated_lines(final_img_lines, xi, yi)
        result_img = utils.draw_inerpolated_lines(result_img, xi, yi)

    else:
        img_with_lines = utils.draw_lines(final_img_lines, sorted_corners)
        result_img = utils.draw_lines(result_img, sorted_corners)

    final_corner_mask = utils.draw_corners_on_mask(corners, final_corner_mask)
    result_img = utils.draw_corners_on_mask(corners, result_img)

    empty_mask_with_lines = utils.draw_lines(empty_mask, sorted_corners)

    return img_with_lines, empty_mask_with_lines, final_corner_mask, result_img
