"""Corner processing and tracing (adapted)."""
from __future__ import annotations
import cv2
import numpy as np
from . import config
from . import utils
from . import MST
from . import greedy


def shitomasi(cropped_img, cropped_mask, xmin, ymin, offset):
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
	corners = utils.convert_corners_to_coords(raw_corners)
	corners = utils.transform_corners(corners, xmin, ymin)
	return corners


def orb(cropped_img, cropped_mask, xmin, ymin):
	if cropped_img.shape[0] < 32 or cropped_img.shape[1] < 32:
		cropped_img = cv2.resize(cropped_img, (32, 32))
	if cropped_img is None or cropped_img.size == 0:
		return None
	orb_detector = cv2.ORB_create(nfeatures=config.orb_features)
	keypoints, _desc = orb_detector.detectAndCompute(cropped_img, cropped_mask)
	if not keypoints:
		return None
	raw_corners = np.array([kp.pt for kp in keypoints])
	if raw_corners is None or (len(raw_corners) - 1) < config.interpolation_degree:
		return None
	corners = utils.transform_corners(raw_corners, xmin, ymin)
	return corners


def main(contour, gray_img, mask, final_img_lines, final_corner_mask, empty_mask, result_img):
	x, y, w, h = cv2.boundingRect(contour)
	xmin, ymin, xmax, ymax = x, y, x + w, y + h
	cropped_img = gray_img[ymin:ymax, xmin:xmax]
	cropped_mask = mask[ymin:ymax, xmin:xmax]
	corners = None
	if config.detection_method == "orb":
		corners = orb(cropped_img, cropped_mask, xmin, ymin)
	elif config.detection_method == "shitomasi":
		idx = 0.0
		while idx <= 0.06:
			corners = shitomasi(cropped_img, cropped_mask, xmin, ymin, idx)
			if corners is None:
				idx += 0.01
				continue
			# heuristic quality check
			dist = utils.calculate_euclidean_distance(corners[0], corners[-1]) if corners else 0
			if dist and (dist / max(len(corners), 1)) > config.corner_distance_threshold:
				idx += 0.01
				continue
			break
	if corners is None:
		return final_img_lines, empty_mask, final_corner_mask, result_img, 0.0
	if config.sort_method == "MST":
		sorted_corners = MST.main(corners)
	elif config.sort_method == "greedy":
		sorted_corners = greedy.main(corners)
	else:
		sorted_corners = utils.sort_corners(corners)
	if config.tracing_method == "int":
		xi, yi = utils.interpolate_points(sorted_corners)
		img_with_lines, crack_length = utils.process_interpolated_lines(final_img_lines, xi, yi)
		result_img, _ = utils.process_interpolated_lines(result_img, xi, yi)
	else:
		img_with_lines, crack_length = utils.process_lines(final_img_lines, sorted_corners)
		result_img, _ = utils.process_lines(result_img, sorted_corners)
	final_corner_mask = utils.draw_corners_on_mask(corners, final_corner_mask)
	result_img = utils.draw_corners_on_mask(corners, result_img)
	empty_mask_with_lines, _ = utils.process_lines(empty_mask, sorted_corners)
	return img_with_lines, empty_mask_with_lines, final_corner_mask, result_img, crack_length


__all__ = ["main"]

