"""Primary processing pipeline (adapted from legacy fineTracing.main)."""
from __future__ import annotations
import logging
import os
import time
from typing import Dict

import cv2
import numpy as np

from . import config
from . import utils
from . import process_corners
from . import metrics as metrics_mod


def main() -> None:  # pragma: no cover (integration style)
	alg_process_start_time = time.process_time()
	alg_start_time = time.time()

	crack_length_file = config.crack_length_file

	logging.info("================== Starting Execution of Algorithm ===================")

	frames_dir = config.frames_dir
	masks_dir = config.masks_dir
	output_dir = config.frames_output_dir

	if not os.path.exists(frames_dir):
		logging.error("Frames directory %s not found", frames_dir)
		return
	if not os.path.exists(masks_dir):
		logging.error("Masks directory %s not found", masks_dir)
		return

	frames_total = len(os.listdir(masks_dir))
	logging.info("Number of frames to be processed: %d", frames_total)

	frame_crack_length: Dict[str, Dict[str, float]] = {}

	for mask_file in os.listdir(masks_dir):
		crack_length: Dict[str, float] = {}
		mask_path = os.path.join(masks_dir, mask_file)
		frame_number = os.path.splitext(mask_file)[0]
		image_path = os.path.join(frames_dir, f"{frame_number}.{config.input_format}")

		if not os.path.exists(mask_path):
			logging.warning("Mask %s not found", frame_number)
			continue
		if not os.path.exists(image_path):
			logging.warning("Image %s not found", frame_number)
			continue

		mask_lines_dir = os.path.join(output_dir, "mask_lines_out")
		lines_dir = os.path.join(output_dir, "lines_out")
		corners_dir = os.path.join(output_dir, "corners_out")
		pure_lines_out = os.path.join(output_dir, "pure-lines-out")
		result_dir = os.path.join(output_dir, "resultant-image")
		utils.define_dir(mask_lines_dir, lines_dir, corners_dir, pure_lines_out, result_dir)

		mask_lines_file = os.path.join(mask_lines_dir, f"{frame_number}.{config.output_format}")
		lines_file = os.path.join(lines_dir, f"{frame_number}.{config.output_format}")
		corners_file = os.path.join(corners_dir, f"{frame_number}.{config.output_format}")
		result_file = os.path.join(result_dir, f"{frame_number}.{config.output_format}")
		pure_lines_mask_file = os.path.join(pure_lines_out, f"{frame_number}.{config.output_format}")

		img = cv2.imread(image_path)
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
		mask = utils.convert_mask_to_binary(mask)
		transparent_mask = utils.make_mask_transparent(mask)

		final_img_lines = img.copy()
		result_img = img.copy()
		final_corner_mask = np.zeros_like(img, dtype=np.uint8)
		final_no_img_lines = np.zeros_like(gray_img, dtype=np.uint8)

		gray_img = utils.histogram_equalization(gray_img)

		logging.info("Processing Frame %s", frame_number)

		idx = 0
		contours = utils.find_contours(mask)
		for contour in contours:
			idx += 1
			(
				final_img_lines,
				final_no_img_lines,
				final_corner_mask,
				result_img,
				_crack_length,
			) = process_corners.main(
				contour,
				gray_img,
				mask,
				final_img_lines,
				final_corner_mask,
				final_no_img_lines,
				result_img,
			)
			crack_length[f"Crack-{idx}"] = _crack_length
		frame_crack_length[f"Frame-{frame_number}"] = crack_length

		mask_with_lines = utils.overlay_mask_on_image(final_img_lines, transparent_mask)
		result_img = utils.overlay_mask_on_image(result_img, transparent_mask)

		cv2.imwrite(mask_lines_file, mask_with_lines)
		cv2.imwrite(lines_file, final_img_lines)
		cv2.imwrite(corners_file, final_corner_mask)
		cv2.imwrite(
			pure_lines_mask_file, cv2.cvtColor(final_no_img_lines, cv2.COLOR_GRAY2BGR)
		)
		cv2.imwrite(result_file, result_img)

	utils.save_lengths(frame_crack_length, crack_length_file)

	alg_process_end_time = time.process_time()
	alg_end_time = time.time()
	logging.info(
		"Algorithm Process Time (seconds): %.3f",
		(alg_process_end_time - alg_process_start_time),
	)
	logging.info(
		"Algorithm Wall Time (seconds): %.3f", (alg_end_time - alg_start_time)
	)
	fps = 0.0
	if (alg_process_end_time - alg_process_start_time) > 0:
		fps = round(frames_total / (alg_process_end_time - alg_process_start_time), 3)
		logging.info("FPS = %s frames per second", fps)
		if fps:
			logging.info("Time per image ≈ %s seconds", round(1 / fps, 3))

	logging.info("================== Ending Execution of Algorithm ===================")
	logging.info("================== Starting Execution of Metrics ===================")

	metrics_process_start_time = time.process_time()
	metrics_start_time = time.time()
	metrics_mod.main()
	metrics_process_end_time = time.process_time()
	metrics_end_time = time.time()
	logging.info(
		"Metrics Process Time (seconds): %.3f",
		(metrics_process_end_time - metrics_process_start_time),
	)
	logging.info(
		"Metrics Wall Time (seconds): %.3f", (metrics_end_time - metrics_start_time)
	)
	logging.info("================== Ending Execution of Metrics ===================")


if __name__ == "__main__":  # pragma: no cover
	logging.basicConfig(level=logging.INFO)
	main()

