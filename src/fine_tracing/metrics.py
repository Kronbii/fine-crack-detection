"""Metrics computation module (adapted from legacy implementation)."""
from __future__ import annotations
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from . import config
from . import utils


def interpolate_contour_points(contour, y_values) -> np.ndarray:
	contour = contour.squeeze()
	x = contour[:, 0]
	y = contour[:, 1]
	sorted_indices = np.argsort(y)
	x = x[sorted_indices]
	y = y[sorted_indices]
	interpolated_x = np.interp(y_values, y, x)
	interpolated_points = np.column_stack((interpolated_x, y_values))
	return interpolated_points


def preprocess_image(image) -> np.ndarray:
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced_img = clahe.apply(image)
	_, binary_image = cv2.threshold(enhanced_img, 125, 255, cv2.THRESH_BINARY)
	return binary_image


def calculate_crack_accuracy(true_contour, gen_contour, distance_threshold=5) -> dict:
	min_y = max(np.min(true_contour[:, :, 1]), np.min(gen_contour[:, :, 1]))
	max_y = min(np.max(true_contour[:, :, 1]), np.max(gen_contour[:, :, 1]))
	y_values = np.arange(min_y, max_y + 1)
	true_points = interpolate_contour_points(true_contour, y_values)
	gen_points = interpolate_contour_points(gen_contour, y_values)
	distances = np.abs(true_points[:, 0] - gen_points[:, 0])
	true_tree = cKDTree(true_points)
	gen_tree = cKDTree(gen_points)
	distances_to_gen, _ = true_tree.query(gen_points, k=1)
	true_matched = np.sum(distances_to_gen <= distance_threshold)
	distances_to_true, _ = gen_tree.query(true_points, k=1)
	gen_matched = np.sum(distances_to_true <= distance_threshold)
	mean_distance = np.mean(distances)
	max_distance = np.max(distances)
	rms_error = np.sqrt(np.mean(np.square(distances)))
	precision = gen_matched / len(gen_points) if len(gen_points) > 0 else 0
	recall = true_matched / len(true_points) if len(true_points) > 0 else 0
	f1_score = (
		2 * (precision * recall) / (precision + recall)
		if (precision + recall) > 0
		else 0
	)
	return {
		"Precision": round(precision, 4),
		"Recall": round(recall, 4),
		"F1-score": round(f1_score, 4),
		"Mean-Distance": round(mean_distance, 4),
		"Max-Distance": round(max_distance, 4),
		"RMS-Error": round(rms_error, 4),
	}


def calculate_iou(true_mask, gen_mask) -> float:
	intersection = np.logical_and(true_mask, gen_mask)
	union = np.logical_or(true_mask, gen_mask)
	iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
	return iou_score


def pair_countours(true_contours, gen_contours) -> list:
	paired_cracks = []
	for gen_contour in gen_contours:
		min_distance = float("inf")
		best_match = None
		for true_contour in true_contours:
			gen_center = np.mean(gen_contour[:, 0, :], axis=0)
			true_center = np.mean(true_contour[:, 0, :], axis=0)
			distance = np.linalg.norm(gen_center - true_center)
			if distance < min_distance:
				min_distance = distance
				best_match = true_contour
		if best_match is not None:
			paired_cracks.append((best_match, gen_contour))
	return paired_cracks


def main() -> None:  # pragma: no cover (integration)
	ground_frames_dir = config.ground_frames_dir
	gen_frames_dir = config.gen_frames_dir
	metrics_output_dir = config.metrics_output_dir
	output_overlay_dir = config.output_overlay_dir
	utils.define_dir(metrics_output_dir, output_overlay_dir)
	output_json_file = config.output_json_file
	output_csv_file = config.output_csv_file
	final_metrics = {}
	for file in os.listdir(ground_frames_dir):
		true_frame_path = os.path.join(ground_frames_dir, file)
		frame_number = os.path.splitext(file)[0]
		gen_frame_path = os.path.join(gen_frames_dir, f"{frame_number}.{config.output_format}")
		output_overlayed_path = os.path.join(output_overlay_dir, f"{frame_number}.{config.output_format}")
		true_crack_img = cv2.imread(true_frame_path, cv2.IMREAD_GRAYSCALE)
		gen_crack_img = cv2.imread(gen_frame_path, cv2.IMREAD_GRAYSCALE)
		if true_crack_img is None or gen_crack_img is None:
			continue
		true_bin = preprocess_image(true_crack_img)
		gen_bin = preprocess_image(gen_crack_img)
		true_contours, _ = cv2.findContours(true_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		gen_contours, _ = cv2.findContours(gen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not true_contours or not gen_contours:
			continue
		paired_cracks = pair_countours(true_contours, gen_contours)
		it_metrics = {}
		for idx, (true_contour, gen_contour) in enumerate(paired_cracks):
			metrics_d = calculate_crack_accuracy(
				true_contour,
				gen_contour,
				distance_threshold=config.metrics_distance_threshold,
			)
			true_mask = np.zeros_like(true_bin)
			gen_mask = np.zeros_like(gen_bin)
			cv2.drawContours(true_mask, [true_contour], -1, 255, thickness=cv2.FILLED)
			cv2.drawContours(gen_mask, [gen_contour], -1, 255, thickness=cv2.FILLED)
			iou_score = round(calculate_iou(true_mask, gen_mask), 4)
			metrics_d["IoU"] = iou_score
			it_metrics[f"Crack {idx + 1}"] = metrics_d
		final_metrics[f"Frame {frame_number}"] = it_metrics
		utils.write_to_json(final_metrics, output_json_file)
		utils.write_to_csv(final_metrics, output_csv_file)
		overlay_ramy = overlay_cracks(true_bin, gen_bin)
		cv2.imwrite(output_overlayed_path, overlay_ramy)


def overlay_cracks(true_crack_img, gen_crack_img) -> np.ndarray:
	if true_crack_img.shape != gen_crack_img.shape:
		gen_crack_img = cv2.resize(
			gen_crack_img, (true_crack_img.shape[1], true_crack_img.shape[0])
		)
	overlay = np.zeros((true_crack_img.shape[0], true_crack_img.shape[1], 3), dtype=np.uint8)
	overlay[:, :, 2] = gen_crack_img
	overlay[:, :, 1] = true_crack_img
	for y_range in range(overlay.shape[0]):
		if y_range % 3 != 0:
			continue
		true_indices = np.where(overlay[y_range, :, 1] > 0)
		gen_indices = np.where(overlay[y_range, :, 2] > 0)
		if true_indices[0].size > 0 and gen_indices[0].size > 0:
			x_true = int(np.median(true_indices))
			x_gen = int(np.median(gen_indices))
			cv2.line(overlay, (x_true, y_range), (x_gen, y_range), (255, 255, 255), 1)
	return overlay


__all__ = [name for name in globals().keys() if not name.startswith("_")]

