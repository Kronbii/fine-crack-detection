import cv2
import numpy as np
import json
import os
import csv
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt
from . import config


def interpolate_contour_points(contour, y_values):
    """Interpolates contour points so that there is a point for every specified y-coordinate."""
    contour = contour.squeeze()  # Shape becomes (n, 2)
    x = contour[:, 0]
    y = contour[:, 1]

    sorted_indices = np.argsort(y)
    x = x[sorted_indices]
    y = y[sorted_indices]

    interpolated_x = np.interp(y_values, y, x)
    interpolated_points = np.column_stack((interpolated_x, y_values))

    return interpolated_points


def preprocess_image(image, invert=True):
    """Preprocess an image for contour extraction."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)

    _, binary_image = cv2.threshold(enhanced_img, 200, 255, cv2.THRESH_BINARY)
    return binary_image if not invert else cv2.bitwise_not(binary_image)


def calculate_crack_accuracy(true_contour, gen_contour, distance_threshold=5):
    """Calculate accuracy metrics for an individual crack."""
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
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score,
        "Mean Distance": mean_distance,
        "Max Distance": max_distance,
        "RMS Error": rms_error,
    }


def calculate_iou(true_mask, gen_mask):
    """Calculate IoU score for an individual crack."""
    intersection = np.logical_and(true_mask, gen_mask)
    union = np.logical_or(true_mask, gen_mask)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou_score


def draw_contours_with_labels(image, contours, color):
    """Draw contours on the image and label each crack."""
    img_with_labels = image.copy()

    for idx, contour in enumerate(contours):
        cv2.drawContours(img_with_labels, [contour], -1, color, 1)

        # Find the center of the crack
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(
                img_with_labels,
                f"Crack {idx + 1}",
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    return img_with_labels


def write_to_json(metrics, filename):
    """Writes metrics to a JSON file."""
    with open(filename, "w") as file:
        json.dump(metrics, file, indent=4)


def write_to_csv(metrics, filename):
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
                "Mean Distance",
                "Max Distance",
                "RMS Error",
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
                        metric["Mean Distance"],
                        metric["Max Distance"],
                        metric["RMS Error"],
                    ]
                )


def compare_avg_metrics(
    fileA,
    fileB,
    modelA_name="Model A",
    modelB_name="Model B",
    metrics=(
        "Precision",
        "Recall",
        "F1-score",
        "IoU",
        "Mean Distance",
        "Max Distance",
        "RMS Error",
    ),
):
    """
    Reads two CSV files containing metrics with columns like:
    Frame,Crack,Precision,Recall,F1-score,IoU,Mean Distance,Max Distance,RMS Error

    1) Averages each metric across all rows in each file.
    2) Plots each metric in a separate subplot with two bars: one for Model A, one for Model B.

    :param fileA: Path to CSV file for model A.
    :param fileB: Path to CSV file for model B.
    :param modelA_name: Label for model A (used in plots).
    :param modelB_name: Label for model B (used in plots).
    :param metrics: Tuple/list of metric column names to compare.
    """
    # 1. Read the CSVs
    dfA = pd.read_csv(fileA)
    dfB = pd.read_csv(fileB)

    # 2. Compute the average of each metric for each model
    #    Make sure all metrics exist in the columns; if not, you may need to handle missing columns
    meanA = dfA[list(metrics)].mean()
    meanB = dfB[list(metrics)].mean()

    # 3. Create subplots: one subplot per metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, num_metrics, figsize=(4 * num_metrics, 5), squeeze=False
    )

    for i, metric in enumerate(metrics):
        ax = axes[0, i]  # 1 row, i-th column

        # 4. Plot two bars: one for A, one for B
        x_vals = [0, 1]
        y_vals = [meanA[metric], meanB[metric]]
        ax.bar(x_vals, y_vals, color=["blue", "orange"], width=0.6)

        # 5. Labeling
        ax.set_title(metric)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([modelA_name, modelB_name])
        ax.set_ylabel("Value")

        # Add a bit of space above the max bar
        max_val = max(y_vals)
        ax.set_ylim([0, max_val * 1.2 if max_val > 0 else 1])

    plt.tight_layout()
    plt.show()


def plot_per_frame_metrics(
    fileA,
    fileB,
    modelA_name="Model A",
    modelB_name="Model B",
    metrics=(
        "Precision",
        "Recall",
        "F1-score",
        "IoU",
        "Mean Distance",
        "Max Distance",
        "RMS Error",
    ),
):
    """
    Reads two CSV files containing columns: [Frame, Crack, <metrics>].
    1) Groups by Frame (averaging across cracks) for each model.
    2) Merges the two DataFrames on 'Frame'.
    3) For each metric in `metrics`, creates a separate figure (window):
       - Plots a grouped bar chart with frames on the x-axis
         and raw metric values on the y-axis (no normalization).
    """

    # 1. Read and group by "Frame"
    dfA = pd.read_csv(fileA)
    dfB = pd.read_csv(fileB)

    # Average each metric by Frame for Model A and Model B
    dfA_grouped = dfA.groupby("Frame")[list(metrics)].mean().reset_index()
    dfB_grouped = dfB.groupby("Frame")[list(metrics)].mean().reset_index()

    # Rename columns to keep them distinct in the merged DataFrame
    # For example, Precision -> Precision_A, Precision_B
    dfA_grouped = dfA_grouped.rename(columns={m: f"{m}_A" for m in metrics})
    dfB_grouped = dfB_grouped.rename(columns={m: f"{m}_B" for m in metrics})

    # Merge on Frame
    df_merged = pd.merge(dfA_grouped, dfB_grouped, on="Frame", how="outer")
    df_merged = df_merged.sort_values("Frame")

    # List of frames for the x-axis
    frames = df_merged["Frame"].unique()

    # 2. Loop over each metric, creating a separate figure
    for m in metrics:
        colA = f"{m}_A"
        colB = f"{m}_B"

        # Skip if this metric wasn't found in one of the DataFrames
        if colA not in df_merged.columns or colB not in df_merged.columns:
            print(f"Warning: Metric '{m}' not found in one of the files. Skipping.")
            continue

        # Prepare data for plotting
        modelA_values = []
        modelB_values = []

        # We'll convert frames to a list so we can iterate consistently
        frame_list = list(frames)
        for frame in frame_list:
            row = df_merged[df_merged["Frame"] == frame]
            if row.empty:
                # If no data for this frame, use NaN
                modelA_values.append(np.nan)
                modelB_values.append(np.nan)
            else:
                valA = row[colA].values[0]
                valB = row[colB].values[0]
                modelA_values.append(valA)
                modelB_values.append(valB)

        # 3. Plot in a new figure for this metric (raw values)
        fig, ax = plt.subplots(figsize=(10, 6))

        x_positions = np.arange(len(frame_list))
        width = 0.4

        ax.bar(
            x_positions - width / 2,
            modelA_values,
            width,
            label=modelA_name,
            color="blue",
        )
        ax.bar(
            x_positions + width / 2,
            modelB_values,
            width,
            label=modelB_name,
            color="orange",
        )

        ax.set_title(f"{m} (Raw Values)")
        ax.set_xticks(x_positions)
        # Convert frames to strings if needed
        ax.set_xticklabels(frame_list, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.legend()

        # Adjust y-limits based on data
        valid_vals = [v for v in (modelA_values + modelB_values) if not np.isnan(v)]
        if valid_vals:
            y_min, y_max = min(valid_vals), max(valid_vals)
            # Add a bit of padding if max > 0
            if y_min == y_max:
                # If all values are identical, just show a small range around it
                ax.set_ylim([y_min - 1, y_min + 1])
            else:
                ax.set_ylim([y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1])

        plt.tight_layout()
        plt.show()


def main():
    # Directories
    ground_frames_dir = config.ground_frames_dir
    gen_frames_dir = config.gen_frames_dir
    output_json_file = config.output_json_file
    output_csv_file = config.output_csv_file

    final_metrics = {}

    for file in os.listdir(ground_frames_dir):
        true_frame_path = os.path.join(ground_frames_dir, file)
        frame_number = os.path.splitext(file)[0]
        gen_frame_path = os.path.join(gen_frames_dir, f"{frame_number}.ppm")

        # Load images
        true_crack_img = cv2.imread(true_frame_path, cv2.IMREAD_GRAYSCALE)
        gen_crack_img = cv2.imread(gen_frame_path, cv2.IMREAD_GRAYSCALE)

        if true_crack_img is None or gen_crack_img is None:
            print(f"Skipping frame {frame_number}: Image not found")
            continue

        # Preprocess images
        true_bin = preprocess_image(true_crack_img, invert=True)
        gen_bin = preprocess_image(gen_crack_img, invert=False)

        # Find contours
        true_contours, _ = cv2.findContours(
            true_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        gen_contours, _ = cv2.findContours(
            gen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        it_metrics = {}

        if not true_contours or not gen_contours:
            print(f"No valid cracks detected in Frame {frame_number}.")
        else:
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

            for idx, (true_contour, gen_contour) in enumerate(paired_cracks):
                metrics = calculate_crack_accuracy(
                    true_contour, gen_contour, distance_threshold=5
                )

                # Create binary masks for IoU calculation
                true_mask = np.zeros_like(true_bin)
                gen_mask = np.zeros_like(gen_bin)
                cv2.drawContours(
                    true_mask, [true_contour], -1, 255, thickness=cv2.FILLED
                )
                cv2.drawContours(gen_mask, [gen_contour], -1, 255, thickness=cv2.FILLED)
                iou_score = calculate_iou(true_mask, gen_mask)

                metrics["IoU"] = iou_score

                it_metrics[f"Crack {idx + 1}"] = metrics

        final_metrics[f"Frame {frame_number}"] = it_metrics

    # Save to JSON & CSV
    write_to_json(final_metrics, output_json_file)
    write_to_csv(final_metrics, output_csv_file)

    print("Processing complete! Metrics saved.")


if __name__ == "__main__":
    main()
