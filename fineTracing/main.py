import cv2
import numpy as np
import os
import logging
import time
from . import config
from . import utils
from . import process_corners
from . import metrics
from . import preprocessing


def main() -> None:
    # Time variables for execution time
    alg_process_start_time = time.process_time()
    alg_start_time = time.time()

    crack_length_file = config.crack_length_file

    # ================= MAIN PROCESSING SCRIPT =================
    logging.basicConfig(level=logging.INFO, filename="logging.log", filemode="w")
    logging.info(
        "================== Starting Execution of Algorithm ==================="
    )

    frames_dir = config.frames_dir
    masks_dir = config.masks_dir
    output_dir = config.frames_output_dir

    if not os.path.exists(frames_dir):
        print(f"Frames directory {frames_dir} not found.")
        logging.error(f"Frames directory {frames_dir} not found.")
        return
    if not os.path.exists(masks_dir):
        print(f"Masks directory {masks_dir} not found.")
        logging.error(f"Masks directory {masks_dir} not found.")
        return

    frames_total = len(os.listdir(masks_dir))
    logging.info(f"Number of frames to be processed: {frames_total}")

    frame_crack_length = dict()

    for mask_file in os.listdir(masks_dir):
        crack_length = dict()
        # ----------------- Obtaining mask path and frame number -----------------
        mask_path = os.path.join(masks_dir, mask_file)
        frame_number = os.path.splitext(mask_file)[0]
        image_path = os.path.join(frames_dir, f"{frame_number}.{config.input_format}")

        if not os.path.exists(mask_path):
            print(f"Mask {frame_number} not found.")
            logging.warning(f"Mask {frame_number} not found.")
            continue
        if not os.path.exists(image_path):
            print(f"Image {frame_number} not found.")
            logging.warning(f"Image {frame_number} not found.")
            continue

        # ----------------- Creating output directories -----------------
        mask_lines_dir = os.path.join(output_dir, "mask_lines_out")
        lines_dir = os.path.join(output_dir, "lines_out")
        corners_dir = os.path.join(output_dir, "corners_out")
        pure_lines_out = os.path.join(output_dir, "pure-lines-out")
        result_dir = os.path.join(output_dir, "resultant-image")
        utils.define_dir(
            mask_lines_dir, lines_dir, corners_dir, pure_lines_out, result_dir
        )

        # ----------------- Naming output files -----------------
        mask_lines_file = os.path.join(
            mask_lines_dir, f"{frame_number}.{config.output_format}"
        )
        lines_file = os.path.join(lines_dir, f"{frame_number}.{config.output_format}")
        corners_file = os.path.join(
            corners_dir, f"{frame_number}.{config.output_format}"
        )
        result_file = os.path.join(result_dir, f"{frame_number}.{config.output_format}")
        pure_lines_mask_file = os.path.join(
            pure_lines_out, f"{frame_number}.{config.output_format}"
        )

        # ----------------- Reading image and mask -----------------
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # read image in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = utils.convert_mask_to_binary(mask)  # convert mask to binary
        transparent_mask = utils.make_mask_transparent(
            mask
        )  # convert black pixels of the mask to transparent pixels

        # ----------------- Variables to store final images to be saved -----------------
        final_img_lines = img.copy()
        result_img = img.copy()
        final_corner_mask = np.zeros_like(img, dtype=np.uint8)
        final_no_img_lines = np.zeros_like(gray_img, dtype=np.uint8)

        # ----------------- Pre-Processing Image -----------------
        gray_img = preprocessing.main(gray_img)

        logging.info(f"Processing Frame {frame_number}")
        print(f"Processing Frame {frame_number}")

        # ----------------- Find contours and process each contour -----------------
        idx = 0
        contours = utils.find_contours(mask)
        for contour in contours:
            idx = idx + 1
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

        # ----------------- Save images -----------------
        mask_with_lines = utils.overlay_mask_on_image(
            final_img_lines, transparent_mask
        )  # overlay final image with mask
        result_img = utils.overlay_mask_on_image(result_img, transparent_mask)

        cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("gray", 1200, 800)
        cv2.imshow("gray", result_img)
        cv2.waitKey(0)

        cv2.imwrite(mask_lines_file, mask_with_lines)
        cv2.imwrite(lines_file, final_img_lines)
        cv2.imwrite(corners_file, final_corner_mask)
        cv2.imwrite(
            pure_lines_mask_file, cv2.cvtColor(final_no_img_lines, cv2.COLOR_GRAY2BGR)
        )
        cv2.imwrite(result_file, result_img)

    # ----------------- Save crack lengths -----------------
    utils.save_lengths(frame_crack_length, crack_length_file)

    print("[INFO] Done Processing")
    print("[INFO] Starting Perfomance Evaluation")
    alg_process_end_time = time.process_time()
    alg_end_time = time.time()
    logging.info(
        f"Algorithm Process Time (seconds): {round(alg_process_end_time - alg_process_start_time, 3)}"
    )
    logging.info(
        f"Algorithm Wall Time (seconds): {round(alg_end_time - alg_start_time, 3)}"
    )

    fps = round(frames_total / (alg_process_end_time - alg_process_start_time), 3)
    logging.info(f"FPS = {fps} frames per second")
    logging.info(f"Time per image = {round(1/fps)} seconds")

    logging.info("================== Ending Execution of Algorithm ===================")
    logging.info("================== Starting Execution of Metrics ===================")
    metrics_process_start_time = time.process_time()
    metrics_start_time = time.time()

    metrics.main()

    metrics_process_end_time = time.process_time()
    metrics_end_time = time.time()
    logging.info(
        f"Metrics Process Time (seconds): {round(metrics_process_end_time - metrics_process_start_time, 3)}"
    )
    logging.info(
        f"Metrics Wall Time (seconds): {round(metrics_end_time - metrics_start_time, 3)}"
    )
    logging.info("================== Ending Execution of Metrics ===================")


if __name__ == "__main__":
    main()
