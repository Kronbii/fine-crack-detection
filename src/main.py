import cv2
import numpy as np
import os
from . import config
from . import utils
from . import shitomasi
from . import metrics


def main():
    # ================= MAIN PROCESSING SCRIPT =================
    images_dir = config.frames_dir
    masks_dir = config.masks_dir
    output_dir = config.frames_output_dir

    for mask_number in os.listdir(masks_dir):
        # ----------------- Obtaining mask path and frame number -----------------
        mask_path = os.path.join(masks_dir, mask_number)
        mask_number = os.path.splitext(mask_number)[0]
        image_number = mask_number
        image_path = os.path.join(images_dir, f"{image_number}.ppm")

        if not os.path.exists(image_path):
            print(f"Image {image_number} not found.")
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
        mask_lines_file = os.path.join(mask_lines_dir, f"{image_number}.ppm")
        lines_file = os.path.join(lines_dir, f"{image_number}.ppm")
        corners_file = os.path.join(corners_dir, f"{image_number}.ppm")
        result_file = os.path.join(result_dir, f"{image_number}.ppm")
        pure_lines_mask_file = os.path.join(pure_lines_out, f"{image_number}.ppm")

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

        # ----------------- Find contours and process each contour -----------------
        contours = utils.find_contours(mask)
        for contour in contours:
            final_img_lines, final_no_img_lines, final_corner_mask, result_img = (
                shitomasi.process_contour(
                    contour,
                    gray_img,
                    mask,
                    final_img_lines,
                    final_corner_mask,
                    final_no_img_lines,
                    result_img,
                )
            )

        # ----------------- Save images -----------------
        mask_with_lines = utils.overlay_mask_on_image(
            final_img_lines, transparent_mask
        )  # overlay final image with mask
        result_img = utils.overlay_mask_on_image(result_img, transparent_mask)

        cv2.imwrite(mask_lines_file, mask_with_lines)
        cv2.imwrite(lines_file, final_img_lines)
        cv2.imwrite(corners_file, final_corner_mask)
        cv2.imwrite(
            pure_lines_mask_file, cv2.cvtColor(final_no_img_lines, cv2.COLOR_GRAY2BGR)
        )
        cv2.imwrite(result_file, result_img)

        print(f"Processed Frame {image_number}")
    print("[INFO] Done Processing")
    print("[INFO] Starting Perfomance Evaluation")
    metrics.main()


if __name__ == "__main__":
    main()
