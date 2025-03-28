import numpy as np
import cv2


def main(img):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()
    # Apply Gaussian blur to smooth the image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply logarithmic transform for better visualization
    epsilon = 1e-5  # To prevent divide by zero
    blur = np.clip(blur, 1, 255)  # Ensure no zero values
    img_log = (np.log(blur) / (np.log(1 + np.max(blur) + epsilon))) * 255
    img_log = img_log.astype(np.uint8)

    # Apply morphological opening (remove small noise)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img_log, cv2.MORPH_OPEN, kernel)

    # Compute median pixel intensity for adaptive thresholding
    median_intensity = np.median(opening)

    # Calculate lower and upper thresholds for Canny
    lower_threshold = int(max(0, 0.66 * median_intensity)) / 15
    upper_threshold = int(min(255, 1.33 * median_intensity)) / 15

    # Compute 10th and 90th percentiles for adaptive thresholds
    low_percentile = np.percentile(opening, 10) / 2
    high_percentile = np.percentile(opening, 90) / 5

    # Apply Canny edge detection using adaptive thresholds
    edges = cv2.Canny(opening, 22, 23)  # 50, 51

    # Morphological closing to fill gaps in detected edges
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closing = cv2.threshold(closing, 1, 255, cv2.THRESH_BINARY)[1]

    return closing


if __name__ == "__main__":
    main()
