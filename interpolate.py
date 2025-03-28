import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def Canny_detector(img, weak_th=None, strong_th=None):
    """
    Custom Canny-like edge detection.
    For demonstration, uses gradient magnitude (mag) with non-maximum suppression.
    """
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction via Gaussian blur
    img = cv2.GaussianBlur(img, (11, 11), 1.4)

    # Calculate gradients using Sobel
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Convert cartesian to polar coords (gradient magnitude and direction)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Set thresholds if not provided
    mag_max = np.max(mag)
    if not weak_th:
        weak_th = mag_max * 0.1
    if not strong_th:
        strong_th = mag_max * 0.5

    # Dimensions
    height, width = img.shape

    # Non-Maximum Suppression
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # Identify neighbor coords based on angle range
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Bounds check and compare magnitudes
            if 0 <= neighb_1_x < width and 0 <= neighb_1_y < height:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if 0 <= neighb_2_x < width and 0 <= neighb_2_y < height:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    # Double threshold
    ids = np.zeros_like(img)
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    return mag


# 1) Load original image
frame = cv2.imread("205.jpg")

canny_img = Canny_detector(frame)

# If you just want to see the Canny edges in an OpenCV window
cv2.namedWindow("Canny Result", cv2.WINDOW_NORMAL)
cv2.imshow("Canny Result", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
