import cv2
import numpy as np

# Read original mask as grayscale
img = cv2.imread("/home/kronbii/YUMA_batches/NRV0406_masks/NRV0406_N999_IM00334.png", 0)
anh = img.copy()
cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Overlay", 1200, 800)

# Create an empty skeleton mask
skel = np.zeros(img.shape, np.uint8)

# Threshold the image to binary
ret, img = cv2.threshold(img, 1, 255, 0)

# Find contours
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Skeletonize each contour independently
for cnt in contours:
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    done = False
    while not done:
        eroded = cv2.erode(mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()

        if cv2.countNonZero(mask) == 0:
            done = True

# Convert original mask to color
color_mask = cv2.cvtColor(anh, cv2.COLOR_GRAY2BGR)

# Apply morphological closing to the skeleton
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
ramy = skel.copy()
skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel, skel, iterations=3)
skel = cv2.dilate(skel, kernel, skel, iterations=1)
skel = cv2.erode(skel, kernel, skel, iterations=1)

cv2.imshow("Overlay", skel)
cv2.waitKey(0)


# Convert skeleton to red (on black background)
red_skel = cv2.merge((skel * 0, skel * 0, skel))  # BGR → Red channel active only
red_skel[:, :, 2] = skel  # Set the red channel directly

cv2.imshow("Overlay", red_skel)
cv2.waitKey(0)

# Overlay the skeleton on the original mask using transparency
overlay = cv2.addWeighted(color_mask, 0.7, red_skel, 1.0, 0)

# Show the result
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours in the skeletonized image
contours, _ = cv2.findContours(
    ramy.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"Number of contours found: {len(contours)}")

# Draw contours on a copy of the original image
contour_image = cv2.cvtColor(
    img, cv2.COLOR_GRAY2BGR
)  # Convert to BGR for color display
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# Display the image with contours
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contours", 1200, 800)
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
