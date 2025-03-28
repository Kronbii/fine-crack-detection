from skimage.morphology import skeletonize, thin
import cv2
import time
import matplotlib.pyplot as plt

image = cv2.imread(
    "/home/kronbii/YUMA_batches/NRV0406_masks/NRV0406_N7N999N34_IM00064.png", 0
)

# Measure runtime for skeletonize
start_time = time.time()
skeleton = skeletonize(image)
skeleton_time = time.time() - start_time

# Print runtimes
print(f"Runtime for skeletonize: {skeleton_time:.6f} seconds")

# Create a dictionary to map window names to images
windows = {
    "Original": image,
    "Skeleton": skeleton.astype("uint8") * 255,  # Convert boolean to uint8 for display
}

# Loop through the dictionary to overlay each processed image with the original and display
for name, img in windows.items():
    if name != "Original":  # Skip overlay for the original image
        overlay = cv2.addWeighted(
            image, 0.5, img, 0.5, 0
        )  # Blend original and processed images
        cv2.namedWindow(name + " Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name + " Overlay", 1200, 800)
        cv2.imshow(name + " Overlay", overlay)
    else:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1200, 800)
        cv2.imshow(name, img)
    cv2.waitKey(0)

# Find contours in the skeletonized image
contours, _ = cv2.findContours(
    skeleton.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(contours[0])

print(f"Number of contours found: {len(contours)}")

# Draw contours on a copy of the original image
contour_image = cv2.cvtColor(
    image, cv2.COLOR_GRAY2BGR
)  # Convert to BGR for color display
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# Display the image with contours
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contours", 1200, 800)
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
# Wait for a key press and close all windows
cv2.destroyAllWindows()

plt.show()
