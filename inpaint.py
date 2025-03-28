import numpy as np
import cv2

img = cv2.imread("/home/kronbii/YUMA_batches/NRV0406/NRV0406_00312.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("/home/kronbii/YUMA_batches/NRV0406_masks/NRV0406_00312.png", cv2.IMREAD_GRAYSCALE)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
cv2.resizeWindow("dst", 1200, 800)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Subtract the inpainted image from the original image
difference = cv2.absdiff(img, dst)

# Display the difference
cv2.namedWindow("difference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("difference", 1200, 800)
cv2.imshow("difference", difference)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the average value of all the image pixels
avg_value = np.median(img)

# Subtract the image from the average value
subtracted_img = cv2.absdiff(img, avg_value)

# Display the subtracted image
cv2.namedWindow("subtracted_img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("subtracted_img", 1200, 800)
cv2.imshow("subtracted_img", subtracted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()