import cv2
# Load the .ppm image
img = cv2.imread('im1.ppm')

# Save it as .jpg
cv2.imwrite('TEST_images/VENUS_R.jpg', img)
