import cv2
# Load the .ppm image
img = cv2.imread('scene1.row3.col2.ppm')

# Save it as .jpg
cv2.imwrite('scene1.row3.col2.jpg', img)