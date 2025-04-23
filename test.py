import cv2
# Load the .ppm image
img = cv2.imread('im4.ppm')

# Save it as .jpg
cv2.imwrite('TESTA_R.jpg', img)