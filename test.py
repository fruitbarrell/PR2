import cv2 as cv

img=cv.imread("feature_output/SSD/size7x7.jpg")
blur=cv.medianBlur(img,9)
cv.imshow("blur",blur)
k= cv.waitKey()
cv.destroyAllWindows
