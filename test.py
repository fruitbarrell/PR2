import cv2 as cv
from region import region_based
from validition_functions import left_right_consistency_check


Tsukuba_L=cv.imread("TEST_images/Tsukuba_L.jpg")
Tsukuba_R=cv.imread("TEST_images/Tsukuba_R.jpg")

Bull_L=cv.imread("TEST_images/bull_L.png")
Bull_R=cv.imread("TEST_images/bull_R.png")

Venus_L=cv.imread("TEST_images/VENUS_L.jpg")
Venus_R=cv.imread("TEST_images/VENUS_R.jpg")

images_L=[Tsukuba_L,Bull_L,Venus_L]
images_R=[Tsukuba_R,Bull_R,Venus_R]
names=["Tsukuba","Bull","Venus"]
DISTANCES=["SSD","SAD","NCC"]
TEMPLATE_SIZES=[3,7,11]
SEARCH_RANGE=64

for DISTANCE in DISTANCES:
    for left_image,right_image,TEMPLATE_SIZE,name in zip(images_L,images_R,TEMPLATE_SIZES,names):
        TEMPLATE_SIZE_X=TEMPLATE_SIZE
        TEMPLATE_SIZE_Y=TEMPLATE_SIZE
        D_L = region_based(left_image, right_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, direction="L2R")
        print("Left to right done")
        D_R=region_based(left_image,right_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y, direction="R2L")
        print("Right to Left done")
        print("Commencing averaging in the neighbourhood")
        D_L_consistent = left_right_consistency_check(D_L,D_R)
        D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        cv.imwrite(f"region_output/{DISTANCE}/{name}{TEMPLATE_SIZE_X}x{TEMPLATE_SIZE_Y}.jpg",D_L_consistent)


