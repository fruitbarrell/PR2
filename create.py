from region import region_based
from feature import feature_based
from validition_functions import fill_disparity_gaps_adaptive,left_right_consistency_check
import cv2 as cv
import multiprocessing

if __name__ == '__main__':
    
    multiprocessing.freeze_support() 
    print("Starting")
    left_image = cv.imread('TEST_images\TESTA_L.jpg')
    right_image = cv.imread('TEST_images\TESTA_R.jpg')

    D_L = feature_based(left_image, right_image, "NCC", 64, 7, 7)
    print("Left to right done")
    D_R=feature_based(right_image, left_image , "NCC", 64, 7, 7)
    print("Right to Left done")

# Apply consistency check
    D_L_consistent = left_right_consistency_check(D_L,D_R)
    D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite("feature_output\SSD\size7x7.jpg",D_L_consistent)
    cv.imshow("Dmap",D_L_consistent)
    k= cv.waitKey()
    cv.destroyAllWindows()