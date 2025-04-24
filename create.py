from region import region_based
from feature import feature_based
from validition_functions import fill_disparity_gaps_adaptive,left_right_consistency_check
import cv2 as cv
import multiprocessing

if __name__ == '__main__':
    
    multiprocessing.freeze_support() 
    
    left_image=None
    right_image=None
    DISTANCE=None
    SEARCH_RANGE=None

    print("Starting")
    print("Enter the number of the input of images to work with:")
    print("1. Tsukuba input")
    print("2. Sawtooth input")
    print("3. Venus input")

    choice=None
    while choice not in [1, 2, 3]:
        choice = int(input("Your choice (1-3): "))
        if (choice==1):
            left_image = cv.imread('TEST_images\Tsukuba_L.jpg')
            right_image = cv.imread('TEST_images\Tsukuba_R.jpg')
        elif (choice==2):
            left_image = cv.imread('TEST_images\sawtooth_L.jpg')
            right_image = cv.imread('TEST_images\sawtooth_R.jpg')
        elif (choice==3):
            left_image = cv.imread('TEST_images\VENUS_L.jpg')
            right_image = cv.imread('TEST_images\VENUS_R.jpg')
        else:
            print("Please input a number from 1-3")
    
    if left_image is None or right_image is None:
        print("Error: Could not load one or both images.")
    
    print("Enter the number of the matching score you want to use :")
    print("1. SAD")
    print("2. SSD")
    print("3. NCC")

    choice=None
    while choice not in [1, 2, 3]:
        choice = int(input("Your choice (1-3): "))
        if (choice==1):
            DISTANCE="SAD"
        elif (choice==2):
            DISTANCE="SSD"
        elif (choice==3):
            DISTANCE="NCC"
        else:
            print("Please input a number from 1-3")
    

    while True:
        try:
            SEARCH_RANGE = int(input("Enter the SEARCH RANGE "))
            if SEARCH_RANGE <= 0:
                print("Please enter a positive number.")
            else:
                break  # valid input
        except ValueError:
            print("Please enter a valid number.")

           
    while True:
        try:
            TEMPLATE_SIZE_X = int(input("Enter the size of the matching window in the X direction (Preferably an ODD number): "))
            if TEMPLATE_SIZE_X <= 0:
                print("Please enter a positive number.")
            else:
                break  # valid input
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            TEMPLATE_SIZE_Y = int(input("Enter the size of the matching window in the Y direction (Preferably an ODD number): "))
            if TEMPLATE_SIZE_Y <= 0:
                print("Please enter a positive number.")
            else:
                break  # valid input
        except ValueError:
            print("Please enter a valid number.")

    print("Enter the Desired matching method:")
    print("1. Feature-Based")
    print("2. Region-Based")
  

    choice=None
    while choice not in [1, 2]:
        choice = int(input("Your choice (1-2): "))
        if (choice==1):
            D_L = feature_based(left_image, right_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_X)
            print("Left to right done")
            D_R=feature_based(right_image, left_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_X)
            print("Right to Left done")
            print("Commencing consistency check")
            D_L_consistent = left_right_consistency_check(D_L,D_R)
            D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            cv.imwrite(f"feature_output\{DISTANCE}\size{TEMPLATE_SIZE_X}x{TEMPLATE_SIZE_Y}.jpg",D_L_consistent)
            cv.imshow("Dmap",D_L_consistent)
            k= cv.waitKey()
            cv.destroyAllWindows()

        elif (choice==2):
            D_L = region_based(left_image, right_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_X, direction="L2R")
            print("Left to right done")
            D_R=region_based(left_image,right_image , DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_X, direction="R2L")
            print("Right to Left done")
            cv.imshow("DmapL",D_L)
            cv.imshow("DmapR",D_R)
            k= cv.waitKey()
            cv.destroyAllWindows()
            print("Commencing averaging in the neighbourhood")
            D_L_consistent = left_right_consistency_check(D_L,D_R)
            D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            cv.imwrite(f"region_output\{DISTANCE}\size{TEMPLATE_SIZE_X}x{TEMPLATE_SIZE_Y}.jpg",D_L_consistent)
            cv.imshow("Dmap",D_L_consistent)
            k= cv.waitKey()
            cv.destroyAllWindows()
        else:
            print("Please input a number from 1-2")
    
    

    


    