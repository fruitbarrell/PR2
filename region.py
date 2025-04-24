from tqdm import tqdm
import cv2 as cv
import numpy as np
from score_functions import SAD,SSD,NCC
from validition_functions import left_right_consistency_check

def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y,direction="L2R"):
    if DISTANCE not in ["SAD","SSD","NCC"]:
        print("Incorrect input for DISTANCE")
        return
    h,w = left_image.shape[:2]
    Dmap= np.zeros_like(left_image)
    halfY=TEMPLATE_SIZE_Y//2
    halfX=TEMPLATE_SIZE_X//2
    

    if direction == "R2L":
        left_image, right_image = right_image, left_image
    
    total_pixels = (h - 2*halfY) * (w - halfX - SEARCH_RANGE - halfX)
    with tqdm(total=total_pixels, desc="Block Matching Progress") as pbar:
        for y in range(halfY,h - halfY):
            for x in range(halfX + SEARCH_RANGE, w - halfX):
                

                left_block=left_image[y-halfY: y+halfY+1, x-halfX:x+halfX+1 ]
                
                best_offset = 0
                if DISTANCE == "SAD" or DISTANCE == "SSD":
                    best_score = float("inf")
                else:  # NCC
                    best_score = float("-inf")
                    

                for d in range(SEARCH_RANGE):
                    if direction == "L2R":
                        x_right = x - d
                    else:  # R2L
                        x_right = x + d
                    if x_right - halfX < 0 or x_right + halfX >= w:
                        continue

                    right_block=right_image[y-halfY: y+halfY+1, x_right-halfX:x_right+halfX+1]

                    if DISTANCE == "SAD":
                        score = SAD(left_block,right_block)
                        if score < best_score:
                            best_offset=d
                            best_score=score

                    elif DISTANCE == "SSD":
                        score = SSD(left_block,right_block)
                        if score < best_score:
                            best_offset=d
                            best_score=score

                    else:
                        score = NCC(left_block,right_block)
                        if score > best_score:
                            best_offset=d
                            best_score=score

                Dmap[y,x]=best_offset
                pbar.update(1)

    return cv.cvtColor(Dmap,cv.COLOR_BGR2GRAY)


# left_image = cv.imread('TEST_images\TESTA_L.jpg')
# right_image = cv.imread('TEST_images\TESTA_R.jpg')

# D_L = region_based(left_image, right_image, "NCC", 64, 7, 7)
# D_R = region_based(right_image, left_image, "NCC", 64, 7, 7)

#     # Normalize if not already
# # D_L = cv.normalize(D_L, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
# # D_R = cv.normalize(D_R, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# cv.imshow("DLmap",D_L)
# cv.imshow("DRmap",D_R)
# k= cv.waitKey()
# cv.destroyAllWindows()

# # Apply consistency check
# D_L_consistent = left_right_consistency_check(D_L, D_R)
# D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
# # cv.imwrite("SSDMap.jpg",Dmap)
# cv.imshow("Dmap",D_L_consistent)
# k= cv.waitKey()
# cv.destroyAllWindows()