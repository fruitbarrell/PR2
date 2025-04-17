from tqdm import tqdm
import cv2 as cv
import numpy as np
from score_functions import SAD,SSD,NCC

def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y):
    if DISTANCE not in ["SAD","SSD","NCC"]:
        print("Incorrect input for DISTANCE")
        return
    h,w = left_image.shape[:2]
    Dmap= np.zeros_like(left_image)
    halfY=TEMPLATE_SIZE_Y//2
    halfX=TEMPLATE_SIZE_X//2
    

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
                    x_right= x -d
                    if x_right - halfX < 0:
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

    return Dmap

left_image = cv.imread('TESTL.jpg')
right_image = cv.imread('TESTR.jpg')

Dmap=region_based(left_image, right_image, "SSD", 64, 7, 7)
Dmap=cv.normalize(Dmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite("SSDMap.jpg",Dmap)
cv.imshow("Dmap",Dmap)
k= cv.waitKey()
cv.destroyAllWindows()