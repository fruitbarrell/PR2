from tqdm import tqdm
import cv2 as cv
import numpy as np
from score_functions import SAD,SSD,NCC


def region_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y,direction="L2R"):
    """
    Perform region-based stereo matching using block comparison methods over the entire image.

    Parameters:
        left_image: Left stereo image (BGR format)
        right_image: Right stereo image (BGR format)
        DISTANCE: Block comparison method ("SAD", "SSD", or "NCC")
        SEARCH_RANGE: Maximum disparity to search in x-direction
        TEMPLATE_SIZE_X: Width of the matching block
        TEMPLATE_SIZE_Y: Height of the matching block
        direction: Matching direction ("L2R" for Left-to-Right, "R2L" for Right-to-Left)

    Returns:
        Grayscale disparity map as a 2D numpy array (same height and width as input images)
    """
    # --- Type and value checks ---
    if not isinstance(left_image, np.ndarray):
        print("Error: 'left_image' must be a numpy array.")
        return None
    if not isinstance(right_image, np.ndarray):
        print("Error: 'right_image' must be a numpy array.")
        return None
    if DISTANCE not in ["SAD", "SSD", "NCC"]:
        print("Error: 'DISTANCE' must be one of ['SAD', 'SSD', 'NCC'].")
        return None
    if not isinstance(SEARCH_RANGE, int) or SEARCH_RANGE <= 0:
        print("Error: 'SEARCH_RANGE' must be a positive integer.")
        return None
    if not isinstance(TEMPLATE_SIZE_X, int) or TEMPLATE_SIZE_X <= 0 :
        print("Error: 'TEMPLATE_SIZE_X' must be an positive integer.")
        return None
    if not isinstance(TEMPLATE_SIZE_Y, int) or TEMPLATE_SIZE_Y <= 0:
        print("Error: 'TEMPLATE_SIZE_Y' must be an positive integer.")
        return None
    if direction not in ["L2R", "R2L"]:
        print("Error: 'direction' must be either 'L2R' or 'R2L'.")
        return None
    
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

