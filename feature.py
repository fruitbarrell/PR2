from tqdm import tqdm
import cv2 as cv
import numpy as np
from score_functions import SAD,SSD,NCC
import multiprocessing

def extract_patch(image, pt, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y):
    """
    Extract a rectangular patch centered at a given point from the input image.

    Parameters:
        image: 2D numpy array (grayscale image)
        pt: Tuple (y, x) representing the center of the patch
        TEMPLATE_SIZE_X: Width of the patch
        TEMPLATE_SIZE_Y: Height of the patch

    Returns:
        Patch as a 2D numpy array, or None if the patch extends beyond image boundaries
    """
     # --- Type and value checks ---
    if not isinstance(image, np.ndarray):
        print("Error: 'image' must be a numpy array.")
        return None
    if image.ndim != 2:
        print("Error: 'image' must be a 2D (grayscale) array.")
        return None
    if not (isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(i, int) for i in pt)):
        print("Error: 'pt' must be a tuple of two integers (y, x).")
        return None
    if not (isinstance(TEMPLATE_SIZE_X, int)  == 1 and TEMPLATE_SIZE_X > 1):
        print("Error: 'TEMPLATE_SIZE_X' must be an  integer greater than 1.")
        return None
    if not (isinstance(TEMPLATE_SIZE_Y, int)  and TEMPLATE_SIZE_Y > 1):
        print("Error: 'TEMPLATE_SIZE_Y' must be an  integer greater than 1.")
        return None
    
    y, x = pt
    halfY=TEMPLATE_SIZE_Y//2
    halfX=TEMPLATE_SIZE_X//2

    if x-halfX< 0 or y-halfY < 0 or x+halfX >= image.shape[1] or y+halfY >= image.shape[0]:
        return None  # skip out-of-bounds
    return image[y-halfY:y+halfY+1, x-halfX:x+halfX+1]

def match_feature(args):
        """
        WARNING: THIS FUNCTION IS NOT MEANT TO BE USED OUTSIDE THE FEATURE BASED FUNCTION BELOW
        
        Match a feature point from the left image with the best corresponding point in the right image 
        based on a block similarity metric.

        Parameters:
            args: Tuple containing:
                - pt1: Feature point in left image
                - grayL: Grayscale left image
                - grayR: Grayscale right image
                - featuresR: List of feature points in right image
                - DISTANCE: Matching method ("SAD", "SSD", or "NCC")
                - SEARCH_RANGE: Maximum horizontal distance to search
                - TEMPLATE_SIZE_X: Width of the matching patch
                - TEMPLATE_SIZE_Y: Height of the matching patch

        Returns:
            Tuple (pt1, best_offset): pt1 is the feature point in the left image, 
            best_offset is the best match offset in the x-direction (disparity), or (None, None) if no match found
        """
        pt1, grayL, grayR, featuresR, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y = args
        left_block = extract_patch(grayL, pt1, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y)
        if left_block is None:
            return None, None

        best_score = float("inf") if DISTANCE in ["SAD", "SSD"] else float("-inf")
        best_offset = None

        for pt2 in featuresR:
            if pt1[0] != pt2[0] or abs(pt1[1] - pt2[1]) > SEARCH_RANGE:
                continue
            right_block = extract_patch(grayR, pt2, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y)
            if right_block is None:
                continue

            if DISTANCE == "SAD":
                score = SAD(left_block, right_block)
                if score < best_score:
                    best_score = score
                    best_offset = abs(pt1[1] - pt2[1])
            elif DISTANCE == "SSD":
                score = SSD(left_block, right_block)
                if score < best_score:
                    best_score = score
                    best_offset = abs(pt1[1] - pt2[1])
            else:  # NCC
                score = NCC(left_block, right_block)
                if score > best_score:
                    best_score = score
                    best_offset = abs(pt1[1] - pt2[1])

        return pt1, best_offset


def feature_based(left_image, right_image, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y):
    """
    Perform feature-based stereo matching using Harris corners and block matching.

    Parameters:
        left_image: Left stereo image (BGR format)
        right_image: Right stereo image (BGR format)
        DISTANCE: Block comparison method ("SAD", "SSD", or "NCC")
        SEARCH_RANGE: Maximum search range in x-direction (disparity range)
        TEMPLATE_SIZE_X: Width of the matching block
        TEMPLATE_SIZE_Y: Height of the matching block

    Returns:
        Disparity map (2D numpy array of same width and height as input images)
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
    if DISTANCE not in ["SAD","SSD","NCC"]:
        print("Incorrect input for DISTANCE")
        return
    h,w = left_image.shape[:2]
    Dmap= np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
  

    grayL=np.float32(cv.cvtColor(left_image,cv.COLOR_BGR2GRAY))
    grayR=np.float32(cv.cvtColor(right_image,cv.COLOR_BGR2GRAY))

    featuresL=cv.cornerHarris(grayL, blockSize=2, ksize=3, k=0.045)
    featuresL=np.argwhere(featuresL> 0.0000000001 * featuresL.max()) # This should return a coordinates array
    # featuresL=np.argwhere(featuresL)

    featuresR=cv.cornerHarris(grayR, blockSize=2, ksize=3, k=0.045)
    featuresR=np.argwhere(featuresR> 0.0000000001 * featuresR.max()) # This should return a coordinates array
    # featuresR=np.argwhere(featuresR)

    
    args_list = [(pt1, grayL, grayR, featuresR, DISTANCE, SEARCH_RANGE, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y) for pt1 in featuresL]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(match_feature, args_list), total=len(args_list),leave=False))

    for pt1, best_offset in results:
        if pt1 is None or best_offset is None:
            continue
        y, x = pt1
        value = int(best_offset * 255 / SEARCH_RANGE)
        Dmap[y, x] = min(255, value)
        mask[y, x] = 255

    return Dmap




