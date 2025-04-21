from tqdm import tqdm
import cv2 as cv
import numpy as np
from score_functions import SAD,SSD,NCC
from validition_functions import fill_disparity_gaps_adaptive,left_right_consistency_check
import multiprocessing

def extract_patch(image, pt, TEMPLATE_SIZE_X, TEMPLATE_SIZE_Y):
    y, x = pt
    halfY=TEMPLATE_SIZE_Y//2
    halfX=TEMPLATE_SIZE_X//2

    if x-halfX< 0 or y-halfY < 0 or x+halfX >= image.shape[1] or y+halfY >= image.shape[0]:
        return None  # skip out-of-bounds
    return image[y-halfY:y+halfY+1, x-halfX:x+halfX+1]

def match_feature(args):
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
    if DISTANCE not in ["SAD","SSD","NCC"]:
        print("Incorrect input for DISTANCE")
        return
    h,w = left_image.shape[:2]
    Dmap= np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    halfY=TEMPLATE_SIZE_Y//2
    halfX=TEMPLATE_SIZE_X//2

    grayL=np.float32(cv.cvtColor(left_image,cv.COLOR_BGR2GRAY))
    grayR=np.float32(cv.cvtColor(right_image,cv.COLOR_BGR2GRAY))

    featuresL=cv.cornerHarris(grayL, blockSize=2, ksize=3, k=0.045)
    # featuresL=np.argwhere(featuresL> 0.00000001 * featuresL.max()) # This should return a coordinates array
    featuresL=np.argwhere(featuresL)

    featuresR=cv.cornerHarris(grayR, blockSize=2, ksize=3, k=0.045)
    # featuresR=np.argwhere(featuresR> 0.00000001 * featuresR.max()) # This should return a coordinates array
    featuresR=np.argwhere(featuresR)

    
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

    #Next two lines are to clear out the disparity map
    Dmap_inpainted = cv.inpaint(Dmap, 255 - mask, 3, cv.INPAINT_TELEA)
    Dmap_inpainted = cv.bilateralFilter(Dmap_inpainted,9,75,75)
    return Dmap




if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    left_image = cv.imread('TESTL.jpg')
    right_image = cv.imread('TESTR.jpg')

    D_L = feature_based(left_image, right_image, "NCC", 64, 7, 7)
    D_R = feature_based(right_image, left_image, "NCC", 64, 7, 7)

    # # Normalize if not already
    # D_L = cv.normalize(D_L, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # D_R = cv.normalize(D_R, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    cv.imshow("DLmap",D_L)
    cv.imshow("DRmap",D_R)
    k= cv.waitKey()
    cv.destroyAllWindows()

# Apply consistency check
    D_L_consistent = left_right_consistency_check(D_L, D_R)
    D_L_consistent = cv.normalize(D_L_consistent, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imwrite("SSDMap.jpg",Dmap)
    cv.imshow("Dmap",D_L_consistent)
    k= cv.waitKey()
    cv.destroyAllWindows()