import cv2 as cv
import numpy as np

def fill_disparity_gaps_adaptive(disparity_map, min_valid=5, max_window=15):
    """
    Fill zero-value (gap) pixels in a disparity map by adaptive averaging.
    
    Parameters:
        disparity_map: 2D numpy array (grayscale disparity map)
        min_valid: Minimum number of non-zero neighbors required to interpolate
        max_window: Maximum neighborhood size allowed (must be odd)

    Returns:
        Filled disparity map (same shape as input)
    """
    h, w = disparity_map.shape
    filled_map = disparity_map.copy()

    for y in range(h):
        for x in range(w):
            if filled_map[y, x] == 0:
                window_size = 3
                valid_vals = []

                while window_size <= max_window:
                    half = window_size // 2
                    y1, y2 = max(0, y - half), min(h, y + half + 1)
                    x1, x2 = max(0, x - half), min(w, x + half + 1)

                    region = filled_map[y1:y2, x1:x2]
                    non_zero_vals = region[region != 0]

                    if len(non_zero_vals) >= min_valid:
                        filled_map[y, x] = int(np.mean(non_zero_vals))
                        break
                    else:
                        window_size += 2  # expand window size (odd only)
    return filled_map


def left_right_consistency_check(D_L, D_R, threshold=1):
    h, w = D_L.shape[:2]
    D_L = D_L.astype(np.int32)
    D_R = D_R.astype(np.int32)
    consistency_map = np.copy(D_L)

    for y in range(h):
        for x in range(w):
            d=int(D_L[y, x])
            x_r = x - d

            if 0 <= x_r < w:
                d_r = D_R[y, x_r]
                
                if abs(d - d_r) > threshold:
                    consistency_map[y, x] = 0  # invalid match
            else:
                consistency_map[y, x] = 0  # out of bounds
            
     # Convert to 8-bit for inpainting (required by OpenCV)
    consistency_map_8u = np.clip(consistency_map, 0, 255).astype(np.uint8)
    consistency_map = fill_disparity_gaps_adaptive(consistency_map_8u)
    

    return consistency_map

