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
    
    # --- Type and value checks ---
    if not isinstance(disparity_map, np.ndarray):
        print("Error: 'disparity_map' must be a numpy array.")
        return None
    if disparity_map.ndim != 2:
        print("Error: 'disparity_map' must be a 2D array.")
        return None
    if not isinstance(min_valid, int) or min_valid <= 0:
        print("Error: 'min_valid' must be a positive integer.")
        return None
    if not isinstance(max_window, int) or max_window % 2 == 0 or max_window <= 1:
        print("Error: 'max_window' must be an odd integer greater than 1.")
        return None
    
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
    """
    Perform left-right consistency check on disparity maps to remove mismatches.

    Parameters:
        D_L: Disparity map computed from Left to Right (2D numpy array)
        D_R: Disparity map computed from Right to Left (2D numpy array)
        threshold: Maximum allowed difference between corresponding disparities

    Returns:
        Refined disparity map (2D numpy array, 8-bit) with invalid matches removed and gaps filled
    """
    if not isinstance(D_L, np.ndarray) or not isinstance(D_R, np.ndarray):
        print("Error: 'D_L' and 'D_R' must both be numpy arrays.")
        return None
    if D_L.shape != D_R.shape:
        print("Error: 'D_L' and 'D_R' must have the same shape.")
        return None
    if D_L.ndim != 2 or D_R.ndim != 2:
        print("Error: Disparity maps must be 2D arrays.")
        return None
    if not isinstance(threshold, int) or threshold < 0:
        print("Error: 'threshold' must be a non-negative integer.")
        return None
    
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

