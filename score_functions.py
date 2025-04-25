import numpy as np

def SAD(left_block,right_block):
    """
    Compute the Sum of Absolute Differences (SAD) between two blocks after mean normalization.

    Parameters:
        left_block: 2D numpy array (patch from the left image)
        right_block: 2D numpy array (patch from the right image)

    Returns:
        SAD score as a float (lower means better match)
    """
    if not isinstance(left_block, np.ndarray) or not isinstance(right_block, np.ndarray):
        print("Error: Both left_block and right_block must be numpy arrays.")
        return None
    if left_block.shape != right_block.shape:
        print("Error: Blocks must have the same shape.")
        return None
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    return np.sum(np.abs(left_mean - right_mean))

def SSD(left_block,right_block):
    """
    Compute the Sum of Squared Differences (SSD) between two blocks after mean normalization.

    Parameters:
        left_block: 2D numpy array (patch from the left image)
        right_block: 2D numpy array (patch from the right image)

    Returns:
        SSD score as a float (lower means better match)
    """
    if not isinstance(left_block, np.ndarray) or not isinstance(right_block, np.ndarray):
        print("Error: Both left_block and right_block must be numpy arrays.")
        return None
    if left_block.shape != right_block.shape:
        print("Error: Blocks must have the same shape.")
        return None
    
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    return np.sum((left_mean - right_mean) ** 2)

def NCC(left_block, right_block):
    """
    Compute the Normalized Cross-Correlation (NCC) between two blocks after mean normalization.

    Parameters:
        left_block: 2D numpy array (patch from the left image)
        right_block: 2D numpy array (patch from the right image)

    Returns:
        NCC score as a float between -1 and 1 (higher means better match)
    """
    if not isinstance(left_block, np.ndarray) or not isinstance(right_block, np.ndarray):
        print("Error: Both left_block and right_block must be numpy arrays.")
        return None
    if left_block.shape != right_block.shape:
        print("Error: Blocks must have the same shape.")
        return None
    
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    numerator = np.sum(left_mean * right_mean)
    denominator = np.sqrt(np.sum(left_mean ** 2) * np.sum(right_mean ** 2))

    if denominator == 0:
        return 0  # avoid division by zero (happens with flat blocks)

    return numerator / denominator