import numpy as np

def SAD(left_block,right_block):
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    return np.sum(np.abs(left_mean - right_mean))

def SSD(left_block,right_block):
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    return np.sum((left_mean - right_mean) ** 2)

def NCC(left_block, right_block):
    left_mean = left_block - np.mean(left_block)
    right_mean = right_block - np.mean(right_block)

    numerator = np.sum(left_mean * right_mean)
    denominator = np.sqrt(np.sum(left_mean ** 2) * np.sum(right_mean ** 2))

    if denominator == 0:
        return 0  # avoid division by zero (happens with flat blocks)

    return numerator / denominator