
import cv2
import numpy as np

def normalize_iris(iris_segment):
    """
    Normalize the segmented iris region to a fixed size.
    Args:
    - iris_segment (np.ndarray): Segmented iris region.
    Returns:
    - normalized_iris (np.ndarray): Normalized iris image.
    """
    fixed_size = (64, 512)
    normalized_iris = cv2.resize(iris_segment, fixed_size)
    return normalized_iris
