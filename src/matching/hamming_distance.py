
import numpy as np

def hamming_distance(feature1, feature2):
    """
    Compute the Hamming distance between two feature vectors.
    Args:
    - feature1 (np.ndarray): First feature vector.
    - feature2 (np.ndarray): Second feature vector.
    Returns:
    - distance (float): Hamming distance.
    """
    return np.sum(feature1 != feature2) / len(feature1)
