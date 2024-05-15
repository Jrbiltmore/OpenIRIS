
import cv2
import numpy as np

def apply_gabor_filters(normalized_iris):
    """
    Apply Gabor filters to extract features from the normalized iris image.
    Args:
    - normalized_iris (np.ndarray): Normalized iris image.
    Returns:
    - gabor_features (np.ndarray): Extracted Gabor features.
    """
    gabor_kernels = []
    for theta in range(0, 360, 45):
        theta = np.deg2rad(theta)
        gabor_kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(gabor_kernel)
    
    gabor_features = np.zeros_like(normalized_iris, dtype=np.float32)
    for kernel in gabor_kernels:
        filtered_img = cv2.filter2D(normalized_iris, cv2.CV_8UC3, kernel)
        gabor_features = np.maximum(gabor_features, filtered_img)
    
    return gabor_features
