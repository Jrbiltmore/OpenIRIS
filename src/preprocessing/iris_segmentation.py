
import cv2
import numpy as np

def segment_iris(image):
    """
    Segment the iris from the input image.
    Args:
    - image (np.ndarray): Input eye image.
    Returns:
    - iris_segment (np.ndarray): Segmented iris region.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use HoughCircles to detect the iris
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 8,
                               param1=100, param2=30, minRadius=30, maxRadius=75)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            iris_segment = image[y-r:y+r, x-r:x+r]
            return iris_segment
    return None
