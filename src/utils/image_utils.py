
import cv2

def load_image(image_path):
    """
    Load an image from the given path.
    Args:
    - image_path (str): Path to the image file.
    Returns:
    - image (np.ndarray): Loaded image.
    """
    return cv2.imread(image_path)

def save_image(image, save_path):
    """
    Save an image to the given path.
    Args:
    - image (np.ndarray): Image to save.
    - save_path (str): Path to save the image.
    """
    cv2.imwrite(save_path, image)
