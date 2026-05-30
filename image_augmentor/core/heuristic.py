import cv2
import numpy as np

def is_dark(image, threshold=100):
    """Check if image is dark (mean pixel value below threshold)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold

def is_blurry(image, threshold=100):
    """Check if image is blurry using variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def low_color_variance(image, threshold=500):
    """Check if image has low color variance (dull colors)."""
    return np.var(image) < threshold