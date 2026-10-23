"""Image conversion helpers used by the GUI."""
from PIL import Image
import cv2


def bgr_to_pil(image_bgr):
    """Convert an OpenCV BGR image into a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def load_rgb_image(path):
    """Safely load an image as a detached RGB PIL image."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.convert("RGB").copy()
