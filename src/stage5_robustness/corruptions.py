"""Image corruptions for robustness testing (placeholder implementations)."""
from PIL import Image, ImageFilter, ImageEnhance

def brightness(img, magnitude=0.2):
    return ImageEnhance.Brightness(img).enhance(1.0 - magnitude)

def gaussian_blur(img, radius=1.0):
    return img.filter(ImageFilter.GaussianBlur(radius))
