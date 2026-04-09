"""
Image Preprocessing & Utility Functions
========================================
Handles image loading, resizing, normalization, and validation
for the skin disease detection pipeline.
"""

import numpy as np
from PIL import Image
import io
import base64


# Constants
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """
    Preprocess an image for model prediction.
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        tuple: (preprocessed_array, original_image)
            - preprocessed_array: numpy array shaped (1, 224, 224, 3), normalized to [0, 1]
            - original_image: PIL Image object (for Grad-CAM overlay)
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original for Grad-CAM
    original_image = image.copy()
    
    # Resize to model input size
    image_resized = image.resize(IMG_SIZE, Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_image


def image_to_base64(image):
    """
    Convert a PIL Image to a base64-encoded string for the frontend.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64-encoded PNG image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', quality=95)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def numpy_to_base64(img_array):
    """
    Convert a numpy array image to base64-encoded string.
    
    Args:
        img_array: numpy array representing an image
        
    Returns:
        str: Base64-encoded PNG image string
    """
    # Ensure values are in [0, 255] range
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    
    image = Image.fromarray(img_array)
    return image_to_base64(image)


def validate_image_size(image_bytes, max_size_mb=10):
    """
    Validate that the uploaded image is within size limits.
    
    Args:
        image_bytes: Raw image bytes
        max_size_mb: Maximum allowed file size in megabytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
    return True, None
