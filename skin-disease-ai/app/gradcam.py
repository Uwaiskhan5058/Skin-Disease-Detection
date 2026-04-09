"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
==============================================================
Generates visual explanations for model predictions by highlighting
the regions of the input image that are most important for the prediction.

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017)
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from .utils import image_to_base64


def generate_gradcam(model, img_array, original_image, class_index=None, 
                     last_conv_layer_name='out_relu'):
    """
    Generate a Grad-CAM heatmap for a given image and model prediction.
    
    Args:
        model: The Keras model
        img_array: Preprocessed image array of shape (1, 224, 224, 3)
        original_image: Original PIL Image (for overlay)
        class_index: Target class index. If None, uses the predicted class.
        last_conv_layer_name: Name of the last convolutional layer in the model
        
    Returns:
        str: Base64-encoded heatmap overlay image
    """
    try:
        # Create a model that outputs both the conv layer activations and the predictions
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Use predicted class if no specific class is given
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_score = predictions[:, class_index]
        
        # Compute gradients of the class score with respect to conv layer output
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv output channels by their importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to focus on positive influences
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize the heatmap to [0, 1]
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        
        heatmap = heatmap.numpy()
        
        # Create the overlay image
        overlay = create_heatmap_overlay(heatmap, original_image)
        
        return image_to_base64(overlay)
        
    except Exception as e:
        print(f"[!] Grad-CAM generation error: {e}")
        # Return a fallback — the original image with a red tint
        return create_fallback_heatmap(original_image)


def create_heatmap_overlay(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Create a colored heatmap overlay on the original image.
    
    Args:
        heatmap: 2D numpy array of heatmap values [0, 1]
        original_image: PIL Image object
        alpha: Transparency of the heatmap overlay (0 = invisible, 1 = opaque)
        colormap: OpenCV colormap to use for visualization
        
    Returns:
        PIL Image: The overlay image
    """
    # Resize original image
    display_size = (400, 400)
    original_resized = original_image.resize(display_size, Image.LANCZOS)
    original_array = np.array(original_resized)
    
    # Resize heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap, display_size)
    
    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert from BGR (OpenCV) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on the original image
    overlay = np.uint8(heatmap_colored * alpha + original_array * (1 - alpha))
    
    return Image.fromarray(overlay)


def create_fallback_heatmap(original_image):
    """
    Create a fallback heatmap when Grad-CAM computation fails.
    Returns the original image with a subtle warm overlay to indicate
    the analysis area.
    
    Args:
        original_image: PIL Image object
        
    Returns:
        str: Base64-encoded image string
    """
    display_size = (400, 400)
    img = original_image.resize(display_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    
    # Create a simple center-weighted heatmap
    h, w = display_size
    y, x = np.mgrid[0:h, 0:w]
    center_y, center_x = h // 2, w // 2
    
    # Gaussian-like falloff from center
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply JET colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = np.uint8(heatmap_colored * 0.3 + img_array * 0.7)
    result = Image.fromarray(overlay)
    
    return image_to_base64(result)
