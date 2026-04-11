"""
Model Handler Module
=====================
Handles loading, initializing, and running predictions
with the MobileNetV2-based skin disease classification model.
"""

import os
import numpy as np

# Optimize TensorFlow for minimal memory/CPU overhead on constrained environments (Render)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow as tf

# Limit threads explicitly to save memory
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

from .disease_info import CLASS_NAMES, CLASS_LABELS, RISK_LEVELS


# Number of disease classes in HAM10000
NUM_CLASSES = 7

# Input image dimensions
IMG_SIZE = (224, 224, 3)

# Path to saved model weights
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_disease_model.weights.h5')

# Global model instance (loaded once, cached)
_model = None


def build_model():
    """
    Build the MobileNetV2-based classification model architecture.
    
    Architecture:
        MobileNetV2 (frozen base) → GlobalAveragePooling2D → 
        Dense(256, ReLU) → BatchNorm → Dropout(0.5) → 
        Dense(128, ReLU) → BatchNorm → Dropout(0.3) → 
        Dense(7, Softmax)
    
    Returns:
        tensorflow.keras.Model: Compiled model ready for prediction
    """
    # Load MobileNetV2 with ImageNet weights, without top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False
    
    # Build custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile with optimizer and loss
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_model():
    """
    Load the trained model from disk, or build a fresh model
    if no saved weights exist.
    
    The model is cached globally for subsequent predictions.
    
    Returns:
        tensorflow.keras.Model: The loaded/built model
    """
    global _model
    
    if _model is not None:
        return _model
    
    # Build the model architecture
    _model = build_model()
    
    # Load saved weights if available
    if os.path.exists(MODEL_PATH):
        try:
            _model.load_weights(MODEL_PATH)
            print(f"[OK] Model weights loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"[!] Could not load weights: {e}")
            print("[i] Using model with ImageNet base weights (untrained classifier)")
    else:
        print(f"[i] No saved model found at: {MODEL_PATH}")
        print("[i] Using model with ImageNet base weights (untrained classifier)")
        print("[i] Run train_model.py to train on HAM10000 dataset")
    
    return _model


def predict(img_array):
    """
    Run prediction on a preprocessed image.
    
    Args:
        img_array: numpy array of shape (1, 224, 224, 3), normalized to [0, 1]
        
    Returns:
        dict: Prediction results containing:
            - 'top_prediction': dict with class, label, confidence, risk_level, risk_color
            - 'top_3': list of top 3 predictions
            - 'all_probabilities': dict of all class probabilities
    """
    model = load_model()
    
    # Run inference
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]  # Shape: (7,)
    
    # Get indices sorted by confidence (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Build top-3 predictions
    top_3 = []
    for i, idx in enumerate(sorted_indices[:3]):
        class_code = CLASS_NAMES[idx]
        top_3.append({
            'rank': i + 1,
            'class_code': class_code,
            'label': CLASS_LABELS[class_code],
            'confidence': float(probabilities[idx]),
            'confidence_pct': round(float(probabilities[idx]) * 100, 2),
            'risk_level': RISK_LEVELS[class_code]
        })
    
    # Top prediction details
    top_class = CLASS_NAMES[sorted_indices[0]]
    
    # Determine risk color
    risk_colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
    risk_level = RISK_LEVELS[top_class]
    
    result = {
        'top_prediction': {
            'class_code': top_class,
            'label': CLASS_LABELS[top_class],
            'confidence': float(probabilities[sorted_indices[0]]),
            'confidence_pct': round(float(probabilities[sorted_indices[0]]) * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_colors.get(risk_level, '#f59e0b')
        },
        'top_3': top_3,
        'all_probabilities': {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2) 
            for i in range(NUM_CLASSES)
        }
    }
    
    return result


def get_model_for_gradcam():
    """
    Get the model instance and the name of the last convolutional layer
    for Grad-CAM computation.
    
    Returns:
        tuple: (model, last_conv_layer_name)
    """
    model = load_model()
    
    # Find the last convolutional layer in MobileNetV2
    # MobileNetV2's last conv layer is typically 'out_relu' or the last Conv2D
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
        # Also check for the activation after the last conv block
        if layer.name == 'out_relu':
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        # Fallback to a known MobileNetV2 layer
        last_conv_layer_name = 'out_relu'
    
    return model, last_conv_layer_name
