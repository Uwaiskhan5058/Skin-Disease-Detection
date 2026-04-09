"""
Model Training Script — HAM10000 Skin Disease Classification
==============================================================
Trains a MobileNetV2-based model on the HAM10000 dataset for
multi-class skin disease classification.

Usage:
    python train_model.py --data_dir ./data/HAM10000

Dataset Setup:
    1. Download HAM10000 from Kaggle:
       https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    2. Extract the dataset into ./data/HAM10000/
       Expected structure:
       data/HAM10000/
       ├── HAM10000_images_part_1/
       ├── HAM10000_images_part_2/
       └── HAM10000_metadata.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image


# ─── Configuration ──────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 7
LEARNING_RATE = 0.001

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_LABELS = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train skin disease classification model')
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\uwais\Downloads\archive',
                        help='Path to HAM10000 dataset directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained model')
    return parser.parse_args()


def load_dataset(data_dir):
    """
    Load the HAM10000 dataset from directory.
    
    Args:
        data_dir: Path to the HAM10000 dataset directory
        
    Returns:
        DataFrame with image paths and labels
    """
    print("\n[1/6] Loading dataset...")
    
    # Read metadata CSV
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}\n"
            "Please ensure the HAM10000 dataset is properly extracted."
        )
    
    df = pd.read_csv(metadata_path)
    print(f"  → Total samples in metadata: {len(df)}")
    
    # Build image path mapping
    # Images may be in part_1 and part_2 directories
    image_paths = {}
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2', 
                    'ham10000_images_part_1', 'ham10000_images_part_2',
                    'HAM10000_images']:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_id = os.path.splitext(img_file)[0]
                    image_paths[image_id] = os.path.join(folder_path, img_file)
    
    # Also check if images are directly in data_dir
    for img_file in os.listdir(data_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = os.path.splitext(img_file)[0]
            if image_id not in image_paths:
                image_paths[image_id] = os.path.join(data_dir, img_file)
    
    print(f"  → Found {len(image_paths)} images on disk")
    
    # Map image paths to DataFrame
    df['image_path'] = df['image_id'].map(image_paths)
    
    # Remove rows without corresponding images
    before = len(df)
    df = df.dropna(subset=['image_path'])
    if len(df) < before:
        print(f"  → Dropped {before - len(df)} entries without matching images")
    
    # Encode labels
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df['label'] = df['dx'].map(label_map)
    
    print(f"  → Final dataset size: {len(df)} images")
    print(f"\n  Class Distribution:")
    for cls, count in df['dx'].value_counts().items():
        print(f"    {CLASS_LABELS.get(cls, cls):30s} ({cls}): {count:5d}")
    
    return df


def load_and_preprocess_images(df, img_size=IMG_SIZE):
    """
    Load images and preprocess them for training.
    
    Args:
        df: DataFrame with 'image_path' and 'label' columns
        img_size: Target image size (height, width)
        
    Returns:
        tuple: (images_array, labels_array)
    """
    print("\n[2/6] Loading and preprocessing images...")
    
    images = []
    labels = []
    errors = 0
    
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  → Processing image {idx + 1}/{total}...")
        
        try:
            img = Image.open(row['image_path'])
            img = img.convert('RGB')
            img = img.resize(img_size, Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(row['label'])
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [!] Error loading {row['image_path']}: {e}")
    
    if errors > 0:
        print(f"  → Skipped {errors} images due to errors")
    
    X = np.array(images)
    y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    
    print(f"  → Images shape: {X.shape}")
    print(f"  → Labels shape: {y.shape}")
    
    return X, y


def build_model():
    """
    Build the MobileNetV2 transfer learning model.
    
    Returns:
        Compiled Keras model
    """
    print("\n[3/6] Building model architecture...")
    
    # Load MobileNetV2 with ImageNet weights
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Print model summary
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  → Total parameters: {trainable + non_trainable:,}")
    print(f"  → Trainable parameters: {trainable:,}")
    print(f"  → Non-trainable parameters: {non_trainable:,}")
    
    return model


def create_data_augmentation():
    """Create data augmentation generator for training."""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )


def train_model(model, X_train, y_train, X_val, y_val, args):
    """
    Train the model with data augmentation and callbacks.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        args: Command line arguments
        
    Returns:
        Training history
    """
    print("\n[4/6] Training model...")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    # Data augmentation
    datagen = create_data_augmentation()
    datagen.fit(X_train)
    
    # Callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'skin_disease_model.weights.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Calculate class weights for imbalanced dataset
    y_integers = np.argmax(y_train, axis=1)
    class_counts = Counter(y_integers)
    total = sum(class_counts.values())
    class_weights = {
        cls: total / (NUM_CLASSES * count) 
        for cls, count in class_counts.items()
    }
    
    print(f"\n  Class weights (for imbalance correction):")
    for cls, weight in class_weights.items():
        print(f"    {CLASS_NAMES[cls]:10s}: {weight:.3f}")
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=len(X_train) // args.batch_size,
        verbose=1
    )
    
    # Save final weights
    model.save_weights(model_path)
    print(f"\n  → Model weights saved to: {model_path}")
    
    return history


def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the model and generate classification report.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        output_dir: Directory to save evaluation results
    """
    print("\n[5/6] Evaluating model...")
    
    # Predict
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    target_names = [f"{CLASS_LABELS[c]} ({c})" for c in CLASS_NAMES]
    report = classification_report(y_true_classes, y_pred_classes, 
                                    target_names=target_names, digits=4)
    print("\n  Classification Report:")
    print("  " + report.replace("\n", "\n  "))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Save evaluation plots
    save_evaluation_plots(cm, output_dir)
    
    # Overall accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"\n  → Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def save_evaluation_plots(cm, output_dir):
    """Save confusion matrix and training history plots."""
    print("\n[6/6] Saving evaluation plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    short_names = [c.upper() for c in CLASS_NAMES]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=short_names,
           yticklabels=short_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Confusion matrix saved to: {plot_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("  [+] Skin Disease Classification -- Model Training")
    print("=" * 60)
    print(f"\n  Data directory: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output directory: {args.output_dir}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  [GPU] GPU detected: {gpus[0].name}")
        # Prevent TF from taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  [!] No GPU detected. Training will use CPU (slower).")
    
    # Load dataset
    df = load_dataset(args.data_dir)
    
    # Load and preprocess images
    X, y = load_and_preprocess_images(df)
    
    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"\n  Dataset split:")
    print(f"    Train:      {X_train.shape[0]:,} samples")
    print(f"    Validation: {X_val.shape[0]:,} samples")
    print(f"    Test:       {X_test.shape[0]:,} samples")
    
    # Build model
    model = build_model()
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val, args)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"  [OK] Training Complete!")
    print(f"  → Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"  -> Model saved to: {args.output_dir}/skin_disease_model.weights.h5")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
