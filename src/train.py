"""
Training script for behavior classification model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
from pathlib import Path
import json
from model import BehaviorClassifier
from data_loader import BehaviorDataLoader

def train_model(
    data_dir,
    output_dir='./models',
    epochs=25,
    batch_size=32,
    learning_rate=0.001,
    freeze_base=True,
    fine_tune_epochs=10
):
    """
    Train behavior classification model
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save models
        epochs: Number of epochs for initial training
        batch_size: Batch size
        learning_rate: Initial learning rate
        freeze_base: Whether to freeze base model initially
        fine_tune_epochs: Number of epochs for fine-tuning
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize model and data loader
    print("Initializing model and data loader...")
    classifier = BehaviorClassifier(num_classes=5)
    data_loader = BehaviorDataLoader(batch_size=batch_size)
    
    # Build model
    print("Building MobileNetV2 transfer learning model...")
    model = classifier.build_model(freeze_base=freeze_base)
    print(f"Model built. Trainable parameters: {model.count_params()}")
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            f'{output_dir}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Load data
    print(f"Loading data from {data_dir}...")
    try:
        train_generator, val_generator = data_loader.load_from_directory(data_dir)
        num_train_samples = train_generator.samples
        num_val_samples = val_generator.samples
        print(f"Training samples: {num_train_samples}, Validation samples: {num_val_samples}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure data directory has subdirectories for each class")
        return
    
    # Initial training with frozen base
    print(f"\nPhase 1: Training with frozen base model ({epochs} epochs)...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning with unfrozen layers
    print(f"\nPhase 2: Fine-tuning with unfrozen layers ({fine_tune_epochs} epochs)...")
    classifier.unfreeze_base(num_layers=50)
    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model_path = f'{output_dir}/behavior_classifier.h5'
    classifier.save_model(model_path)
    
    # Also save in native Keras format for better TFLite compatibility
    keras_path = f'{output_dir}/behavior_classifier.keras'
    print(f"Saving model in native Keras format: {keras_path}")
    model.save(keras_path, save_format='keras')
    
    # Convert to TFLite using the native Keras format
    tflite_path = f'{output_dir}/behavior_classifier.tflite'
    print(f"\nConverting to TFLite for mobile deployment...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")
        print("The H5 model is still available for use.")
    
    # Save training history
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"TFLite model saved to: {tflite_path}")
    
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train behavior classification model')
    parser.add_argument('--data_dir', type=str, default='./data/behavior_dataset',
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
