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

CLASS_NAMES = ['Absent', 'Using_Computer', 'Reading_Writing', 'Distracted', 'Sleepy']

def train_model(
    data_dir,
    output_dir='./models',
    epochs=25,
    batch_size=32,
    learning_rate=0.001,
    freeze_base=True,
    fine_tune_epochs=10,
    use_class_weights=True,
    class_names=CLASS_NAMES,
    distracted_weight_boost=1.2,
    sleepy_weight_boost=1.0,
    extra_finetune_epochs=5,
    extra_finetune_lr=5e-5
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
        use_class_weights: Whether to use class weights to handle imbalance
        class_names: Ordered list of class names matching directory names
        distracted_weight_boost: Multiplier for the Distracted class weight
        sleepy_weight_boost: Multiplier for the Sleepy class weight
        extra_finetune_epochs: Additional fine-tune epochs at low LR
        extra_finetune_lr: Learning rate for the extra fine-tune phase
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize model and data loader
    print("Initializing model and data loader...")
    classifier = BehaviorClassifier(num_classes=len(class_names))
    data_loader = BehaviorDataLoader(batch_size=batch_size, class_names=class_names)
    
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
    
    # Calculate class weights if enabled
    class_weights = None
    if use_class_weights:
        print("\nCalculating class weights to handle imbalance...")
        counts = np.bincount(train_generator.classes, minlength=len(class_names))
        missing = np.where(counts == 0)[0]

        if len(missing) > 0:
            missing_names = [name for name, idx in train_generator.class_indices.items() if idx in missing]
            print(f"  ⚠️ Missing samples for classes: {missing_names}. Disabling class weights.")
        else:
            total_samples = np.sum(counts)
            num_classes = len(class_names)
            class_weights = {}
            for idx, count in enumerate(counts):
                weight = total_samples / (num_classes * count)
                class_weights[idx] = weight
                class_name = [k for k, v in train_generator.class_indices.items() if v == idx][0]
                print(f"  {class_name}: weight = {weight:.3f} (samples: {count})")

            # Boost Distracted class weight modestly to improve recall
            if 'Distracted' in train_generator.class_indices:
                d_idx = train_generator.class_indices['Distracted']
                class_weights[d_idx] *= distracted_weight_boost
                print(f"  Applied distracted_weight_boost x{distracted_weight_boost:.2f} -> {class_weights[d_idx]:.3f}")
            
            # Boost Sleepy class weight to improve recall
            if sleepy_weight_boost > 1.0 and 'Sleepy' in train_generator.class_indices:
                s_idx = train_generator.class_indices['Sleepy']
                class_weights[s_idx] *= sleepy_weight_boost
                print(f"  Applied sleepy_weight_boost x{sleepy_weight_boost:.2f} -> {class_weights[s_idx]:.3f}")
    
    # Initial training with frozen base
    print(f"\nPhase 1: Training with frozen base model ({epochs} epochs)...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
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
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Optional extra fine-tune with very low LR to refine decision boundaries
    if extra_finetune_epochs and extra_finetune_epochs > 0:
        print(f"\nPhase 3: Extra fine-tuning at low LR ({extra_finetune_epochs} epochs, lr={extra_finetune_lr})...")
        try:
            if hasattr(model.optimizer, "learning_rate") and hasattr(model.optimizer.learning_rate, "assign"):
                model.optimizer.learning_rate.assign(extra_finetune_lr)
            else:
                keras.backend.set_value(model.optimizer.lr, extra_finetune_lr)
        except Exception as e:
            print(f"  Warning: could not set learning rate dynamically ({e}). Using existing LR.")
        history_extra = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=extra_finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history_extra = None
    
    # Save model
    model_path = f'{output_dir}/behavior_classifier.h5'
    classifier.save_model(model_path)
    
    # Also save in native Keras and SavedModel formats for TFLite compatibility
    keras_path = f'{output_dir}/behavior_classifier.keras'
    saved_model_path = f'{output_dir}/behavior_classifier_saved'
    print(f"Saving model in native Keras format: {keras_path}")
    model.save(keras_path)
    print(f"Exporting SavedModel to: {saved_model_path}")
    model.export(saved_model_path)
    
    # Convert to TFLite using SavedModel
    tflite_path = f'{output_dir}/behavior_classifier.tflite'
    print(f"\nConverting to TFLite for mobile deployment...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")
        print("The H5/Keras models are still available for use.")
    
    # Save training history
    history_dict = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'fine_tune_accuracy': history_finetune.history['accuracy'],
        'fine_tune_val_accuracy': history_finetune.history['val_accuracy'],
        'fine_tune_loss': history_finetune.history['loss'],
        'fine_tune_val_loss': history_finetune.history['val_loss'],
    }

    if history_extra:
        history_dict.update({
            'extra_accuracy': history_extra.history['accuracy'],
            'extra_val_accuracy': history_extra.history['val_accuracy'],
            'extra_loss': history_extra.history['loss'],
            'extra_val_loss': history_extra.history['val_loss'],
        })
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
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='Use class weights to handle imbalance (default: True)')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false',
                       help='Disable class weights')
    parser.add_argument('--distracted_weight_boost', type=float, default=1.2,
                        help='Multiplier applied to Distracted class weight (default: 1.2)')
    parser.add_argument('--sleepy_weight_boost', type=float, default=1.0,
                        help='Multiplier applied to Sleepy class weight (default: 1.0)')
    parser.add_argument('--extra_finetune_epochs', type=int, default=5,
                        help='Additional fine-tune epochs at low LR after main fine-tune (default: 5)')
    parser.add_argument('--extra_finetune_lr', type=float, default=5e-5,
                        help='Learning rate for the extra fine-tune phase (default: 5e-5)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=args.use_class_weights,
        distracted_weight_boost=args.distracted_weight_boost,
        sleepy_weight_boost=args.sleepy_weight_boost,
        extra_finetune_epochs=args.extra_finetune_epochs,
        extra_finetune_lr=args.extra_finetune_lr
    )
