"""
MobileNetV2 Transfer Learning Model for Human Behavior Classification
Optimized for real-time inference on mobile devices
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class BehaviorClassifier:
    """Transfer learning model for 5 human behaviors: absent, using computer, reading/writing, distracted, sleepy"""
    
    def __init__(self, num_classes=5, input_shape=(224, 224, 3), model_name='mobilenetv2', class_names=None):
        """
        Initialize the behavior classifier with transfer learning
        
        Args:
            num_classes: Number of behavior classes (default: 5)
            input_shape: Input image shape (default: 224x224x3)
            model_name: Base model ('mobilenetv2' for mobile optimization)
            class_names: List of class names (optional)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_name = model_name
        self.model = None
        self.class_names = class_names or ['Absent', 'Using_Computer', 'Reading_Writing', 'Distracted', 'Sleepy']
        
    def build_model(self, freeze_base=True, dropout_rate=0.5):
        """
        Build transfer learning model using MobileNetV2
        
        Args:
            freeze_base: Whether to freeze base model weights during training
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers if specified
        if freeze_base:
            base_model.trainable = False
        
        # Build custom head
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        )
        
        self.model = model
        return model
    
    def get_model(self):
        """Get the compiled model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model
    
    def unfreeze_base(self, num_layers=50):
        """
        Unfreeze top layers of base model for fine-tuning
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        # Find the base model (MobileNetV2)
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # Check if it's a model with layers
                base_model = layer
                break
        
        if base_model is None:
            print("Warning: Could not find base model for unfreezing. Skipping fine-tuning phase.")
            return
        
        base_model.trainable = True
        
        # Freeze bottom layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def convert_to_tflite(self, output_path, quantize=False):
        """
        Convert model to TensorFlow Lite for mobile deployment
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to apply dynamic range quantization (no representative_dataset needed)
        """
        if self.model is None:
            raise ValueError("No model to convert")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Use dynamic range quantization (doesn't require representative_dataset)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    def get_summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.summary()
