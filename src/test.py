"""
Test and inference script for behavior classification model
Supports both Keras H5 and TFLite models
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import time

class BehaviorTester:
    """Test and inference for behavior classification"""
    
    def __init__(self, model_path, class_names=None, use_tflite=False):
        """
        Initialize tester
        
        Args:
            model_path: Path to model file (.h5 or .tflite)
            class_names: List of class names
            use_tflite: Whether to use TFLite model
        """
        self.model_path = model_path
        self.class_names = class_names or ['Absent', 'Using_Computer', 'Reading_Writing', 'Distracted', 'Sleepy']
        self.use_tflite = use_tflite
        
        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✓ TFLite model loaded: {model_path}")
        else:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✓ Keras model loaded: {model_path}")
    
    def predict_image(self, image_path, confidence_threshold=0.0):
        """
        Predict behavior from image file
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Prediction result dict
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict(image, confidence_threshold)
    
    def predict(self, image, confidence_threshold=0.0):
        """
        Predict behavior from image array
        
        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Dict with predictions
        """
        from src.data_loader import BehaviorDataLoader
        
        # Apply letterboxing
        image = BehaviorDataLoader._letterbox(image, (224, 224))
        image = image.astype('float32') / 255.0
        image_batch = np.expand_dims(image, axis=0)
        
        # Get predictions
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], image_batch)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            predictions = self.model.predict(image_batch, verbose=0)
        
        # Process results
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])
        pred_class = self.class_names[pred_idx]
        
        if confidence < confidence_threshold:
            return {
                'prediction': 'UNCERTAIN',
                'confidence': confidence,
                'all_scores': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
            }
        
        return {
            'prediction': pred_class,
            'confidence': confidence,
            'all_scores': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        }
    
    def test_directory(self, directory, confidence_threshold=0.0):
        """
        Test on all images in a directory
        
        Args:
            directory: Directory containing images
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of results
        """
        results = []
        image_files = list(Path(directory).glob('*.jpg')) + list(Path(directory).glob('*.png'))
        
        print(f"Testing {len(image_files)} images from {directory}...")
        
        for image_path in image_files:
            try:
                result = self.predict_image(str(image_path), confidence_threshold)
                result['file'] = image_path.name
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def benchmark_speed(self, num_iterations=100):
        """
        Benchmark inference speed
        
        Args:
            num_iterations: Number of iterations to test
            
        Returns:
            Speed metrics
        """
        # Create dummy image
        dummy_image = np.random.rand(224, 224, 3).astype('uint8')
        
        print(f"Benchmarking inference speed ({num_iterations} iterations)...")
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(dummy_image)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times[10:])  # Skip first 10 for warmup
        
        metrics = {
            'mean_ms': np.mean(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1 / np.mean(times)
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Mean: {metrics['mean_ms']:.2f}ms")
        print(f"  Median: {metrics['median_ms']:.2f}ms")
        print(f"  Min: {metrics['min_ms']:.2f}ms")
        print(f"  Max: {metrics['max_ms']:.2f}ms")
        print(f"  FPS: {metrics['fps']:.1f}")
        
        return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test behavior classification model')
    parser.add_argument('--model', type=str, default='./models/behavior_classifier.tflite',
                       help='Path to model file')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite model')
    parser.add_argument('--image', type=str, help='Test single image')
    parser.add_argument('--directory', type=str, help='Test directory of images')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Use TFLite by default
    use_tflite = args.tflite or args.model.endswith('.tflite')
    
    # Initialize tester
    tester = BehaviorTester(args.model, use_tflite=use_tflite)
    
    # Run tests
    if args.benchmark:
        tester.benchmark_speed()
    elif args.image:
        result = tester.predict_image(args.image, args.threshold)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"All scores:")
        for class_name, score in result['all_scores'].items():
            print(f"  {class_name:>20}: {score:.4f}")
    elif args.directory:
        results = tester.test_directory(args.directory, args.threshold)
        print(f"\nTested {len(results)} images")
    else:
        print("Please specify --image, --directory, or --benchmark")
