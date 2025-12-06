"""
Real-time inference utilities for behavior classification
Optimized for speed and mobile deployment
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

class BehaviorInference:
    """Real-time behavior inference"""
    
    def __init__(self, model_path, class_names=None, tflite_mode=False):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved model or TFLite model
            class_names: List of class names
            tflite_mode: Whether to use TFLite model
        """
        self.model_path = model_path
        self.class_names = class_names or ['Using Computer', 'Writing', 'Reading', 'Distracted', 'Sleepy']
        self.tflite_mode = tflite_mode
        
        if tflite_mode:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)
    
    def predict_image(self, image_path, return_confidence=True):
        """
        Predict behavior from image file
        
        Args:
            image_path: Path to image
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted class name and optional confidence scores
        """
        image = self._load_image(image_path)
        return self.predict(image, return_confidence=return_confidence)
    
    def predict(self, image_array, return_confidence=True):
        """
        Predict behavior from image array using letterboxing
        
        Args:
            image_array: Input image as numpy array (any size)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted class name and optional confidence scores
        """
        # Apply letterboxing to preserve aspect ratio
        from data_loader import BehaviorDataLoader
        image_array = BehaviorDataLoader._letterbox(image_array, (224, 224))
        image_array = image_array.astype('float32') / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Inference
        if self.tflite_mode:
            predictions = self._predict_tflite(image_batch)
        else:
            predictions = self.model.predict(image_batch, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        if return_confidence:
            confidence_dict = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'all_scores': confidence_dict
            }
        
        return predicted_class, confidence
    
    def predict_batch(self, image_paths):
        """
        Predict behavior for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of predictions
        """
        predictions = []
        for image_path in image_paths:
            pred = self.predict_image(image_path)
            predictions.append(pred)
        return predictions
    
    def _predict_tflite(self, image_batch):
        """Run inference with TFLite model"""
        self.interpreter.set_tensor(self.input_details[0]['index'], 
                                    image_batch.astype(np.float32))
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    @staticmethod
    def _load_image(image_path, target_size=(224, 224)):
        """Load and preprocess image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        return image
    
    @staticmethod
    def capture_webcam(model_path, class_names=None, tflite_mode=False, confidence_threshold=0.7):
        """
        Real-time inference from webcam
        Press 'q' to quit
        
        Args:
            model_path: Path to model
            class_names: List of class names
            tflite_mode: Whether to use TFLite
            confidence_threshold: Minimum confidence to display
        """
        inference = BehaviorInference(model_path, class_names, tflite_mode)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every 3 frames for speed
            if frame_count % 3 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = inference.predict(rgb_frame)
                
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Draw results if confidence is high
                if confidence > confidence_threshold:
                    text = f"{prediction}: {confidence:.2f}"
                    cv2.putText(frame, text, (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.imshow('Behavior Classifier', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
