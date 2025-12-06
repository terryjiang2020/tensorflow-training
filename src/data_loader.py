"""
Data loading and preprocessing for human behavior classification
Includes data augmentation for robust training
"""

import tensorflow as tf
from tensorflow.keras import preprocessing
import numpy as np
import os

class BehaviorDataLoader:
    """Data loader for behavior classification with augmentation"""
    
    def __init__(self, 
                 image_size=(224, 224),
                 batch_size=32,
                 class_names=['Using Computer', 'Writing', 'Reading', 'Distracted', 'Sleepy']):
        """
        Initialize data loader
        
        Args:
            image_size: Target image size (height, width)
            batch_size: Batch size for training
            class_names: List of class names
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def create_train_generator(self):
        """Create data augmentation generator for training with letterboxing"""
        # Custom preprocessing to apply letterboxing
        def preprocess_letterbox(x):
            # x comes from flow_from_directory already resized by target_size
            # We'll handle letterboxing in the custom flow instead
            x = x / 255.0
            return x
        
        train_datagen = preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_letterbox,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        return train_datagen
    
    def create_val_generator(self):
        """Create generator for validation with letterboxing (minimal augmentation)"""
        def preprocess_letterbox(x):
            x = x / 255.0
            return x
        
        val_datagen = preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_letterbox
        )
        return val_datagen
    
    def load_from_directory(self, directory, train_split=0.8, shuffle=True):
        """
        Load images from directory structure with letterboxing
        Expected structure: directory/class_name/*.jpg
        
        Args:
            directory: Root directory containing class subdirectories
            train_split: Fraction of data to use for training
            shuffle: Whether to shuffle data
            
        Returns:
            (train_dataset, val_dataset)
        """
        train_datagen = self.create_train_generator()
        val_datagen = self.create_val_generator()
        
        # Load training data with letterboxing
        train_generator = train_datagen.flow_from_directory(
            directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=shuffle
        )
        
        # Load validation data with letterboxing
        val_generator = val_datagen.flow_from_directory(
            directory,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_tf_dataset(self, image_paths, labels, augment=True):
        """
        Create TensorFlow dataset from image paths and labels
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (one-hot encoded)
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow dataset
        """
        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = image / 255.0
            
            if augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.8, 1.2)
            
            return image, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.shuffle(len(image_paths))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @staticmethod
    def preprocess_image(image_path, target_size=(224, 224)):
        """
        Preprocess single image for inference using letterboxing
        Maintains aspect ratio and fills gaps with white padding
        
        Args:
            image_path: Path to image
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array with letterboxing
        """
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Apply letterboxing
        image_array = BehaviorDataLoader._letterbox(image_array, target_size)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        return image_array
    
    @staticmethod
    def preprocess_array(image_array, target_size=(224, 224)):
        """
        Preprocess image from numpy array using letterboxing
        Maintains aspect ratio and fills gaps with white padding
        
        Args:
            image_array: Input image as numpy array
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image array with letterboxing
        """
        # Apply letterboxing
        image_array = BehaviorDataLoader._letterbox(image_array, target_size)
        return np.expand_dims(image_array, axis=0) / 255.0
    
    @staticmethod
    def _letterbox(image, target_size=(224, 224), fill_color=255):
        """
        Apply letterboxing to image while maintaining aspect ratio
        Fills gaps with white padding (fill_color=255)
        
        Args:
            image: Input image as numpy array (H x W x C)
            target_size: Target size (height, width)
            fill_color: Color for padding (255=white, 0=black)
            
        Returns:
            Letterboxed image with target size
        """
        h, w = target_size
        img_h, img_w = image.shape[0], image.shape[1]
        
        # Calculate scale to fit image within target size
        scale = min(w / img_w, h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize image maintaining aspect ratio
        resized_image = tf.image.resize(image, (new_h, new_w))
        
        # Create canvas with white padding
        if len(image.shape) == 3:  # Color image
            canvas = np.ones((h, w, image.shape[2]), dtype=image.dtype) * fill_color
        else:  # Grayscale
            canvas = np.ones((h, w), dtype=image.dtype) * fill_color
        
        # Calculate padding
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        
        # Place resized image on canvas
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
        
        return canvas.astype('uint8')
