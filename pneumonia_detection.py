# Pneumonia Detection System
# A machine learning project to detect pneumonia from chest X-ray images

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class PneumoniaDetector:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def create_model(self, use_pretrained=True):
        """Create CNN model for pneumonia detection"""
        if use_pretrained:
            # Use pre-trained VGG16 model
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            base_model.trainable = False
            
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
        else:
            # Custom CNN model
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', 
                             input_shape=(self.img_height, self.img_width, 3)),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            classes=['NORMAL', 'PNEUMONIA']
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary',
            classes=['NORMAL', 'PNEUMONIA']
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=20):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_single_image(self, image_path, show_image=True):
        """Predict pneumonia for a single image"""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return None
        
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)
        probability = prediction[0][0]
        
        result = "PNEUMONIA" if probability > 0.5 else "NORMAL"
        confidence = probability if probability > 0.5 else 1 - probability
        
        if show_image:
            plt.figure(figsize=(8, 6))
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            plt.imshow(original_img)
            plt.title(f'Prediction: {result}\nConfidence: {confidence:.2%}')
            plt.axis('off')
            plt.show()
        
        return {
            'prediction': result,
            'probability': probability,
            'confidence': confidence
        }
    
    def evaluate_model(self, test_generator):
        """Evaluate model on test data"""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return None
        
        # Get predictions
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=class_labels))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and setup
def main():
    """Main function to demonstrate usage"""
    print("Pneumonia Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = PneumoniaDetector()
    
    # Example directory structure (you need to organize your data like this):
    """
    data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
    """
    
    # Paths (adjust these to your data location)
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'
    
    print("1. Creating model...")
    model = detector.create_model(use_pretrained=True)
    print(f"Model created with {model.count_params():,} parameters")
    
    # Check if data directories exist
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("\n2. Preparing data generators...")
        train_gen, val_gen = detector.prepare_data_generators(train_dir, val_dir)
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        
        print("\n3. Training model...")
        history = detector.train_model(train_gen, val_gen, epochs=10)
        
        print("\n4. Plotting training history...")
        detector.plot_training_history()
        
        if os.path.exists(test_dir):
            print("\n5. Evaluating model...")
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_gen = test_datagen.flow_from_directory(
                test_dir,
                target_size=(detector.img_height, detector.img_width),
                batch_size=32,
                class_mode='binary',
                classes=['NORMAL', 'PNEUMONIA'],
                shuffle=False
            )
            
            results = detector.evaluate_model(test_gen)
            print(f"Test Accuracy: {results['accuracy']:.2%}")
        
        print("\n6. Saving model...")
        detector.save_model('pneumonia_detector.h5')
    
    else:
        print("Data directories not found. Please organize your data as shown above.")
        print("You can download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")

# Data download helper function
def download_dataset_info():
    """Information about downloading the dataset"""
    info = """
    Dataset Information:
    ===================
    
    1. Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:
       https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    
    2. Extract the dataset and organize it as follows:
       data/
       ├── train/
       │   ├── NORMAL/      (1,341 images)
       │   └── PNEUMONIA/   (3,875 images)
       ├── val/
       │   ├── NORMAL/      (8 images)
       │   └── PNEUMONIA/   (8 images)
       └── test/
           ├── NORMAL/      (234 images)
           └── PNEUMONIA/   (390 images)
    
    3. Install required packages:
       pip install tensorflow opencv-python pillow matplotlib seaborn scikit-learn
    
    4. Run the main() function to start training
    """
    print(info)

if __name__ == "__main__":
    # Show dataset download information
    download_dataset_info()