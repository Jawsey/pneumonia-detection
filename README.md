# Pneumonia Detection System

A machine learning project that uses deep learning to detect pneumonia from chest X-ray images. The system leverages convolutional neural networks (CNNs) to classify X-ray images as either normal or showing signs of pneumonia Developed by Jawad for free .

## 🚀 Features

- **Dual Model Architecture**: Choose between pre-trained VGG16 or custom CNN
- **Data Augmentation**: Improves model robustness with image transformations
- **Real-time Prediction**: Classify individual X-ray images
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Model Persistence**: Save and load trained models
- **Interactive Visualization**: Training history and confusion matrix plots

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
tensorflow>=2.8.0
opencv-python>=4.5.0
pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
numpy>=1.19.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/jawsey/pneumonia-detection.git
cd pneumonia-detection
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Download Dataset
1. Visit the [Kaggle Chest X-Ray Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download and extract the dataset
3. Organize the data according to the structure below

### Dataset Structure
```
data/
├── train/
│   ├── NORMAL/      # 1,341 normal chest X-ray images
│   └── PNEUMONIA/   # 3,875 pneumonia chest X-ray images
├── val/
│   ├── NORMAL/      # 8 normal validation images
│   └── PNEUMONIA/   # 8 pneumonia validation images
└── test/
    ├── NORMAL/      # 234 normal test images
    └── PNEUMONIA/   # 390 pneumonia test images
```

### Dataset Statistics
- **Total Images**: 5,856
- **Training Images**: 5,216 (1,341 Normal + 3,875 Pneumonia)
- **Validation Images**: 16 (8 Normal + 8 Pneumonia)
- **Test Images**: 624 (234 Normal + 390 Pneumonia)

## 🎯 Usage

### Basic Usage

```python
from pneumonia_detector import PneumoniaDetector

# Initialize detector
detector = PneumoniaDetector()

# Create model (use pre-trained VGG16)
model = detector.create_model(use_pretrained=True)

# Prepare data
train_gen, val_gen = detector.prepare_data_generators('data/train', 'data/val')

# Train model
history = detector.train_model(train_gen, val_gen, epochs=20)

# Visualize training
detector.plot_training_history()

# Save model
detector.save_model('pneumonia_model.h5')
```

### Single Image Prediction

```python
# Load trained model
detector.load_model('pneumonia_model.h5')

# Predict single image
result = detector.predict_single_image('path/to/xray.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Model Evaluation

```python
# Evaluate on test set
test_gen = detector.prepare_test_generator('data/test')
results = detector.evaluate_model(test_gen)
print(f"Test Accuracy: {results['accuracy']:.2%}")
```

## 🏗️ Model Architecture

### Pre-trained Model (VGG16)
- Base: VGG16 pre-trained on ImageNet
- Global Average Pooling
- Dense layers with dropout for regularization
- Binary classification output

### Custom CNN Model
- 4 Convolutional blocks with MaxPooling
- Progressive filter increase (32→64→128→128)
- Dropout layers for regularization
- Dense layers for classification

## 📈 Performance

### Expected Results
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: ~85-90%

### Model Metrics
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve Analysis

## 🔧 Configuration

### Model Parameters
```python
# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
```

### Data Augmentation Settings
- Rotation: ±20°
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom range: ±20%
- Shear range: ±20%

## 📝 Examples

### Complete Training Pipeline
```python
def train_pneumonia_detector():
    detector = PneumoniaDetector()
    
    # Create and compile model
    model = detector.create_model(use_pretrained=True)
    
    # Prepare data generators
    train_gen, val_gen = detector.prepare_data_generators(
        'data/train', 'data/val', batch_size=32
    )
    
    # Train model
    history = detector.train_model(train_gen, val_gen, epochs=20)
    
    # Visualize results
    detector.plot_training_history()
    
    # Save model
    detector.save_model('pneumonia_detector.h5')
    
    return detector

# Run training
detector = train_pneumonia_detector()
```

### Batch Prediction
```python
import os

def predict_directory(detector, directory_path):
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(directory_path, filename)
            result = detector.predict_single_image(filepath, show_image=False)
            results.append({
                'filename': filename,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
    return results
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce batch size
   - Use smaller image dimensions
   - Enable mixed precision training

2. **Low Accuracy**
   - Increase training epochs
   - Adjust learning rate
   - Try different model architectures

3. **Overfitting**
   - Increase dropout rates
   - Add more data augmentation
   - Use early stopping

### Performance Optimization
```python
# Enable mixed precision for faster training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation including:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve**: Receiver Operating Characteristic analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Paul Mooney - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Pre-trained models: TensorFlow/Keras
- Inspiration: Medical imaging research community

## 🔮 Future Enhancements

- [ ] Multi-class classification (Normal, Bacterial, Viral)
- [ ] Web interface for easy image upload
- [ ] Mobile app 

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.
