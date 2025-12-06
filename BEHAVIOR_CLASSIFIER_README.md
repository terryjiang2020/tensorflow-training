# Human Behavior Classification with TensorFlow

A complete real-time human behavior classification system using MobileNetV2 transfer learning, optimized for mobile deployment.

## Features

✅ **Transfer Learning** - MobileNetV2 pre-trained backbone for fast training  
✅ **5 Behavior Classes** - Using Computer, Writing, Reading, Distracted, Sleepy  
✅ **Mobile Optimized** - TensorFlow Lite conversion with quantization  
✅ **Real-time Inference** - Webcam support with sub-50ms latency  
✅ **Data Augmentation** - Rotation, zoom, brightness, horizontal flip  
✅ **Two-phase Training** - Frozen base + fine-tuning for best results  

## Project Structure

```
├── src/
│   ├── model.py           # MobileNetV2 classifier with TFLite export
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── train.py           # Training script
│   └── inference.py       # Real-time inference utilities
├── train_behavior_classifier.ipynb  # Complete training notebook
├── models/                # Saved models (after training)
├── data/                  # Dataset directory
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your images in the following structure:

```
data/behavior_dataset/
├── Using Computer/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Writing/
├── Reading/
├── Distracted/
└── Sleepy/
```

### 3. Train Model

**Option A: Using Jupyter Notebook**
```bash
jupyter notebook train_behavior_classifier.ipynb
```

**Option B: Using Python Script**
```bash
python src/train.py \
  --data_dir ./data/behavior_dataset \
  --output_dir ./models \
  --epochs 25 \
  --batch_size 32
```

### 4. Real-time Inference

```python
from src.inference import BehaviorInference

# Initialize with TFLite for mobile
inference = BehaviorInference(
    './models/behavior_classifier.tflite',
    tflite_mode=True
)

# Predict from image
result = inference.predict_image('./path/to/image.jpg')
print(f"Behavior: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Real-time webcam inference
BehaviorInference.capture_webcam('./models/behavior_classifier.tflite')
```

## Model Performance

### Expected Results
- **Training Time**: ~5-10 minutes (with GPU/Apple Metal)
- **Inference Speed**: 20-50ms per image (TFLite)
- **Model Size**: ~30MB (TFLite quantized)
- **Accuracy**: 85-95% (depends on dataset quality)

### Evaluation Metrics
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Per-class accuracy
- Overall accuracy

## Mobile Deployment

### TensorFlow Lite Model
The trained model is automatically converted to TFLite format:
- **File**: `./models/behavior_classifier.tflite`
- **Size**: ~30MB
- **Framework Support**: 
  - iOS (Core ML)
  - Android (TensorFlow Lite)
  - Web (TensorFlow.js)
  - React Native, Flutter

### Integration Examples

**Android (TensorFlow Lite)**
```java
Interpreter tflite = new Interpreter(modelBuffer);
float[][] input = new float[1][224 * 224 * 3];
float[][] output = new float[1][5];
tflite.run(input, output);
```

**iOS (Core ML)**
```swift
let model = try BehaviorClassifier(configuration: MLModelConfiguration())
let prediction = try model.prediction(input: input)
```

## Training Strategy

### Phase 1: Transfer Learning (Frozen Base)
- Freeze MobileNetV2 weights
- Train only custom head layers
- 25 epochs
- Learning rate: 0.001

### Phase 2: Fine-tuning
- Unfreeze top 50 layers of MobileNetV2
- Train entire model
- 10 epochs
- Learning rate: 0.0001

## Data Augmentation

Applied during training:
- Random rotation (±20°)
- Width/height shift (±20%)
- Horizontal flip
- Zoom (±20%)
- Brightness adjustment (0.8-1.2x)

## Dependencies

```
TensorFlow >= 2.16.2
NumPy >= 1.21.0
OpenCV >= 4.5.0
Matplotlib >= 3.5.0
Scikit-learn >= 1.0.0
Pillow >= 9.0.0
```

## Performance Optimization

### Speed
- MobileNetV2: lightweight architecture
- TensorFlow Lite: optimized for mobile
- Quantization: 4x compression
- Batch inference support

### Accuracy
- Transfer learning from ImageNet
- Two-phase training strategy
- Data augmentation
- Early stopping to prevent overfitting

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
data_loader = BehaviorDataLoader(batch_size=16)
```

### Low Accuracy
- Check dataset quality
- Increase epochs
- Add more data augmentation
- Use higher resolution images

### Slow Training
- Use GPU/Metal acceleration
- Reduce image size
- Use smaller batch size

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## License

MIT License

## Author

Behavior Classification Model - TensorFlow Training
