# Mobile Integration Guide

Complete guide for integrating the behavior classification model into mobile applications.

## Model Files

```
models/
├── behavior_classifier.tflite       # TFLite model for mobile (10.1 MB)
├── behavior_classifier.h5           # Keras model for reference (27.31 MB)
├── behavior_classifier_saved/       # SavedModel format
└── confusion_matrix.png             # Evaluation metrics
```

## Model Specifications

- **Model Type**: MobileNetV2 + Custom Head
- **Input**: 224×224 RGB image (letterboxed with white padding)
- **Output**: 5 class probabilities
- **Classes**: 
  - 0: Absent
  - 1: Using_Computer
  - 2: Reading_Writing
  - 3: Distracted
  - 4: Sleepy
- **Size**: 10.1 MB (TFLite)
- **Latency**: 20-50ms per image
- **Hardware**: Optimized for mobile (ARM64, x86)

## Android Integration

### 1. Add TensorFlow Lite Dependency

**build.gradle** (Module: app)
```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### 2. Add Model to Assets

```
app/src/main/assets/
└── behavior_classifier.tflite
```

### 3. Basic Inference Implementation

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeAndPaddingOp
import android.graphics.Bitmap

class BehaviorClassifier(context: Context) {
    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor
    
    init {
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
        
        // Create processor with letterboxing
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeAndPaddingOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
    }
    
    fun classify(bitmap: Bitmap): Map<String, Float> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)
        
        // Run inference
        val outputArray = Array(1) { FloatArray(5) }
        interpreter.run(processedImage.buffer, outputArray)
        
        // Process results
        val classNames = listOf(
            "Absent", "Using_Computer", "Reading_Writing", "Distracted", "Sleepy"
        )
        val results = mutableMapOf<String, Float>()
        outputArray[0].forEachIndexed { index, confidence ->
            results[classNames[index]] = confidence
        }
        
        return results
    }
    
    private fun loadModelFile(context: Context): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("behavior_classifier.tflite")
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = (inputStream as FileInputStream).channel
        
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }
}
```

### 4. Real-time Camera Implementation

```kotlin
class CameraClassifier(private val context: Context) {
    private val classifier = BehaviorClassifier(context)
    
    fun processCameraFrame(bitmap: Bitmap) {
        val results = classifier.classify(bitmap)
        val prediction = results.maxByOrNull { it.value }
        
        println("Behavior: ${prediction?.key}")
        println("Confidence: ${prediction?.value}")
    }
}
```

## iOS Integration

### 1. Add TensorFlow Lite CocoaPod

**Podfile**
```ruby
pod 'TensorFlowLiteSwift'
pod 'TensorFlowLiteSwift/CoreML'
```

### 2. Add Model to Bundle

```
YourApp/
└── Resources/
    └── behavior_classifier.tflite
```

### 3. Basic Inference Implementation

```swift
import TensorFlowLite
import UIKit

class BehaviorClassifier {
    let interpreter: Interpreter
    let classNames = ["Absent", "Using_Computer", "Reading_Writing", "Distracted", "Sleepy"]
    
    init?() {
        guard let modelPath = Bundle.main.path(forResource: "behavior_classifier", 
                                               ofType: "tflite") else {
            return nil
        }
        
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
        } catch {
            print("Failed to load model: \(error)")
            return nil
        }
    }
    
    func classify(image: UIImage) -> [String: Float]? {
        guard let rgbData = imageToData(image) else { return nil }
        
        do {
            let inputTensor = try interpreter.input(at: 0)
            try interpreter.copy(rgbData, toInputAt: 0)
            try interpreter.invoke()
            
            let outputTensor = try interpreter.output(at: 0)
            let results = UnsafeMutableBufferPointer<Float>(start: 
                outputTensor.data.assumingMemoryBound(to: Float.self),
                count: 5)
            
            var predictions: [String: Float] = [:]
            for i in 0..<5 {
                predictions[classNames[i]] = results[i]
            }
            
            return predictions
        } catch {
            print("Inference failed: \(error)")
            return nil
        }
    }
    
    private func imageToData(_ image: UIImage) -> Data? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = 224
        let height = 224
        let bitmapBytesPerRow = width * 3
        let bitmapByteCount = bitmapBytesPerRow * height
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let context = CGContext(data: nil,
                                    width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bytesPerRow: bitmapBytesPerRow,
                                    space: colorSpace,
                                    bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }
        
        // Draw image with letterboxing
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let bytes = context.data else { return nil }
        return Data(bytes: bytes, count: bitmapByteCount)
    }
}
```

### 4. Real-time Camera Implementation

```swift
class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    let classifier = BehaviorClassifier()
    
    func captureOutput(_ output: AVCaptureOutput,
                      didOutput sampleBuffer: CMSampleBuffer,
                      from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Convert CMSampleBuffer to UIImage
        let image = UIImage(pixelBuffer: pixelBuffer)
        
        // Classify
        if let predictions = classifier?.classify(image: image) {
            let topPrediction = predictions.max { $0.value < $1.value }
            print("Behavior: \(topPrediction?.key ?? "Unknown")")
            print("Confidence: \(topPrediction?.value ?? 0.0)")
        }
    }
}
```

## React Native Integration

### 1. Install Dependencies

```bash
npm install @react-native-camera/camera react-native-tensorflow-lite
```

### 2. Model Usage

```javascript
import TensorflowLite from 'react-native-tensorflow-lite';

class BehaviorClassifier {
  constructor() {
    this.classNames = [
      'Absent',
      'Using_Computer',
      'Reading_Writing',
      'Distracted',
      'Sleepy'
    ];
  }

  async loadModel() {
    try {
      await TensorflowLite.loadModel({
        model: require('./assets/behavior_classifier.tflite'),
        inputs: [[1, 224, 224, 3]],
        outputs: [[1, 5]]
      });
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  }

  async classify(imageData) {
    try {
      const predictions = await TensorflowLite.runInference(imageData);
      
      const results = {};
      for (let i = 0; i < this.classNames.length; i++) {
        results[this.classNames[i]] = predictions[0][i];
      }
      
      return results;
    } catch (error) {
      console.error('Inference failed:', error);
    }
  }
}

// Usage in component
const classifier = new BehaviorClassifier();

useEffect(() => {
  classifier.loadModel();
}, []);

const handleClassify = async (image) => {
  const predictions = await classifier.classify(image);
  const behavior = Object.entries(predictions)
    .reduce((a, b) => a[1] > b[1] ? a : b);
  console.log(`Behavior: ${behavior[0]} (${behavior[1].toFixed(2)})`);
};
```

## Flutter Integration

### 1. Add Dependency

**pubspec.yaml**
```yaml
dependencies:
  tflite: ^1.1.2
```

### 2. Model Usage

```dart
import 'package:tflite/tflite.dart';

class BehaviorClassifier {
  static const List<String> classNames = [
    'Absent',
    'Using_Computer',
    'Reading_Writing',
    'Distracted',
    'Sleepy'
  ];

  static Future<void> loadModel() async {
    await Tflite.loadModel(
      model: 'assets/behavior_classifier.tflite',
      numThreads: 1,
      isAsset: true,
      useGpuDelegate: false,
    );
  }

  static Future<Map<String, double>> classify(String imagePath) async {
    var recognitions = await Tflite.runModelOnImage(
      path: imagePath,
      imageMean: 0.0,
      imageStd: 255.0,
      numResults: 5,
      threshold: 0.1,
    );

    Map<String, double> results = {};
    for (int i = 0; i < classNames.length; i++) {
      results[classNames[i]] = (recognitions?[i]['confidence'] as num).toDouble();
    }
    
    return results;
  }

  static Future<void> dispose() async {
    await Tflite.close();
  }
}

// Usage
void classifyImage() async {
  await BehaviorClassifier.loadModel();
  
  final predictions = await BehaviorClassifier.classify('/path/to/image.jpg');
  
  final topPrediction = predictions.entries
      .reduce((a, b) => a.value > b.value ? a : b);
  
  print('Behavior: ${topPrediction.key}');
  print('Confidence: ${topPrediction.value.toStringAsFixed(2)}');
}
```

## Image Preprocessing

The model expects **letterboxed images** (224×224) with white padding to preserve aspect ratio:

```python
def preprocess_image(image_array):
    """Letterbox image to 224x224 with white padding"""
    h, w = image_array.shape[:2]
    scale = min(224 / w, 224 / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize maintaining aspect ratio
    resized = cv2.resize(image_array, (new_w, new_h))
    
    # Create white canvas
    canvas = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Place resized image centered
    y_offset = (224 - new_h) // 2
    x_offset = (224 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize to [0, 1]
    return canvas.astype(np.float32) / 255.0
```

## Performance Tips

1. **Batch Processing**: Process multiple images at once for better throughput
2. **GPU Acceleration**: Enable GPU delegates in Android/iOS for faster inference
3. **Model Quantization**: Consider further quantization for 8-bit integer operations
4. **Thread Management**: Use separate threads for model inference to avoid UI blocking
5. **Memory**: Pre-allocate input/output buffers for faster processing

## Troubleshooting

### Model Not Found
- Ensure `behavior_classifier.tflite` is in the correct assets directory
- Check file permissions are readable

### Slow Inference
- Enable GPU/Metal acceleration
- Reduce image preprocessing overhead
- Use batch processing

### Poor Accuracy
- Ensure input images are preprocessed with letterboxing
- Check image resolution (should be 224×224)
- Verify normalization (divide by 255.0)

## Testing

Use the provided test script:

```bash
# Test single image
python src/test.py --image path/to/image.jpg --tflite

# Benchmark speed
python src/test.py --benchmark --tflite

# Test directory
python src/test.py --directory path/to/images --tflite
```

## Support

For issues or questions, refer to:
- TensorFlow Lite Documentation: https://www.tensorflow.org/lite
- Model Details: See `BEHAVIOR_CLASSIFIER_README.md`
