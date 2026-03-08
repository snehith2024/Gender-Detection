# Gender Detection

A deep learning-based project for detecting gender from facial images using Convolutional Neural Networks (CNN). This project includes training, image-based detection, and real-time webcam detection capabilities.

## 🎯 Project Overview

Gender Detection uses a trained Keras neural network model to classify faces as either **male** or **female**. The project provides multiple ways to use the trained model:
- Train a custom model on your own dataset
- Detect gender from static images
- Detect gender in real-time from your webcam

## 📁 Project Structure

```
Gender-Detection/
├── train.py                          # Script to train the gender detection model
├── detect_gender_image.py            # Script to detect gender in static images
├── detect_gender_webcam.py           # Script to detect gender via webcam
├── gender_detection.keras            # Pre-trained Keras model
├── gender_dataset_face/              # Training dataset directory
│   ├── man/                          # Images of male faces
│   └── woman/                        # Images of female faces
├── plot.png                          # Training history visualization
└── README.md                         # This file
```

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.6+
- Keras
- OpenCV (cv2)
- cvlib
- NumPy
- scikit-learn
- Matplotlib

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/snehith2024/Gender-Detection.git
cd Gender-Detection
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install tensorflow keras opencv-python cvlib scikit-learn matplotlib numpy
```

## 🚀 Usage

### Train the Model
To train a new model on the gender dataset:

```bash
python train.py
```

**Training Parameters (in train.py):**
- **Epochs:** 100
- **Learning Rate:** 1e-3
- **Batch Size:** 64
- **Image Dimensions:** 96x96x3

**What the script does:**
1. Loads images from `gender_dataset_face/man` and `gender_dataset_face/woman`
2. Preprocesses images (resizing, normalization)
3. Splits data into training and validation sets
4. Trains a CNN model with the following architecture:
   - Convolutional layers with batch normalization
   - Max pooling layers
   - Dense layers with dropout regularization
5. Saves the trained model as `gender_detection.keras`
6. Generates training history plot (`plot.png`)

### Detect Gender in Images
To detect gender in a static image:

```bash
python detect_gender_image.py path/to/image.jpg
```

**Example:**
```bash
python detect_gender_image.py sample.jpg
```

**What the script does:**
1. Loads the pre-trained model (`gender_detection.keras`)
2. Detects faces in the image using cvlib
3. Classifies each detected face as male or female
4. Displays confidence percentages
5. Draws rectangles around detected faces with labels
6. Shows the result in a window

### Real-time Webcam Detection
To detect gender in real-time from your webcam:

```bash
python detect_gender_webcam.py
```

**What the script does:**
1. Loads the pre-trained model (`gender_detection.keras`)
2. Accesses your webcam
3. Continuously detects faces and classifies gender
4. Displays live video with gender labels and confidence scores
5. Press 'q' key to exit

## 🧠 Model Architecture

The CNN model consists of:
- **Input Layer:** 96x96x3 (RGB images)
- **Convolutional Blocks:** Multiple Conv2D layers with BatchNormalization and ReLU activation
- **Pooling Layers:** MaxPooling2D for downsampling
- **Dropout Layers:** For regularization (to prevent overfitting)
- **Dense Layers:** Fully connected layers for classification
- **Output Layer:** Softmax activation with 2 outputs (man/woman)

## 📊 Model Performance

The model trained on the gender_dataset_face achieves robust gender classification:
- **Classes:** Man, Woman
- **Input Size:** 96x96 pixels
- **Model Format:** Keras (.keras)
- **Face Detection:** CVLib (based on YOLO)

## 🎨 Key Features

✅ **Face Detection:** Automatic detection of faces in images/webcam using CVLib  
✅ **Gender Classification:** Binary classification (Male/Female)  
✅ **Confidence Scores:** Displays prediction confidence as percentage  
✅ **Real-time Processing:** Live webcam detection with minimal latency  
✅ **Easy Training:** Simple script to retrain on custom datasets  
✅ **Visualization:** Generated training history plots  

## 📝 Dataset Format

Your training dataset should be organized as:
```
gender_dataset_face/
├── man/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
└── woman/
    ├── face_1.jpg
    ├── face_2.jpg
    └── ...
```

## 🔍 How It Works

### Training Process:
1. **Data Loading:** Images are loaded from the organized dataset folders
2. **Preprocessing:** Images are resized to 96x96 and normalized (values 0-1)
3. **Augmentation:** Optional data augmentation to improve generalization
4. **Model Training:** CNN learns to extract gender-related features
5. **Evaluation:** Model performance is validated on test dataset
6. **Saving:** Model is saved in Keras format for later use

### Inference Process:
1. **Face Detection:** CVLib detects all faces in input image/frame
2. **Face Cropping:** Each detected face is extracted
3. **Preprocessing:** Face is resized to 96x96 and normalized
4. **Prediction:** Model predicts gender and confidence
5. **Display:** Results are visualized with bounding boxes and labels

## ⚙️ Configuration

You can modify training parameters in `train.py`:
- `epochs` - Number of training epochs
- `lr` - Learning rate for optimizer
- `batch_size` - Batch size for training
- `img_dims` - Input image dimensions

## 🐛 Troubleshooting

**Issue: Webcam not detected**
- Ensure camera permissions are granted
- Check if another application is using the webcam
- Try using camera index other than 0

**Issue: "Module not found" error**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify virtual environment is activated

**Issue: Poor gender prediction accuracy**
- Ensure adequate lighting in the image/webcam
- Verify face is clearly visible and front-facing
- Consider retraining on a larger, more diverse dataset

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.6+ | Deep learning framework |
| keras | Latest | Neural network API |
| opencv-python | Latest | Image processing |
| cvlib | Latest | Face detection |
| numpy | Latest | Numerical computing |
| scikit-learn | Latest | Data splitting |
| matplotlib | Latest | Visualization |

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🔗 Links

- **GitHub Repository:** [Gender-Detection](https://github.com/snehith2024/Gender-Detection)
- **Author:** snehith2024

## 💡 Tips & Best Practices

1. **For Best Results:**
   - Use well-lit environments
   - Ensure faces are clearly visible
   - Use frontal face images for training

2. **Performance Optimization:**
   - The model runs efficiently on CPU, but GPU acceleration is recommended for faster predictions
   - For large batch processing, consider batching predictions

3. **Accuracy Improvement:**
   - Collect more diverse training data
   - Augment images during training
   - Fine-tune model hyperparameters
   - Use transfer learning from pre-trained models

## ⚡ Quick Start Example

```bash
# Clone and setup
git clone https://github.com/snehith2024/Gender-Detection.git
cd Gender-Detection
pip install tensorflow keras opencv-python cvlib scikit-learn matplotlib numpy

# Run webcam detection (uses pre-trained model)
python detect_gender_webcam.py

# Or detect gender in an image
python detect_gender_image.py yourimage.jpg

# Or train a new model
python train.py
```

## 📞 Support

For issues, questions, or suggestions, please create an issue on the GitHub repository.

---

**Last Updated:** March 2026  
**Status:** Active and Maintained
