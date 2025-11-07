# ECG Heartbeat Analysis

**Authors:** Peter Borger, Matteo Calviello, Connor O'Connell

## Overview

This project implements a web-based ECG (electrocardiogram) image classification system that can detect and classify different heart conditions from ECG images. The application provides an interactive interface where users can select from multiple trained deep learning models to analyze ECG images and receive real-time predictions.

### Features

- **Multi-Model Support**: Choose from 4 different trained models:
  - **MobileNetV2** (Keras, 224×224) - Lightweight and efficient
  - **Small 2-Layer CNN** (Keras, 299×299) - Custom baseline model
  - **InceptionV3** (PyTorch, 299×299) - Transfer learning with fine-tuning
  - **VGG16** (PyTorch, 224×224) - Deep convolutional network

- **ECG Classification**: Detects 4 different heart conditions:
  - Abnormal Heartbeat
  - History of Myocardial Infarction
  - Myocardial Infarction (Heart Attack)
  - Normal

- **Interactive Web Interface**:
  - Select from random test dataset samples
  - Upload your own ECG images
  - View prediction probabilities for all classes
  - See the original image alongside results

- **Model-Specific Preprocessing**: Each model uses its trained preprocessing pipeline (rescaling, ImageNet normalization, or standard normalization)

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch will be installed with CPU support. If you have a GPU and want to use it, install PyTorch with CUDA support instead:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Dataset Structure

Ensure your dataset is organized as follows:

```
Project/
├── Data/
│   ├── train/
│   │   ├── Abnormal_Heartbeat/
│   │   ├── History_Myocardial_Infarction/
│   │   ├── Myocardial_Infarction/
│   │   └── Normal/
│   └── test/
│       ├── Abnormal_Heartbeat/
│       ├── History_Myocardial_Infarction/
│       ├── Myocardial_Infarction/
│       └── Normal/
└── models/
    ├── mobilenetv2_best_stage1.keras
    ├── Small_2CNN_best.keras
    ├── inceptionv3_ecg_full.pth
    └── vgg16_ecg_full.pth
```

### 5. Run the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5555` (or `http://localhost:5555`)

## Usage

### Running Predictions

1. **Open your web browser** and navigate to `http://localhost:5555`

2. **Navigate to "Run Model"** page

3. **Select a Model**: Choose from the available models (MobileNetV2, Small 2-Layer CNN, InceptionV3, or VGG16)

4. **Choose an Input Method**:
   - **Option A - Test Sample**: Select one of the random test images displayed (one per class)
   - **Option B - Upload Image**: Upload your own ECG image (JPG, JPEG, or PNG format)

5. **Run Prediction**: Click the button to run the model

6. **View Results**:
   - Predicted class with confidence
   - Probability distribution across all 4 classes
   - Original image preview

### Model Details

| Model | Framework | Input Size | Preprocessing | Accuracy* |
|-------|-----------|------------|---------------|-----------|
| MobileNetV2 | Keras/TensorFlow | 224×224 | Rescale (0-1) | ~77% |
| Small 2-Layer CNN | Keras/TensorFlow | 299×299 | Rescale (0-1) | ~98% |
| InceptionV3 | PyTorch | 299×299 | ImageNet normalization | ~100% |
| VGG16 | PyTorch | 224×224 | Standard normalization | ~97% |

*Accuracy on test set

## Project Structure

```
Project/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Data/                          # ECG image dataset
│   ├── train/                     # Training images
│   └── test/                      # Test images
├── models/                        # Trained model files
│   ├── mobilenetv2_best_stage1.keras
│   ├── Small_2CNN_best.keras
│   ├── inceptionv3_ecg_full.pth
│   └── vgg16_ecg_full.pth
├── templates/                     # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── run_model.html
│   └── prediction.html
├── static/                        # Static assets (CSS, images)
└── Notebooks/                     # Jupyter notebooks for model training
```

## Technical Implementation

### Backend (Flask)
- **Multi-framework support**: Seamlessly loads both Keras/TensorFlow and PyTorch models
- **Model caching**: Models are loaded once and cached for performance
- **Dynamic preprocessing**: Automatically applies the correct preprocessing pipeline based on the selected model
- **Image handling**: Supports both file uploads and dataset samples

### Frontend
- **Responsive design**: Clean, modern interface that works on all devices
- **Interactive model selection**: Visual feedback for model choice
- **Real-time predictions**: Fast inference with immediate results
- **Image preview**: Display both uploaded and sample images

## Troubleshooting

### Port Already in Use
If port 5555 is already in use, modify the port in `app.py`:

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Change to desired port
```

### Model Files Not Found
Ensure all model files are in the `models/` directory. Missing models will not appear in the model selector but won't cause the application to crash.

### Memory Issues
If you encounter memory issues with PyTorch models:
- Ensure only CPU support is installed if you don't have a GPU
- The models are loaded lazily and cached, so only the selected model is loaded into memory

## Development

### Training New Models
Jupyter notebooks for model training are available in the `Notebooks/` directory. Each notebook contains:
- Data preprocessing and augmentation
- Model architecture definition
- Training loops with callbacks
- Evaluation metrics and visualizations

### Adding New Models
To add a new model:

1. Save your trained model to the `models/` directory
2. Update `MODEL_CONFIGS` in `app.py`:

```python
MODEL_CONFIGS = {
    "your_model": {
        "display_name": "Your Model Name",
        "path": MODELS_DIR / "your_model.keras",
        "framework": "keras",  # or "pytorch"
        "input_size": (224, 224),
        "preprocessing": "rescale",  # "rescale", "imagenet", or "standard"
    },
}
```

3. If using PyTorch, add a loader function similar to `load_pytorch_inceptionv3()`
