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

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project
```

### 2. Create a Virtual Environment

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
