# ECG Heartbeat Analysis

**Authors:** Peter Borger, Matteo Calviello, Connor O'Connell

## Overview

This project implements a web-based ECG (electrocardiogram) image classification system that can detect and classify different heart conditions from ECG images. The application provides an interactive interface where users can select from multiple trained deep learning models to analyze ECG images and receive real-time predictions.

### Features

  - **Multi-Model Support**: Choose from 4 different trained models:

      - **MobileNetV2 (fine-tuned)**
      - **Small 2-Layer CNN** 
      - **InceptionV3 (fine-tuned)**
      - **VGG16 (fine-tuned)**

  - **ECG Classification**: Detects 4 different heart conditions:

      - Abnormal Heartbeat
      - History of Myocardial Infarction
      - Myocardial Infarction (Heart Attack)
      - Normal

  - **Model-Specific Preprocessing**: Each model uses its trained preprocessing pipeline (rescaling, ImageNet normalization, or standard normalization)

##  Live Demo (Fly.io)

You can try this application live without any installation\! It is deployed as a Fly.io:

**[👉 Try the Live ECG Classifier Here]https://ecg-heartbeat-classifier.fly.dev/)**

To make this deployment possible, the application is containerized using **Docker**. The `Dockerfile` and associated configuration files are included in the repository to define the environment for the Hugging Face platform.

## Installation & Setup - For Local Use

### 1\. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2\. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Verify Dataset Structure

Note that the Data is not in this reporistory. Here is the link to the Kaggle Dataset to download: [DataSet Here](https://www.kaggle.com/datasets/evilspirit05/ecg-analysis)
Ensure your dataset is organized as follows (relative to `app.py`):

```
.
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
├── models/
│   ├── mobilenetv2_best_stage1.keras
│   ├── Small_2CNN_best.keras
│   ├── inceptionv3_ecg_full.pth
│   └── vgg16_ecg_full.pth
├── app.py
├── requirements.txt
└── ... (other files)
```

### 5\. Run the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:7860` (or `http://localhost:7860`)

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

1.  Save your trained model to the `models/` directory
2.  Update `MODEL_CONFIGS` in `app.py` with a new entry:

<!-- end list -->

```python
# (in app.py)
MODELS_DIR = BASE_DIR / "models"

MODEL_CONFIGS = {
    # ... existing models
    "your_model_key": {
        "display_name": "Your Model Name",
        "path": MODELS_DIR / "your_model_file.keras", # or .pth
        "framework": "keras",  # or "pytorch"
        "input_size": (224, 224),
        "preprocessing": "rescale",  # "rescale", "imagenet", or "standard"
    },
}
```
