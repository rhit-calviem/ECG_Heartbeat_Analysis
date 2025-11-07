from pathlib import Path
import random
import base64
from io import BytesIO

from flask import Flask, render_template, send_from_directory, abort, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models as torch_models


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "train" / "Normal"
TRAIN_DIR = BASE_DIR / "Data" / "train"
TEST_DIR = BASE_DIR / "Data" / "test"
MODELS_DIR = BASE_DIR / "models"

_class_names = [
    "Abnormal_Heartbeat",
    "History_Myocardial_Infarction",
    "Myocardial_Infarction",
    "Normal",
]

# Model configurations
MODEL_CONFIGS = {
    "mobilenetv2": {
        "display_name": "MobileNetV2 (fine-tuned)",
        "path": MODELS_DIR / "mobilenetv2_best_stage1.keras",
        "framework": "keras",
        "input_size": (224, 224),
        "preprocessing": "rescale",  # rescale to 0-1
    },
    "small_2cnn": {
        "display_name": "Small 2-Layer CNN",
        "path": MODELS_DIR / "Small_2CNN_best.keras",
        "framework": "keras",
        "input_size": (299, 299),
        "preprocessing": "rescale",
    },
    "inceptionv3": {
        "display_name": "InceptionV3 (fine-tuned)",
        "path": MODELS_DIR / "inceptionv3_ecg_full.pth",
        "framework": "pytorch",
        "input_size": (299, 299),
        "preprocessing": "imagenet",  # ImageNet normalization
    },
    "vgg16": {
        "display_name": "VGG16 (fine-tuned)",
        "path": MODELS_DIR / "vgg16_ecg_full.pth",
        "framework": "pytorch",
        "input_size": (224, 224),
        "preprocessing": "standard",  # mean=[0.5], std=[0.5]
    },
}

_loaded_models = {}

def load_pytorch_inceptionv3(model_path: Path, num_classes: int = 4):
    """Load InceptionV3 PyTorch model with custom classifier."""
    # Try loading as state dict first, then as full model
    loaded = torch.load(str(model_path), map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(loaded, dict):
        # It's a state dict
        model = torch_models.inception_v3(weights=None)
        
        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # InceptionV3 also has an auxiliary classifier
        num_aux_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_classes)
        
        model.load_state_dict(loaded)
    else:
        # It's a full model
        model = loaded
    
    model.eval()
    return model

def load_pytorch_vgg16(model_path: Path, num_classes: int = 4):
    """Load VGG16 PyTorch model with custom classifier."""
    # Try loading as state dict first, then as full model
    loaded = torch.load(str(model_path), map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(loaded, dict):
        # It's a state dict
        model = torch_models.vgg16(weights=None)
        
        # Replace the classifier head
        model.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        model.load_state_dict(loaded)
    else:
        # It's a full model
        model = loaded
    
    model.eval()
    return model

def get_model(model_key: str):
    """Load and cache a model based on its key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    
    if model_key in _loaded_models:
        return _loaded_models[model_key]
    
    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if config["framework"] == "keras":
        model = tf.keras.models.load_model(str(model_path))
    elif config["framework"] == "pytorch":
        if model_key == "inceptionv3":
            model = load_pytorch_inceptionv3(model_path)
        elif model_key == "vgg16":
            model = load_pytorch_vgg16(model_path)
        else:
            raise ValueError(f"Unknown PyTorch model: {model_key}")
    else:
        raise ValueError(f"Unknown framework: {config['framework']}")
    
    _loaded_models[model_key] = model
    return model

def preprocess_image_for_model(pil_image: Image.Image, model_key: str) -> np.ndarray:
    """Preprocess image according to model requirements."""
    config = MODEL_CONFIGS[model_key]
    target_size = config["input_size"]
    preprocessing = config["preprocessing"]
    
    # Resize and convert to RGB
    image = pil_image.convert("RGB").resize(target_size)
    array = np.array(image).astype("float32")
    
    if preprocessing == "rescale":
        # Simple rescaling to [0, 1]
        array = array / 255.0
    elif preprocessing == "imagenet":
        # ImageNet normalization
        array = array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        array = (array - mean) / std
    elif preprocessing == "standard":
        # Standard normalization [-1, 1]
        array = (array / 255.0 - 0.5) / 0.5
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    
    # Add batch dimension
    if config["framework"] == "keras":
        array = np.expand_dims(array, axis=0)
    else:  # pytorch
        # Convert to (C, H, W) format and add batch dimension
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)
    
    return array

def find_one_sample_per_class() -> list[dict]:
    """Find one random test image per class."""
    results = []
    for class_name in _class_names:
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            continue
        
        # Collect all valid image files
        candidates = [
            candidate for candidate in class_dir.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        
        if candidates:
            # Choose a random image
            chosen = random.choice(candidates)
            rel_path = str(chosen.relative_to(TEST_DIR)).replace("\\", "/")
            results.append({
                "class_name": class_name,
                "rel_path": rel_path,
            })
    return results


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        # Choose a default image that exists in the dataset
        if not DATA_DIR.exists():
            abort(500, description=f"Dataset directory not found: {DATA_DIR}")

        # Pick the first jpg file we find
        image_name = None
        for candidate in sorted(DATA_DIR.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_name = candidate.name
                break

        if image_name is None:
            abort(500, description="No images found in dataset directory")

        return render_template("index.html", image_name=image_name)

    @app.route("/run-model", methods=["GET"])
    def run_model_page():
        if not TEST_DIR.exists():
            abort(500, description=f"Test directory not found: {TEST_DIR}")
        samples = find_one_sample_per_class()
        
        # Build list of available models
        available_models = []
        for key, config in MODEL_CONFIGS.items():
            if config["path"].exists():
                h, w = config["input_size"]
                available_models.append({
                    "key": key,
                    "display_name": config["display_name"],
                    "input_size": f"{w}×{h}",
                    "framework": config["framework"].capitalize(),
                })
        
        return render_template(
            "run_model.html",
            samples=samples,
            class_names=_class_names,
            available_models=available_models,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        # Get selected model
        model_key = request.form.get("model_key", "mobilenetv2")
        if model_key not in MODEL_CONFIGS:
            raise BadRequest(f"Invalid model: {model_key}")
        
        config = MODEL_CONFIGS[model_key]
        model = get_model(model_key)
        
        # Get image
        image_source = request.form.get("image_source")
        pil_image = None
        selected_sample_path = None
        uploaded_image_data = None

        if image_source == "sample":
            rel_path = request.form.get("sample_path")
            if not rel_path:
                raise BadRequest("No sample selected")
            image_path = TEST_DIR / rel_path
            if not image_path.exists():
                raise BadRequest("Selected sample not found")
            pil_image = Image.open(image_path)
            selected_sample_path = rel_path
        elif image_source == "upload":
            upload = request.files.get("upload_image")
            if upload is None or upload.filename == "":
                raise BadRequest("No file uploaded")
            pil_image = Image.open(upload.stream)
            
            # Convert uploaded image to base64 data URL for display
            img_copy = pil_image.copy()
            buffered = BytesIO()
            img_copy.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            uploaded_image_data = f"data:image/png;base64,{img_str}"
        else:
            raise BadRequest("Invalid image source")

        # Preprocess and predict
        input_tensor = preprocess_image_for_model(pil_image, model_key)
        
        if config["framework"] == "keras":
            probs = model.predict(input_tensor, verbose=0)[0]
        else:  # pytorch
            with torch.no_grad():
                input_tensor_torch = torch.from_numpy(input_tensor).float()
                outputs = model(input_tensor_torch)
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
        
        pred_index = int(np.argmax(probs))
        pred_class = _class_names[pred_index] if pred_index < len(_class_names) else str(pred_index)

        return render_template(
            "prediction.html",
            predicted_class=pred_class.replace("_", " "),
            probabilities=list(map(float, probs)),
            class_names=[name.replace("_", " ") for name in _class_names],
            selected_sample_path=selected_sample_path,
            uploaded_image_data=uploaded_image_data,
            model_name=config["display_name"],
        )

    @app.route("/data/<path:filename>")
    def data_file(filename: str):
        # Serve files safely from the dataset directory
        return send_from_directory(DATA_DIR, filename)

    @app.route("/dataset/<path:filename>")
    def dataset_file(filename: str):
        # Serve from test directory across class subfolders
        return send_from_directory(TEST_DIR, filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)

    


