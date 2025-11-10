from pathlib import Path
import random
import base64
from io import BytesIO
import os

from flask import Flask, render_template, send_from_directory, abort, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Environment and reproducibility
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(0)
np.random.seed(0)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "train" / "Normal"
TRAIN_DIR = BASE_DIR / "Data" / "train"
TEST_DIR = BASE_DIR / "Data" / "test"
MODELS_DIR = BASE_DIR / "models"

# Classes
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
        "preprocessing": "rescale",
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
        "preprocessing": "imagenet",
    },
    "vgg16": {
        "display_name": "VGG16 (fine-tuned)",
        "path": MODELS_DIR / "vgg16_ecg_full.pth",
        "framework": "pytorch",
        "input_size": (224, 224),
        "preprocessing": "standard",
    },
}

_loaded_models = {}

# Model loading functions
def load_pytorch_inceptionv3(model_path: Path, num_classes: int = 4):
    import torch
    import torch.nn as nn
    from torchvision import models as torch_models

    loaded = torch.load(str(model_path), map_location=torch.device("cpu"), weights_only=False)
    
    if isinstance(loaded, dict):
        model = torch_models.inception_v3(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        num_aux_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_classes)
        model.load_state_dict(loaded)
    else:
        model = loaded
    
    model.eval()
    return model

def load_pytorch_vgg16(model_path: Path, num_classes: int = 4):
    import torch
    import torch.nn as nn
    from torchvision import models as torch_models

    loaded = torch.load(str(model_path), map_location=torch.device("cpu"), weights_only=False)
    
    if isinstance(loaded, dict):
        model = torch_models.vgg16(weights=None)
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
        model = loaded
    
    model.eval()
    return model

def get_model(model_key: str):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    
    if model_key in _loaded_models:
        return _loaded_models[model_key]
    
    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if config["framework"] == "keras":
        import tensorflow as tf
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

# Image preprocessing
def preprocess_image_for_model(pil_image: Image.Image, model_key: str) -> np.ndarray:
    config = MODEL_CONFIGS[model_key]
    target_size = config["input_size"]
    preprocessing = config["preprocessing"]
    
    image = pil_image.convert("RGB").resize(target_size)
    array = np.array(image).astype("float32")
    
    if preprocessing == "rescale":
        array = array / 255.0
    elif preprocessing == "imagenet":
        array = array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        array = (array - mean) / std
    elif preprocessing == "standard":
        array = (array / 255.0 - 0.5) / 0.5
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    
    if config["framework"] == "keras":
        array = np.expand_dims(array, axis=0)
    else:
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)
    
    return array

# Sample selection
def find_one_sample_per_class() -> list[dict]:
    results = []
    for class_name in _class_names:
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            continue
        candidates = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if candidates:
            chosen = random.choice(candidates)
            rel_path = str(chosen.relative_to(TEST_DIR)).replace("\\", "/")
            results.append({"class_name": class_name, "rel_path": rel_path})
    return results

# Flask app
def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        if not DATA_DIR.exists():
            abort(500, description=f"Dataset directory not found: {DATA_DIR}")
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
        return render_template("run_model.html", samples=samples, class_names=_class_names, available_models=available_models)

    @app.route("/predict", methods=["POST"])
    def predict():
        model_key = request.form.get("model_key", "mobilenetv2")
        if model_key not in MODEL_CONFIGS:
            raise BadRequest(f"Invalid model: {model_key}")
        config = MODEL_CONFIGS[model_key]
        model = get_model(model_key)

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
            img_copy = pil_image.copy()
            buffered = BytesIO()
            img_copy.save(buffered, format="PNG")
            uploaded_image_data = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        else:
            raise BadRequest("Invalid image source")

        input_tensor = preprocess_image_for_model(pil_image, model_key)
        if config["framework"] == "keras":
            probs = model.predict(input_tensor, verbose=0)[0]
        else:
            import torch
            with torch.no_grad():
                input_tensor_torch = torch.from_numpy(input_tensor).float()
                outputs = model(input_tensor_torch)
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
        return send_from_directory(DATA_DIR, filename)

    @app.route("/dataset/<path:filename>")
    def dataset_file(filename: str):
        return send_from_directory(TEST_DIR, filename)

    @app.route("/health")
    def health():
        return {"status": "ok"}, 200

    return app

# Create Flask app
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
