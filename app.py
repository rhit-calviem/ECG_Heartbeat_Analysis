from pathlib import Path

from flask import Flask, render_template, send_from_directory, abort, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "train" / "Normal"
TRAIN_DIR = BASE_DIR / "Data" / "train"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "mobilenetv2_best_finetuned.keras"
MODEL_DISPLAY_NAME = "MobileNetV2 (fine-tuned)"

_model = None
_class_names = [
    "Abnormal_Heartbeat",
    "History_Myocardial_Infarction",
    "Myocardial_Infarction",
    "Normal",
]

def get_model():
    global _model
    if _model is None:
        if not DEFAULT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {DEFAULT_MODEL_PATH}")
        _model = tf.keras.models.load_model(str(DEFAULT_MODEL_PATH))
    return _model

def get_model_input_size(model) -> tuple:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    _, h, w, _ = input_shape
    return int(h), int(w)

def preprocess_image_for_model(pil_image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    image = pil_image.convert("RGB").resize(target_size)
    array = np.array(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def find_one_sample_per_class() -> list[dict]:
    results = []
    for class_name in _class_names:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            continue
        chosen = None
        for candidate in sorted(class_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                chosen = candidate
                break
        if chosen is not None:
            rel_path = str(chosen.relative_to(TRAIN_DIR)).replace("\\", "/")
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

        return render_template("index.html", image_name=image_name, model_name=MODEL_DISPLAY_NAME)

    @app.route("/run-model", methods=["GET"])
    def run_model_page():
        if not TRAIN_DIR.exists():
            abort(500, description=f"Train directory not found: {TRAIN_DIR}")
        samples = find_one_sample_per_class()
        # Try to fetch input size for display
        try:
            h, w = get_model_input_size(get_model())
            input_size = f"{w}×{h}"
        except Exception:
            input_size = None
        return render_template(
            "run_model.html",
            samples=samples,
            class_names=_class_names,
            model_name=MODEL_DISPLAY_NAME,
            input_size=input_size,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        model = get_model()
        h, w = get_model_input_size(model)
        image_source = request.form.get("image_source")
        pil_image = None

        if image_source == "sample":
            rel_path = request.form.get("sample_path")
            if not rel_path:
                raise BadRequest("No sample selected")
            image_path = TRAIN_DIR / rel_path
            if not image_path.exists():
                raise BadRequest("Selected sample not found")
            pil_image = Image.open(image_path)
        elif image_source == "upload":
            upload = request.files.get("upload_image")
            if upload is None or upload.filename == "":
                raise BadRequest("No file uploaded")
            pil_image = Image.open(upload.stream)
        else:
            raise BadRequest("Invalid image source")

        input_tensor = preprocess_image_for_model(pil_image, (w, h))
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_index = int(np.argmax(probs))
        pred_class = _class_names[pred_index] if pred_index < len(_class_names) else str(pred_index)

        selected_sample_path = request.form.get("sample_path") if image_source == "sample" else None
        return render_template(
            "prediction.html",
            predicted_class=pred_class.replace("_", " "),
            probabilities=list(map(float, probs)),
            class_names=[name.replace("_", " ") for name in _class_names],
            selected_sample_path=selected_sample_path,
            model_name=MODEL_DISPLAY_NAME,
        )

    @app.route("/data/<path:filename>")
    def data_file(filename: str):
        # Serve files safely from the dataset directory
        return send_from_directory(DATA_DIR, filename)

    @app.route("/dataset/<path:filename>")
    def dataset_file(filename: str):
        # Serve from full train directory across class subfolders
        return send_from_directory(TRAIN_DIR, filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

    


