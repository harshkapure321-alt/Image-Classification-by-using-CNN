"""
Flask Web Application for CNN Image Classification.
Provides web interface, RESTful prediction API with Grad-CAM Explainable AI,
Webcam support, Image URL fetching, and Scavenger Game functionality.
"""

import os
import io
import json
import base64
import logging
import requests
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

# Grad-CAM Explainable AI integration
from gradcam import generate_gradcam_overlay_base64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Security & Upload Configuration
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Image dimensions matching train.py
IMG_SIZE = 150

# Global variables for model and classes
model = None
class_names = []


def load_classifier_model():
    """Loads the trained Keras model and class names JSON."""
    global model, class_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        logger.warning(
            f"Model artifacts not found at '{MODEL_DIR}'. "
            "Please train the model using 'python train.py' before making predictions."
        )
        model = None
        class_names = []
        return False

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        logger.info(f"Model loaded successfully with {len(class_names)} classes: {class_names}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        model = None
        class_names = []
        return False


def allowed_file(filename):
    """Verifies that filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_bytes(image_bytes):
    """
    Validates and preprocesses image bytes:
    - Verifies format with PIL
    - Converts to RGB
    - Resizes to (IMG_SIZE, IMG_SIZE)
    - Returns (input_tensor, original_pil_image)
    """
    try:
        img_verify = Image.open(io.BytesIO(image_bytes))
        img_verify.verify()

        # Reopen after verify()
        original_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        resized_pil = original_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        # Convert to numpy array with float32 format
        img_array = np.array(resized_pil, dtype=np.float32)
        # Expand dimensions to create batch: (1, 150, 150, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        return input_tensor, original_pil
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image: {str(e)}")


@app.route("/")
def index():
    """Serves the main single-page web interface."""
    is_ready = model is not None and len(class_names) > 0
    return render_template("index.html", model_ready=is_ready, classes=class_names)


@app.route("/health", methods=["GET"])
def health():
    """Health and model status endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "classes": class_names,
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE}",
        "features": ["gradcam", "webcam", "url_fetch", "scavenger_game"]
    }), 200


@app.route("/classes", methods=["GET"])
def get_classes():
    """Returns available classes for game and UI targets."""
    return jsonify({"classes": class_names}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Unified inference endpoint:
    Accepts:
    1. multipart/form-data with file field 'image' or 'file' (file upload or webcam blob)
    2. JSON payload {"image_base64": "data:image/jpeg;base64,..."} (webcam dataURL)
    3. JSON payload {"url": "https://..."} (remote image URL)
    Returns:
    - predicted_class, confidence, top_3 ranking
    - gradcam_image: base64 Data URI highlighting key attention regions
    """
    global model, class_names

    if model is None or len(class_names) == 0:
        if not load_classifier_model():
            return jsonify({
                "error": "Trained model not found on server. Please run 'python train.py' first."
            }), 503

    image_bytes = None
    filename = "input_image.jpg"

    # Case 1: JSON payload (URL or Base64 Webcam snapshot)
    if request.is_json:
        data = request.get_json() or {}

        # 1a. Direct Image URL
        if "url" in data and data["url"].strip():
            target_url = data["url"].strip()
            if not (target_url.startswith("http://") or target_url.startswith("https://")):
                return jsonify({"error": "Invalid URL protocol. Must begin with http:// or https://"}), 400
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageClassifierBot/1.0"}
                resp = requests.get(target_url, headers=headers, timeout=9)
                if resp.status_code != 200:
                    return jsonify({"error": f"Failed to download image from URL (HTTP {resp.status_code})"}), 400
                image_bytes = resp.content
                filename = target_url.split("/")[-1].split("?")[0] or "url_image.jpg"
            except Exception as e:
                return jsonify({"error": f"Error fetching image URL: {str(e)}"}), 400

        # 1b. Base64 Image (from Webcam snapshot)
        elif "image_base64" in data and data["image_base64"]:
            b64_str = data["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64_str)
                filename = "webcam_snapshot.jpg"
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400

    # Case 2: Multipart file upload (Dropzone or FormData blob)
    if image_bytes is None:
        upload_file = request.files.get("image") or request.files.get("file")
        if not upload_file or upload_file.filename == "":
            return jsonify({"error": "No image provided. Please upload a file, capture from camera, or provide a URL."}), 400

        raw_filename = secure_filename(upload_file.filename) or "uploaded_image.jpg"
        if not allowed_file(raw_filename):
            return jsonify({
                "error": f"Invalid file format: '{raw_filename}'. Allowed formats are JPG, JPEG, and PNG."
            }), 400

        image_bytes = upload_file.read()
        filename = raw_filename

    if not image_bytes:
        return jsonify({"error": "Image payload is empty."}), 400

    try:
        # Preprocess image into tensor & keep original PIL instance for Grad-CAM
        input_tensor, original_pil = preprocess_image_bytes(image_bytes)

        # Run model inference
        predictions = model.predict(input_tensor, verbose=0)[0]

        # Top 1 Prediction
        top_idx = int(np.argmax(predictions))
        top_class = class_names[top_idx] if top_idx < len(class_names) else f"Class_{top_idx}"
        top_confidence = float(predictions[top_idx])

        # Top 3 Predictions
        top_k = min(3, len(predictions))
        sorted_indices = np.argsort(predictions)[::-1][:top_k]

        top_3 = []
        for rank, idx in enumerate(sorted_indices, start=1):
            c_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            prob = float(predictions[idx])
            top_3.append({
                "rank": rank,
                "class": c_name,
                "confidence": round(prob, 4),
                "percentage": f"{prob * 100:.2f}%",
                "percentage_value": round(prob * 100, 2)
            })

        # Generate Grad-CAM Explainable AI Heatmap
        gradcam_uri = generate_gradcam_overlay_base64(
            model=model,
            img_array=input_tensor,
            original_pil_img=original_pil,
            pred_index=top_idx,
            alpha=0.45
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "predicted_class": top_class,
            "confidence": round(top_confidence, 4),
            "confidence_percentage": f"{top_confidence * 100:.2f}%",
            "top_3": top_3,
            "gradcam_image": gradcam_uri
        }), 200

    except ValueError as ve:
        logger.warning(f"Bad image payload: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal prediction failure: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the 16 MB maximum upload limit."}), 413


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "An internal server error occurred."}), 500


# Initial model load on startup
load_classifier_model()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
