# CNN Image Classifier & AI Vision Lab

A fully-upgraded, production-ready image classification web application powered by **TensorFlow / Keras**, **Flask**, and a modern **vanilla HTML5 / CSS3 / JavaScript** frontend — featuring Explainable AI, multi-source image input, and an interactive live-camera scavenger game.

---

## ✨ Features

| Feature | Description |
|---|---|
| **CNN Classification** | Deep MobileNetV2 feature extractor with custom dense head, classifying mammals, birds, and reptiles |
| **Grad-CAM Heatmaps** | Explainable AI (XAI): highlights CNN attention regions with jet-colormap overlay |
| **File Upload** | Drag-and-drop or browse for JPG/PNG images |
| **Live Webcam Capture** | Snap and classify directly from browser camera — flip front/back camera |
| **Image URL Input** | Paste any public image URL to classify it on-the-fly |
| **Scavenger Game** | Live real-time webcam game challenging you to find target animal classes |
| **Audio Victory Chime** | Zero-dependency Web Audio API synthesized arpeggio on game success |
| **Confetti Celebration** | CSS confetti burst on every scavenger game win |
| **Top 3 Predictions** | Ranked candidates with confidence percentage and animated progress bars |
| **Error Handling** | Extension validation, 16 MB upload limit, corrupt image detection, bad URL rejection |

---

## 📁 Project Structure

```
cnn-image-classifier/
├── dataset/                     # Folder-based image dataset
│   ├── mammals/                 # 20 images: whales, lion, elephant, etc.
│   ├── birds/                   # 18 images: eagles, penguins, macaws, etc.
│   └── reptiles/                # 19 images: crocodile, cobra, chameleon, etc.
├── model/                       # Generated model artifacts
│   ├── cnn_model.keras          # Saved best-checkpoint Keras model
│   ├── class_names.json         # ["birds", "mammals", "reptiles"]
│   └── training_history.png    # Training & validation accuracy / loss curves
├── templates/
│   └── index.html               # Tabbed single-page UI (File, Webcam, URL, Game)
├── static/
│   ├── css/style.css            # Modern responsive styles, game HUD, Grad-CAM toggle
│   └── js/script.js             # Drag & drop, webcam, URL fetch, predict, game loop
├── app.py                       # Flask REST backend — File, Base64, URL inference
├── gradcam.py                   # Grad-CAM XAI engine — gradient computation & overlay
├── train.py                     # CNN training pipeline (Custom CNN or MobileNetV2)
├── fetch_all_animals.py         # Animal image downloader script
├── generate_sample_dataset.py   # Geometric shape synthetic dataset generator
├── verify_endpoints.py          # API verification test suite
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🗂️ Dataset Structure

```
dataset/
├── mammals/         # Whales, dolphins, lions, elephants, bears...
├── birds/           # Eagles, penguins, macaws, puffins, falcons...
└── reptiles/        # Crocodiles, chameleons, cobras, geckos, sea turtles...
```

You can use any folder-based dataset — each sub-folder becomes a class. Minimum **15+ images per class** for reliable training.

Generate a geometric shapes starter dataset:
```bash
python generate_sample_dataset.py
```

Download animal photos automatically:
```bash
python fetch_all_animals.py
```

---

## 🚀 Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

With **MobileNetV2** backbone *(recommended for real-world images)*:
```bash
python train.py --model-type mobilenet --epochs 15
```

With **Custom CNN** backbone *(faster, good for simpler datasets)*:
```bash
python train.py --model-type custom --epochs 20
```

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `mobilenet` | `mobilenet` or `custom` |
| `--dataset` | `dataset` | Path to dataset directory |
| `--epochs` | `20` | Maximum training epochs |
| `--batch-size` | `32` | Batch size |
| `--img-size` | `150` | Input image size (square) |
| `--val-split` | `0.2` | Validation fraction |

### 3. Run the Application
```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌐 API Reference

### `GET /health`
Returns model status, loaded classes, and supported features.

### `GET /classes`
Returns the list of class names the model was trained on.

### `POST /predict`
Unified inference endpoint supporting **three input modes**:

**Mode 1: File Upload**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@eagle.jpg"
```

**Mode 2: Image URL**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/whale.jpg"}'
```

**Mode 3: Webcam Base64 Snapshot**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

**Response Example:**
```json
{
  "success": true,
  "filename": "eagle.jpg",
  "predicted_class": "birds",
  "confidence": 0.967,
  "confidence_percentage": "96.70%",
  "top_3": [
    {"rank": 1, "class": "birds",    "confidence": 0.9670, "percentage": "96.70%", "percentage_value": 96.70},
    {"rank": 2, "class": "reptiles", "confidence": 0.0325, "percentage": "3.25%",  "percentage_value": 3.25},
    {"rank": 3, "class": "mammals",  "confidence": 0.0005, "percentage": "0.05%",  "percentage_value": 0.05}
  ],
  "gradcam_image": "data:image/png;base64,iVBORw0KGgoAAAA..."
}
```

---

## 🎮 Scavenger Hunt Game

1. Click the **Object Hunt** tab.
2. Click **Start Game Camera** to activate your webcam.
3. Read the **Mission** displayed (e.g., *"MAMMALS"*).
4. Point your camera at a matching animal — a photo on your phone screen, a plush toy, or a real animal works too!
5. Press **Check Object** (or enable **Auto-Scan** for hands-free mode).
6. When the model detects a match with **≥ 65% confidence**:
   - Score and streak increment
   - Victory chime plays
   - Confetti bursts across the screen
   - Next mission loads automatically

---

## 🧠 Grad-CAM Explainability

After classification, the interface shows a **Grad-CAM Attention** tab above the image preview. Toggle between:
- **Original Image**: Your uploaded / captured photo.
- **Grad-CAM Attention**: Heatmap overlay using jet colormap — warmer reds indicate regions the CNN weighted most heavily when making the prediction (e.g., beak shape for birds, fin profile for mammals, scale texture for reptiles).

**Technical details**: Gradients of the top predicted class score are backpropagated to the last convolutional layer (`Conv_1` in MobileNetV2). Channels are pooled and multiplied with feature map activations, ReLU-clamped, normalized to [0,1], and interpolated to image resolution.
