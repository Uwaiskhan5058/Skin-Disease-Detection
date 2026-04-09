# 🩺 AI Dermatology Assistant

> **AI-powered skin disease detection using deep learning with explainable AI (Grad-CAM)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Overview

An AI-powered dermatology assistant that detects **7 types of skin diseases** from dermatoscopic images using a **MobileNetV2** deep learning model trained on the **HAM10000 dataset** (10,000+ clinical images). Features an interactive web interface with **Grad-CAM heatmap visualization** for explainable predictions.

### Key Features

- 🔬 **Deep Learning Classification** — MobileNetV2 with transfer learning (ImageNet)
- 🎯 **7 Disease Classes** — Melanoma, BCC, Melanocytic Nevi, and more
- 🔥 **Grad-CAM Explainability** — Visual heatmaps showing influential regions
- 📊 **Top-3 Predictions** — Confidence scores with animated progress bars
- ⚡ **Real-time Inference** — Instant analysis with optimized model loading
- 🎨 **Premium UI** — Dark glassmorphism design with smooth animations
- 📱 **Fully Responsive** — Works on desktop, tablet, and mobile

---

## 🧠 Model Architecture

```
MobileNetV2 (ImageNet, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.5)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(7, Softmax)
```

**Training Configuration:**
- Loss: Categorical Cross-Entropy
- Optimizer: Adam (lr=0.001, with ReduceLROnPlateau)
- Data Augmentation: Rotation, flip, zoom, shift
- Class Weighting: Automated for imbalanced data

---

## 📊 Supported Disease Classes

| Code | Disease | Risk Level |
|------|---------|------------|
| `mel` | Melanoma | 🔴 High |
| `bcc` | Basal Cell Carcinoma | 🔴 High |
| `akiec` | Actinic Keratoses | 🟡 Medium |
| `vasc` | Vascular Lesions | 🟡 Medium |
| `nv` | Melanocytic Nevi (Moles) | 🟢 Low |
| `bkl` | Benign Keratosis | 🟢 Low |
| `df` | Dermatofibroma | 🟢 Low |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/skin-disease-ai.git
cd skin-disease-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash


Open your browser at `http://localhost:5000`

### 3. Train on HAM10000 (Optional)

```bash
# Download HAM10000 dataset from Kaggle first
# Extract to ./data/HAM10000/

python train_model.py --data_dir ./data/HAM10000 --epochs 30
```

---

## 📁 Project Structure

```
skin-disease-ai/
├── app/
│   ├── __init__.py           # Package init
│   ├── main.py               # Flask application & API endpoints
│   ├── model_handler.py      # Model loading & prediction logic
│   ├── gradcam.py            # Grad-CAM heatmap generation
│   ├── disease_info.py       # Disease metadata (7 classes)
│   └── utils.py              # Image preprocessing utilities
├── static/
│   ├── css/style.css         # Premium dark theme design system
│   └── js/app.js             # Frontend logic & animations
├── templates/
│   └── index.html            # Main application template
├── models/                   # Saved model weights (.h5)
├── data/                     # Dataset directory
├── train_model.py            # Model training pipeline
├── requirements.txt          # Python dependencies
├── Procfile                  # Render/Heroku deployment
├── Dockerfile                # Container deployment
├── .gitignore
└── README.md
```

---

## 🌐 Deployment

### Render (Recommended — Free)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repository
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app.main:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Deploy!

### Docker

```bash
docker build -t skin-disease-ai .
docker run -p 5000:5000 skin-disease-ai
```

### HuggingFace Spaces

Upload the project files to a new Gradio/Docker Space on [huggingface.co/spaces](https://huggingface.co/spaces).

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **Backend** | Flask, Flask-CORS |
| **ML Framework** | TensorFlow / Keras |
| **Model** | MobileNetV2 (Transfer Learning) |
| **Explainability** | Grad-CAM |
| **Image Processing** | OpenCV, Pillow |
| **Data** | NumPy, Pandas, scikit-learn |
| **Dataset** | HAM10000 (10,015 images) |
| **Deployment** | Render, Docker, Gunicorn |

---

## ⚠️ Disclaimer

> **This application is for educational and research purposes only.**  
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified dermatologist for skin concerns.

---

## 📝 Resume Description

> *"Developed an AI-powered dermatology assistant using deep learning (MobileNetV2) trained on the HAM10000 dataset (10,000+ dermatoscopic images, 7 disease classes). Built a production-grade web application with Flask featuring a premium dark-themed UI with glassmorphism, real-time image classification, confidence scores, and Grad-CAM explainable AI heatmap visualizations. Implemented transfer learning, data augmentation, class weighting for imbalanced data, and comprehensive evaluation metrics."*

---

## 📄 License

This project is licensed under the MIT License.
