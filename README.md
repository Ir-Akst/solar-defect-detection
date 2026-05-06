# Solar Panel Defect Detection API

## Overview

This project is an AI-powered Solar Panel Defect Detection System built using:

* TensorFlow / Keras
* MobileNetV2
* FastAPI
* Computer Vision
* REST API Deployment
* Render Cloud Hosting

The system takes an uploaded solar panel image and predicts whether the panel is:

* Defective
* Normal

The project demonstrates a complete end-to-end machine learning deployment pipeline:

1. Data preprocessing
2. Deep learning model training
3. Model saving and loading
4. API development using FastAPI
5. Cloud deployment using Render
6. Real-time image inference

---

# Features

* Image upload API
* Deep Learning based classification
* MobileNetV2 transfer learning
* FastAPI backend
* Cloud deployment support
* Probability score output
* Health check endpoint
* Production-ready API structure

---

# Tech Stack

| Category            | Technology         |
| ------------------- | ------------------ |
| Language            | Python             |
| Deep Learning       | TensorFlow / Keras |
| Model Architecture  | MobileNetV2        |
| API Framework       | FastAPI            |
| Deployment          | Render             |
| Image Processing    | Pillow             |
| Numerical Computing | NumPy              |
| Server              | Uvicorn            |

---

# Project Structure

```bash
solar-defect-detection/
│
├── app.py
├── model_fixed.keras
├── requirements.txt
├── README.md
└── assets/
```

---

# Model Details

## Architecture

The project uses:

* MobileNetV2
* Transfer Learning
* Binary Classification

MobileNetV2 was selected because:

* Lightweight architecture
* Fast inference
* Good performance on image tasks
* Suitable for deployment environments

---

# Classification Logic

The model outputs a probability score between 0 and 1.

Decision logic:

```python
prediction = "Defective" if prob > THRESHOLD else "Normal"
```

Current threshold:

```python
THRESHOLD = 0.4
```

## Why Threshold = 0.4?

Lowering the threshold improves defect sensitivity.

In solar defect detection:

* Missing a defective panel is costly
* Higher recall is preferred
* Slightly more false positives are acceptable

This threshold helps the system detect defects more aggressively.

---

# API Endpoints

## Base URL

```text
https://solar-defect-detection-2.onrender.com
```

---

## 1. Home Endpoint

### Request

```http
GET /
```

### Response

```json
{
  "message": "Solar Panel Defect Detection API"
}
```

---

## 2. Health Check Endpoint

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok"
}
```

---

## 3. Prediction Endpoint

### Request

```http
POST /predict
```

### Form Data

| Key  | Type       |
| ---- | ---------- |
| file | Image File |

### Example Response

```json
{
  "prediction": "Defective",
  "probability": 0.5015517473220825
}
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd solar-defect-detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Locally

```bash
uvicorn app:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Deployment

The project is deployed using Render.

## Deployment Command

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

# Required Dependencies

```text
tensorflow==2.19.0
fastapi
uvicorn
numpy
pillow
h5py
python-multipart
```

## Why OpenCV Is Not Used

This project does not use OpenCV because the image preprocessing pipeline is handled using:

* Pillow (PIL)
* NumPy
* TensorFlow preprocessing utilities

Current preprocessing workflow:

```text
Image Upload
→ Pillow Image Processing
→ NumPy Conversion
→ MobileNetV2 preprocess_input
→ TensorFlow Model Inference
```

Keeping dependencies minimal improves:

* deployment speed
* project maintainability
* build size
* production efficiency

OpenCV can be added in future versions for advanced computer vision tasks such as:

* thermal image analysis
* segmentation
* contour detection
* defect localization
* video-based inspection

---

# Core Workflow

## Step 1: Image Upload

User uploads a solar panel image.

## Step 2: Image Preprocessing

The image is:

* resized to 224x224
* converted to RGB
* normalized using MobileNetV2 preprocessing
* converted into tensor format

## Step 3: Model Inference

TensorFlow model predicts defect probability.

## Step 4: Decision Logic

Probability is compared with threshold.

## Step 5: API Response

Prediction and confidence score returned as JSON.

---

# Example Prediction Flow

```text
Input Image
     ↓
Preprocessing
     ↓
MobileNetV2 Model
     ↓
Probability Score
     ↓
Threshold Comparison
     ↓
Final Prediction
```

---

# Error Handling

The API handles:

* model loading failure
* invalid image upload
* missing files
* corrupted images

Example:

```json
{
  "error": "Model not loaded"
}
```

---

# Future Improvements

Possible future enhancements:

* Multi-class defect classification
* Grad-CAM heatmaps
* Confidence interpretation
* Database integration
* User authentication
* Frontend dashboard
* Real-time monitoring
* Docker containerization
* CI/CD pipeline
* Edge deployment
* GPU optimization

---

# Learning Outcomes

This project demonstrates understanding of:

* Deep Learning
* Transfer Learning
* CNN-based image classification
* FastAPI backend development
* REST API engineering
* Cloud deployment
* Production ML workflow
* Model inference pipelines
* Image preprocessing

---

# Challenges Faced

During deployment:

* dependency issues
* FastAPI configuration
* multipart form handling
* model loading paths
* Render deployment configuration

These issues were resolved through debugging and dependency management.

---

# Resume Value

This project is valuable for:

* AI/ML internships
* Computer Vision roles
* Deep Learning projects
* Backend + ML integration portfolios

Key strengths:

* Real deployment
* Production API
* Cloud hosting
* TensorFlow integration
* Practical inference pipeline

---

# Author

Akshat

AI/ML and Data Science Enthusiast

---

# License

This project is for educational and portfolio purposes.
