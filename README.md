# BMI-Prediction

# 🧠 BMI Prediction from Face Images

This project explores the feasibility of predicting Body Mass Index (BMI) from facial images using computer vision and machine learning techniques.

## 🔍 Overview

- **Input**: Frontal face shot (RGB image)
- **Output**: Predicted BMI value (float)

## 📦 Key Components

- `eda & vgg.ipynb`: Exploratory data analysis and VGG-based model.
- `insightface & svr.ipynb`: Uses InsightFace for face alignment and embeddings; applies Support Vector Regression (SVR) for BMI prediction.
- `app.py`: Streamlit web app for uploading face images and displaying predicted BMI.

## 📈 Model Architecture

- **Face Embedding**: Pre-trained InsightFace (ArcFace) for extracting facial features.
- **Regressor**: Support Vector Regression (SVR) trained on embedding vectors.
- **Alternative**: VGG19

## 📊 Model Evaluation by Gender (Pearson Correlation)

| Model               | Female | Male | Overall |
|---------------------|--------|------|---------|
| **InsightFace + SVR** | 0.73   | 0.76 | 0.74    |

## 🖥️ Web Application (`app.py`)

The `app.py` file launches a **Streamlit web app** that allows users to:

- 📷 **Use Webcam**: Capture a live image directly from your webcam.
- 📁 **Upload an Image**: Select and upload a face photo from your device.
- 🧠 **Get Prediction**: The app processes the image, extracts facial features using InsightFace, and predicts the BMI using the pre-trained SVR model.
