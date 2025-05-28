import streamlit as st
st.set_page_config(page_title="Face BMI Predictor", layout="centered")
import numpy as np
import cv2
import torch
from PIL import Image
import tempfile
import insightface
import joblib
import os

# ----------------------------
# ArcFace setup
# ----------------------------
@st.cache_resource
def load_arcface_model():
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

arcface = load_arcface_model()

# ----------------------------
# BMI Prediction Logic
# ----------------------------
def extract_arcface_embedding(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = arcface.get(img_bgr)
    if not faces:
        return None, "No face detected."
    return faces[0].embedding, None

def load_model_and_scaler(gender):
    gender = gender.lower()
    model_path = f"svr_model_{gender}.pkl"
    scaler_path = f"scaler_{gender}.pkl"
    print(f"Loading model from: {model_path}")
    print(f"Loading scaler from: {scaler_path}")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def predict_bmi(embedding, gender):
    model, scaler = load_model_and_scaler(gender)
    if model is None or scaler is None:
        return None, "Model or scaler not found."
    X_scaled = scaler.transform([embedding])
    log_bmi = model.predict(X_scaled)[0]
    return round(np.exp(log_bmi), 2), None

# ----------------------------
# Streamlit UI
# ----------------------------
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())
st.title("📷 Predict BMI from Face Photo")
st.markdown("Upload or take a photo of your face and select your gender.")

# Gender Selection
gender = st.selectbox("Your gender", ["female", "male"])

# Image Input
input_method = st.radio("Choose input method", ["📁 Upload an image", "📸 Use webcam"])

image = None
if input_method == "📁 Upload an image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_method == "📸 Use webcam":
    cam_input = st.camera_input("Take a face photo")
    if cam_input:
        image = Image.open(cam_input).convert("RGB")

# Run Prediction
if image and st.button("📈 Predict BMI"):
    st.image(image, caption="Input Image", width=300)

    with st.spinner("Extracting face embedding..."):
        embedding, err = extract_arcface_embedding(image)

    if err:
        st.error(err)
    else:
        with st.spinner("Predicting BMI..."):
            bmi, err = predict_bmi(embedding, gender)
        if err:
            st.error(err)
        else:
            st.success(f"✅ Predicted BMI: **{bmi}**")