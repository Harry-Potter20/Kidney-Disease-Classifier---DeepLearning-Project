import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path
import zipfile

# === Streamlit Config ===
st.set_page_config(
    page_title="🧬 AI Kidney Disease Classifier",
    page_icon="🧠",
    layout="centered"
)

# === Inject Dark Futuristic CSS ===
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #0e1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: scale(1.05);
    }
    .stRadio > div {
        background-color: #161b22;
        padding: 0.5em;
        border-radius: 8px;
    }
    .css-1offfwp {
        background: linear-gradient(to right, #141e30, #243b55);
        padding: 2rem;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Load class names and model ===
CLASS_NAMES_PATH = Path("model/class_names/class_names.json")
MODEL_PATH = Path("model/model.h5")

# Check if class names file exists
if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"Class names file not found at {CLASS_NAMES_PATH}. Please check the file path.")
else:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)["class_names"]

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the file path.")
else:
    model = load_model(MODEL_PATH)

# === Prediction Function ===
def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    img_array = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img: Image.Image):
    img_array = preprocess_image(img)
    preds = model.predict(img_array, verbose=0)
    index = np.argmax(preds[0])
    label = class_names[index]
    confidence = preds[0][index] * 100
    return label, confidence

# === Title & Upload ===
st.title("🧬 Kidney Disease Classifier")
st.subheader("Upload a kidney CT scan image or a folder of images (ZIP).")

upload_type = st.radio("Choose upload type:", ["Single Image", "Folder (ZIP)"])

# === Single Image Upload ===
if upload_type == "Single Image":
    uploaded_file = st.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image..."):
                label, conf = predict_image(img)
            st.success(f"🧠 **Prediction**: {label}")
            st.info(f"📊 Confidence: {conf:.2f}%")

# === ZIP Folder Upload ===
else:
    uploaded_zip = st.file_uploader("Upload a .zip of images", type=["zip"])
    
    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            image_paths = []
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append(os.path.join(root, file))

            st.write(f"📂 Found {len(image_paths)} images.")

            if st.button("🧠 Predict All"):
                with st.spinner("Processing images..."):
                    for img_path in image_paths:
                        img = Image.open(img_path)
                        label, conf = predict_image(img)
                        st.image(img, caption=f"{Path(img_path).name} → {label} ({conf:.2f}%)", use_column_width=True)
