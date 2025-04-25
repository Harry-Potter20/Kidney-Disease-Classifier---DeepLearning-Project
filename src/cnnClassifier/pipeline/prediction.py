import tensorflow as tf
import numpy as np
import json
import os
import sys
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# === Setup base path ===
BASE_DIR = Path(__file__).resolve().parent

# === Load class names from class_names.json ===
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names" / "class_names.json"
if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"Could not find class names file at: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)["class_names"]

# === Load trained model ===
MODEL_PATH = BASE_DIR / "model" / "model.h5"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model file at: {MODEL_PATH}")

model = load_model(MODEL_PATH)

def preprocess_image(img_path: str, target_size=(224, 224)):
    """Preprocess image to match model input"""
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img_path: str) -> tuple:
    """Predict class of a single image and return label and confidence"""
    try:
        img_array = preprocess_image(img_path)
        probs = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(probs, axis=1)[0]
        predicted_label = class_names[predicted_index]
        confidence = probs[0][predicted_index] * 100
        print(f"📷 {img_path} → {predicted_label} ({confidence:.2f}%)")
        return predicted_label, confidence
    except Exception as e:
        print(f"⚠️ Failed to predict {img_path}: {e}")
        return None, None

def predict_folder(folder_path: str):
    """Run predictions for all images in a folder"""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print("🚫 No valid images found in the folder.")
        return

    print(f"📁 Predicting {len(image_files)} images in: {folder_path}\n")
    for img_path in image_files:
        predict_image(img_path)

# === CLI Usage ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n  python prediction.py path/to/image.jpg\n  python prediction.py path/to/folder/")
    else:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            predict_folder(input_path)
        elif os.path.isfile(input_path):
            predict_image(input_path)
        else:
            print("🚫 Invalid path. Please provide a valid image or folder path.")
