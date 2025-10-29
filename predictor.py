# app/predictor.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Path model dan class indices
MODEL_PATH = "models/saved_models/padang_food_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

# Load model
print("🧠 Memuat model makanan Padang...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model berhasil dimuat!")

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Buat mapping index -> nama kelas
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_food(image_path: str):
    """
    Prediksi jenis makanan Padang berdasarkan model lokal.
    """
    try:
        # Baca dan preprocess gambar
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        confidence = float(predictions[0][idx])

        class_name = idx_to_class[idx]

        print(f"🍛 Prediksi: {class_name} ({confidence*100:.2f}%)")
        return {"class": class_name, "confidence": confidence}
    except Exception as e:
        print("❌ Gagal memprediksi:", e)
        return None
