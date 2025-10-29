# app/main.py
from flask import Flask, render_template, request, jsonify
import os
from validator import is_food_image
from predictor import predict_food
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"success": False, "error": "Tidak ada gambar diunggah."})

        img_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(img_path)
        print(f"📸 Mengunggah gambar: {img_path}")

        # 1️⃣ Validasi dengan Gemini
        if not is_food_image(img_path):
            return jsonify({
                "success": True,
                "predictions": [{"class": "Bukan makanan", "confidence": 1.0}]
            })

        # 2️⃣ Prediksi model lokal
        pred = predict_food(img_path)
        if not pred:
            return jsonify({"success": False, "error": "Gagal memprediksi makanan."})

        predictions = [
            {"class": pred["class"], "confidence": pred["confidence"]}
        ]

        return jsonify({"success": True, "predictions": predictions})

    except Exception as e:
        print("❌ ERROR:", e)
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
