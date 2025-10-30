import os
import shutil
import threading
import time
from flask import Flask, render_template, request, jsonify
from predictor import predict_food, reload_model
from validator import is_food_image
from retrain_model import run_retrain

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FEEDBACK_LOG = "feedback_log.txt"
DATASET_PATH = "data/processed/train"
os.makedirs(DATASET_PATH, exist_ok=True)

# Flag supaya retrain tidak dijalankan bersamaan
_retrain_running = False


def background_retrain():
    """Menjalankan retraining model di thread terpisah"""
    global _retrain_running
    if _retrain_running:
        print("⚙️ Retrain sedang berjalan, dilewati sementara.")
        return

    _retrain_running = True
    try:
        print("🔁 Memulai retrain model...")
        run_retrain()  # panggil retrain_model.py
        print("✅ Retrain selesai. Reload model...")
        reload_model()
    except Exception as e:
        print("❌ Error retrain:", e)
    finally:
        _retrain_running = False


@app.route("/")
def index():
    # render file HTML kamu: templates/upload.html
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi makanan"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file upload."})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Nama file kosong."})

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Validasi apakah gambar adalah makanan
    if not is_food_image(save_path):
        return jsonify({
            "success": True,
            "predictions": [{"class": "Bukan makanan", "confidence": 1.0}],
            "image_path": save_path
        })

    preds = predict_food(save_path)
    return jsonify({"success": True, "predictions": preds, "image_path": save_path})


@app.route("/feedback", methods=["POST"])
def feedback():
    """Menerima feedback dari user & trigger retrain"""
    data = request.get_json()
    prediction = data.get("prediction")
    correct = data.get("correct")
    correct_label = data.get("correct_label")

    # Catat feedback
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.asctime()} | pred={prediction} | correct={correct} | label={correct_label}\n")

    # Jika prediksi salah dan user memberikan label benar
    if not correct and correct_label:
        target_dir = os.path.join(DATASET_PATH, correct_label)
        os.makedirs(target_dir, exist_ok=True)

        # Jika kamu ingin menyimpan gambar untuk retrain
        image_path = request.json.get("image_path")
        if image_path and os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(target_dir, os.path.basename(image_path)))

        # Jalankan retrain di thread background
        threading.Thread(target=background_retrain, daemon=True).start()

    return jsonify({
        "success": True,
        "message": "✅ Feedback diterima. Model akan dilatih ulang di background."
    })


if __name__ == "__main__":
    app.run(debug=True)
