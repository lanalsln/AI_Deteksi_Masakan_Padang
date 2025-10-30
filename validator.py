# validator.py
import os
import re
import google.generativeai as genai

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None
    print("⚠️ GEMINI_API_KEY tidak ditemukan — validasi akan dilewati.")

def is_food_image(image_path):
    """Gunakan Gemini untuk memastikan gambar adalah makanan."""
    if not model:
        return True  # jika belum punya key, skip validasi

    try:
        uploaded = genai.upload_file(image_path)
        prompt = "Apakah gambar ini berisi makanan? Jawab hanya 'ya' atau 'tidak'."
        result = model.generate_content([prompt, uploaded])
        answer = result.text.strip().lower()
        return bool(re.search(r"\b(ya|makanan|food|dish|meal)\b", answer))
    except Exception as e:
        print("❌ Error validasi:", e)
        return True
