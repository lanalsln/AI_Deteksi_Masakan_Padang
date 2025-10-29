# app/validator.py
import os
import google.generativeai as genai

# 🔑 Ambil API key dari environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ Environment variable GEMINI_API_KEY tidak ditemukan.")

# ⚙️ Konfigurasi client Gemini
genai.configure(api_key=api_key)

# 🧠 Gunakan model Gemini terbaru
model = genai.GenerativeModel("gemini-2.5-flash")  # atau "gemini-2.5-flash" jika tersedia

def is_food_image(image_path: str) -> bool:
    """
    Validasi apakah gambar merupakan makanan menggunakan Gemini AI.
    Return True jika ya, False jika tidak.
    """
    try:
        print(f"📸 Mengunggah gambar: {image_path}")
        # Upload file ke server Google dulu
        uploaded_file = genai.upload_file(image_path)
        
        prompt = (
            "Tentukan apakah gambar ini menunjukkan makanan atau bukan. "
            "Jawab hanya dengan satu kata: 'ya' jika makanan, 'tidak' jika bukan."
        )

        response = model.generate_content(
            [prompt, uploaded_file],
            request_options={"timeout": 60}
        )

        answer = response.text.strip().lower()
        print("🧠 Jawaban model:", answer)

        # Kata kunci positif
        keywords_yes = ["ya", "makanan", "masakan", "hidangan", "kuliner"]
        return any(k in answer for k in keywords_yes)

    except Exception as e:
        print("❌ Terjadi kesalahan saat validasi gambar:", e)
        return False
