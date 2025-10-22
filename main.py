from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.predict import load_model_and_classes, predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model saat startup
print("\n" + "="*50)
print("🚀 STARTING FLASK APP")
print("="*50)

try:
    print("\n📦 Loading AI model...")
    model, idx_to_class = load_model_and_classes()
    print("✅ Model loaded successfully!\n")
    print("="*50)
except Exception as e:
    print(f"\n❌ ERROR: Gagal load model!")
    print(f"   {str(e)}")
    print("\n⚠️  Pastikan Anda sudah training model terlebih dahulu!")
    print("   Jalankan: python models/train_model.py")
    print("="*50 + "\n")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("\n📸 Menerima request prediksi...")
    
    if 'file' not in request.files:
        print("❌ Tidak ada file yang diupload")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("❌ File kosong")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"💾 File disimpan: {filename}")
        print("🤖 Memproses prediksi...")
        
        try:
            # Prediksi
            results = predict_image(filepath, model, idx_to_class)
            
            print("✅ Prediksi berhasil!")
            print(f"   Top prediction: {results[0]['class']} ({results[0]['confidence']*100:.2f}%)")
            
            return jsonify({
                'success': True,
                'predictions': results,
                'image_url': f'/static/uploads/{filename}'
            })
        except Exception as e:
            print(f"❌ Error saat prediksi: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    print("❌ Tipe file tidak valid")
    return jsonify({'error': 'Invalid file type. Gunakan JPG, JPEG, atau PNG'}), 400

if __name__ == '__main__':
    # Buat folder upload jika belum ada
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("\n🌐 Server berjalan di: http://127.0.0.1:5000")
    print("   Tekan CTRL+C untuk stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)