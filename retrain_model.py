# retrain_model.py
from train_model import train_model

def run_retrain():
    """Menjalankan ulang training"""
    print("🚀 Mulai retrain...")
    train_model(
        data_dir='data/processed/train',
        save_path='models/saved_models/padang_food_model.keras'
    )
    print("✅ Retrain selesai.")
