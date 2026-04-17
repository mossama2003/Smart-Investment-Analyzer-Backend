# src/train_all.py

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(level=logging.INFO)

def train_all_models():
    print("🚀 Starting full training pipeline...\n")

    # ===== XGBoost =====
    try:
        print("📊 Training XGBoost...")
        from src.train_xgb import train_xgb
        train_xgb()
        print("✅ XGBoost done\n")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}\n")

    # ===== GRU =====
    try:
        print("🧠 Training GRU...")
        from src.train_gru import train_gru
        train_gru()
        print("✅ GRU done\n")
    except Exception as e:
        print(f"❌ GRU failed: {e}\n")

    # ===== LSTM (اختياري) =====
    try:
        print("📈 Training LSTM...")
        from src.train_lstm import train_lstm
        train_lstm()
        print("✅ LSTM done\n")
    except Exception as e:
        print(f"⚠️ LSTM skipped: {e}\n")

    # ===== RL (اختياري) =====
    try:
        print("🤖 Training RL...")
        from src.train_rl import train_rl
        train_rl()
        print("✅ RL done\n")
    except Exception as e:
        print(f"⚠️ RL skipped: {e}\n")

    print("🎉 Training pipeline finished!")