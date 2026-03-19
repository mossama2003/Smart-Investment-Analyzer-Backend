"""
main.py

ملف رئيسي لتشغيل المشروع كله:
1️⃣ تحميل البيانات من CSV
2️⃣ إضافة Features
3️⃣ (اختياري) تدريب الموديلات
4️⃣ سؤال المستخدم عن السهم والموديل
5️⃣ عمل Prediction وعرض النتيجة
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd

# ===== إضافة مجلد src للـ Python path =====
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_all_assets
from features import add_features
from predict import predict_asset

# ===== إعداد Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== ملفات البيانات =====
asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]

# ===== تحميل البيانات =====
logging.info("Loading assets...")
all_assets_data = load_all_assets(asset_files)


# ===== (اختياري) تدريب =====
def train_models():
    logging.info("Training XGBoost models...")
    try:
        import train_xgb
    except ImportError:
        logging.warning("train_xgb.py not found")

    logging.info("Training RL models...")
    try:
        import train_rl
    except ImportError:
        logging.warning("train_rl.py not found")


# ===== User Interaction =====
def ask_user():
    print("\n📊 Available Assets:")
    for i, asset in enumerate(asset_files, 1):
        print(f"{i}. {asset.replace('.csv', '')}")

    try:
        choice = int(input("\n👉 Choose asset number: "))
        selected_file = asset_files[choice - 1]
    except:
        print("❌ Invalid choice")
        return None, None

    model_type = input("🤖 Choose model (xgboost / lstm / gru / rl): ").lower()

    return selected_file, model_type


# ===== Main Flow =====
def main():
    # لو عايز تشغل التدريب
    # train_models()

    selected_file, model_type = ask_user()

    if not selected_file:
        return

    logging.info("Running prediction...")

    price, action = predict_asset(selected_file, model_type=model_type)

    print("\n📈 Result:")
    print(f"Asset: {selected_file.replace('.csv','')}")
    print(f"Predicted Price: {price}")
    print(f"Recommended Action: {action}")

    # ===== حفظ النتيجة =====
    result_df = pd.DataFrame([{
        "Asset": selected_file.replace(".csv",""),
        "Predicted_Price": price,
        "Action": action
    }])

    result_df.to_csv("prediction_result.csv", index=False)
    logging.info("Result saved to prediction_result.csv")


if __name__ == "__main__":
    main()