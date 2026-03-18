"""
predict.py

هذا الملف مسؤول عن:
1️⃣ تحميل الموديلات المدربة (XGBoost / GRU / LSTM / RL)
2️⃣ تجهيز البيانات الأخيرة من DataFrame لاستخدامها في التنبؤ
3️⃣ عمل التنبؤ بأسعار المستقبلية لكل Asset
4️⃣ عمل توصية (Action: Buy / Sell / Hold) لكل Asset بناء على Prediction و RL Agent
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging

from data_loader import load_asset_data
from features import add_features

# إعداد Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# المسارات الافتراضية للموديلات
MODEL_DIR = "../models/"

def predict_asset(filename, model_type="xgboost"):
    """
    وظيفة الدالة:
    ----------------
    - تحميل البيانات الخام للـ Asset
    - إضافة الـ Features
    - تحميل الموديل المدرب
    - عمل Prediction للسعر القادم
    - إعطاء Action: Buy / Sell / Hold
    """
    # تحميل البيانات + Features
    df = load_asset_data(filename)
    if df is None:
        logging.error(f"Failed to load {filename}")
        return None, "Hold"
    df = add_features(df)

    # تجهيز Features للـ Model (XGBoost / RandomForest / GRU / LSTM)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X = df[feature_cols].iloc[-1:].values  # آخر صف فقط للتنبؤ

    try:
        # ===== XGBoost / Random Forest =====
        if model_type.lower() in ["xgboost", "randomforest"]:
            model_path = os.path.join(MODEL_DIR, model_type.lower(), filename.replace(".csv", f"_{model_type}.pkl"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = joblib.load(model_path)
            predicted_price = model.predict(X)[0]

        # ===== LSTM =====
        elif model_type.lower() == "lstm":
            import tensorflow as tf
            model_path = os.path.join(MODEL_DIR, "lstm", filename.replace(".csv", "_lstm.h5"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"LSTM model not found: {model_path}")
            model = tf.keras.models.load_model(model_path)
            X_seq = X.reshape((1, X.shape[0], X.shape[1]))  # 3D input
            predicted_price = model.predict(X_seq)[0][0]

        # ===== GRU =====
        elif model_type.lower() == "gru":
            import tensorflow as tf
            model_path = os.path.join(MODEL_DIR, "gru", filename.replace(".csv", "_gru.h5"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GRU model not found: {model_path}")
            model = tf.keras.models.load_model(model_path)
            X_seq = X.reshape((1, X.shape[0], X.shape[1]))  # 3D input
            predicted_price = model.predict(X_seq)[0][0]

        # ===== Reinforcement Learning =====
        elif model_type.lower() == "rl":
            import gymnasium as gym
            from stable_baselines3 import DQN
            model_path = os.path.join(MODEL_DIR, "rl", filename.replace(".csv", "_rl.zip"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"RL model not found: {model_path}")
            model = DQN.load(model_path)

            # نفترض أن الـ state = آخر صف Features
            state = X.flatten()
            action_code, _ = model.predict(state, deterministic=True)
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            return None, action_map.get(action_code, "Hold")

        else:
            raise ValueError("model_type must be xgboost, randomforest, lstm, gru, or rl")

    except Exception as e:
        logging.error(f"Prediction failed for {filename} with {model_type}: {e}")
        return None, "Hold"

    # ===== Simple Decision Logic =====
    last_close = df['Close'].iloc[-1]
    if predicted_price > last_close * 1.01:
        action = "Buy"
    elif predicted_price < last_close * 0.99:
        action = "Sell"
    else:
        action = "Hold"

    return predicted_price, action

def predict_portfolio(asset_files, model_type="xgboost"):
    """
    توقع لكل Asset في قائمة الملفات
    """
    results = {}
    for file in asset_files:
        price, action = predict_asset(file, model_type=model_type)
        results[file.replace(".csv","")] = {"predicted_price": price, "action": action}
    return results

# ===== Example Usage =====
if __name__ == "__main__":
    asset_files = ["AAPL.csv", "GOLD_ETF.csv", "EGX30.csv"]
    results = predict_portfolio(asset_files, model_type="gru")
    for asset, res in results.items():
        logging.info(f"{asset}: Predicted Price = {res['predicted_price']}, Action = {res['action']}")