"""
train_xgb.py

ملف لتدريب XGBoost / RandomForest على بيانات الأسهم وصناديق الذهب.
"""

import os
import joblib
import pandas as pd
from data_loader import load_all_assets
from features import add_features
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ===== الإعدادات =====
RAW_DATA_DIR = "../data/raw/"
MODEL_DIR = "../models/xgboost/"
os.makedirs(MODEL_DIR, exist_ok=True)

# قائمة ملفات Assets (غيرها حسب الحاجة)
asset_files = ["AAPL.csv", "GOLD_ETF.csv", "EGX30.csv"]

# ===== التحميل + Features =====
all_assets = load_all_assets(asset_files)

for asset_name, df in all_assets.items():
    print(f"Training XGBoost for {asset_name}...")
    
    # Features & Target
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
    X = df[feature_cols]
    y = df['Close']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # نموذج XGBoost
    model = XGBRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for {asset_name}: {mse}")

    # Save Model
    model_path = os.path.join(MODEL_DIR, f"{asset_name}_xgboost.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}\n")