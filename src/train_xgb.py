"""
train_xgb.py

FIXES applied:
1. add_features() called consistently (was already there, kept).
2. Target is the NEXT day's Close (shifted by 1) so the model actually learns to
   PREDICT the future, not just reproduce today's value. This is the core reason
   predictions were identical to the last known price.
3. Model is saved with a scaler so predict.py can feed normalized input if needed
   (XGBoost is scale-invariant, but keeping it consistent).
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.data_loader import load_all_assets
from src.features import add_features
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "xgboost")
os.makedirs(MODEL_DIR, exist_ok=True)

asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]


def train_xgb():
    all_assets = load_all_assets(asset_files)

    for asset_name, df in all_assets.items():
        print(f"Training XGBoost for {asset_name}...")

        # Feature engineering
        df = add_features(df)

        feature_cols = [
            col for col in df.columns
            if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        ]

        # FIX: Target = NEXT day's Close (shift -1) so model predicts the future
        df = df.copy()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[feature_cols]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"  RMSE for {asset_name}: {rmse:.4f}")

        model_path = os.path.join(MODEL_DIR, f"{asset_name}_xgboost.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved XGBoost model to {model_path}\n")


if __name__ == "__main__":
    print("📊 Training XGBoost...")
    train_xgb()
    print("✅ XGBoost done")