# src/predict.py

import os
import joblib
import numpy as np
import logging

from data_loader import load_asset_data
from features import add_features

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== Paths (FIXED) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TIMESTEPS = 10


# ===== Helper Functions =====
def get_features(df):
    return [
        col for col in df.columns
        if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ]


def get_action(predicted_price, last_close):
    if predicted_price > last_close * 1.01:
        return "Buy"
    elif predicted_price < last_close * 0.99:
        return "Sell"
    return "Hold"


# ===== Main Prediction =====
def predict_asset(filename, model_type="xgboost"):
    df = load_asset_data(filename)

    if df is None:
        logging.error(f"Failed to load {filename}")
        return None, "Hold"

    df = add_features(df)
    feature_cols = get_features(df)

    try:
        model_type = model_type.lower()

        # ===== XGBoost / RandomForest =====
        if model_type in ["xgboost", "randomforest"]:
            model_path = os.path.join(
                MODEL_DIR,
                model_type,
                filename.replace(".csv", f"_{model_type}.pkl")
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            model = joblib.load(model_path)

            X = df[feature_cols].iloc[-1:].values
            predicted_price = model.predict(X)[0]

        # ===== LSTM =====
        elif model_type == "lstm":
            import tensorflow as tf

            model_path = os.path.join(
                MODEL_DIR, "lstm", filename.replace(".csv", "_lstm.keras")
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"LSTM model not found: {model_path}")

            model = tf.keras.models.load_model(model_path, compile=False)

            X_seq = df[feature_cols].iloc[-TIMESTEPS:].values
            X_seq = X_seq.reshape((1, TIMESTEPS, len(feature_cols)))

            predicted_price = model.predict(X_seq, verbose=0)[0][0]

        # ===== GRU (FIXED) =====
        elif model_type == "gru":
            import tensorflow as tf

            model_path = os.path.join(
                MODEL_DIR, "gru", filename.replace(".csv", "_gru.keras")
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GRU model not found: {model_path}")

            model = tf.keras.models.load_model(model_path, compile=False)

            X_seq = df[feature_cols].iloc[-TIMESTEPS:].values
            X_seq = X_seq.reshape((1, TIMESTEPS, len(feature_cols)))

            predicted_price = model.predict(X_seq, verbose=0)[0][0]

        # ===== RL (DQN) =====
        elif model_type == "rl":
            from stable_baselines3 import DQN

            model_path = os.path.join(
                MODEL_DIR, "rl", filename.replace(".csv", "_rl.zip")
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"RL model not found: {model_path}")

            model = DQN.load(model_path)

            state = df[feature_cols].iloc[-1].values.astype(np.float32)
            action_code, _ = model.predict(state, deterministic=True)

            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            return None, action_map.get(int(action_code), "Hold")

        else:
            raise ValueError("Invalid model_type")

        # ===== Decision =====
        last_close = df['Close'].iloc[-1]
        action = get_action(predicted_price, last_close)

        return float(predicted_price), action

    except Exception as e:
        logging.error(f"Prediction failed for {filename} with {model_type}: {e}")
        return None, "Hold"


# ===== Portfolio Prediction =====
def predict_portfolio(asset_files, model_type="xgboost"):
    results = {}

    for file in asset_files:
        price, action = predict_asset(file, model_type=model_type)

        results[file.replace(".csv", "")] = {
            "predicted_price": price,
            "action": action
        }

    return results


# ===== Run مباشر =====
if __name__ == "__main__":
    asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]

    results = predict_portfolio(asset_files, model_type="gru")

    for asset, res in results.items():
        logging.info(
            f"{asset}: Predicted Price = {res['predicted_price']}, Action = {res['action']}"
        )