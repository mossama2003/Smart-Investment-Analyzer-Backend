# src/train_gru.py

"""train_gru.py

FIXES applied:
1. add_features() is now ENABLED so GRU trains on the same feature set used at prediction time.
2. scaler_X and scaler_y are saved alongside the model so predict.py can inverse-transform
   the scaled output back to a real price.
3. Training logic is fully wrapped in train_gru() — no top-level side effects.
"""

import os
import numpy as np
import joblib

from src.data_loader import load_all_assets
from src.features import add_features

from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


def train_gru():
    # ===== Paths =====
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    model_dir = os.path.join(project_root, "models", "gru")
    os.makedirs(model_dir, exist_ok=True)
    model_dir = os.path.join(project_root, "models", "gru")
    os.makedirs(model_dir, exist_ok=True)

    # ===== Config =====
    asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]
    timesteps = 10

    # ===== Load + Feature Engineering =====
    all_assets = load_all_assets(asset_files)

    for asset_name, df in all_assets.items():
        print(f"Training GRU for {asset_name}...")

        # FIX 1: add_features MUST be called here — same as predict.py
        df = add_features(df)

        feature_cols = [
            col for col in df.columns
            if col not in ["Date", "Close", "Open", "High", "Low", "Volume"]
        ]

        X = df[feature_cols].values
        y = df["Close"].values.reshape(-1, 1)

        # FIX 2: Scale X AND y, then SAVE both scalers
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)

        # Save scalers so predict.py can inverse-transform
        joblib.dump(scaler_X, os.path.join(model_dir, f"{asset_name}_gru_scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(model_dir, f"{asset_name}_gru_scaler_y.pkl"))

        # Build sequences
        X_seq, y_seq = [], []
        for i in range(timesteps, len(X_scaled)):
            X_seq.append(X_scaled[i - timesteps: i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        features_num = X_seq.shape[2]
        features_num = X_seq.shape[2]

        # ===== GRU Model =====
        model = Sequential([
            GRU(64, input_shape=(timesteps, features_num), return_sequences=False),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(
            X_seq, y_seq,
            epochs=150,
            batch_size=16,
            validation_split=0.1,
            callbacks=[es],
            verbose=1,
        )

        # ===== Save model =====
        model_path = os.path.join(model_dir, f"{asset_name}_gru.keras")
        model.save(model_path)
        print(f"✅ Saved GRU model + scalers for {asset_name}\n")

    return True


if __name__ == "__main__":
    train_gru()