# src/train_gru.py

"""train_gru.py

ملف لتدريب GRU على بيانات الأسهم وصناديق الذهب.
بديل مباشر لـ LSTM مع أداء أسرع.

مهم:
- تم تغليف منطق التدريب داخل دالة train_gru()
- تم منع التشغيل التلقائي عند الاستيراد عبر استخدام __name__ == "__main__".
"""

import os
import numpy as np
import pandas as pd

from src.data_loader import load_all_assets
from features import add_features

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


def train_gru():
    # ===== تحديد مسار المشروع =====
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    model_dir = os.path.join(project_root, "models", "gru")
    os.makedirs(model_dir, exist_ok=True)

    # ===== الملفات =====
    asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]

    timesteps = 10

    # هذه القيمة سيتم تحديدها بعد تجهيز البيانات لكل asset
    features_num = None

    # ===== التحميل + Features =====
    all_assets = load_all_assets(asset_files)

    for asset_name, df in all_assets.items():
        print(f"Training GRU for {asset_name}...")

        # (اختياري) لو كانت add_features مطلوبة فعلاً في مشروعك
        # جرّب تفعيلها إذا كانت features مطلوبة لنموذجك:
        # df = add_features(df)

        # Features
        feature_cols = [
            col
            for col in df.columns
            if col not in ["Date", "Close", "Open", "High", "Low", "Volume"]
        ]

        X = df[feature_cols].values
        y = df["Close"].values.reshape(-1, 1)

        # Scaling
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)

        # Sequences
        X_seq, y_seq = [], []
        for i in range(timesteps, len(X_scaled)):
            X_seq.append(X_scaled[i - timesteps : i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        features_num = X_seq.shape[2]

        # ===== GRU Model =====
        model = Sequential()
        model.add(GRU(50, input_shape=(timesteps, features_num)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        # ===== Train =====
        es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
        model.fit(
            X_seq,
            y_seq,
            epochs=100,
            batch_size=16,
            callbacks=[es],
            verbose=1,
        )

        # ===== Save =====
        model_path = os.path.join(model_dir, f"{asset_name}_gru.keras")
        model.save(model_path)

        print(f"✅ Saved GRU model to {model_path}\n")

    return True


if __name__ == "__main__":
    train_gru()
