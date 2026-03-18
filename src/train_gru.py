"""
train_gru.py

ملف لتدريب GRU على بيانات الأسهم وصناديق الذهب.
بديل مباشر لـ LSTM مع أداء أسرع.
"""

import os
import numpy as np
import pandas as pd
from data_loader import load_all_assets
from features import add_features
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ===== الإعدادات =====
MODEL_DIR = "../models/gru/"
os.makedirs(MODEL_DIR, exist_ok=True)

asset_files = ["AAPL.csv", "GOLD_ETF.csv", "EGX30.csv"]

timesteps = 10  # طول الـ sequence
features_num = None  # سيتم تحديده بعد Features

# ===== التحميل + Features =====
all_assets = load_all_assets(asset_files)

for asset_name, df in all_assets.items():
    print(f"Training GRU for {asset_name}...")
    
    # Scale Close Price + Features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
    X = df[feature_cols].values
    y = df['Close'].values.reshape(-1,1)

    # Scale data
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Prepare sequences
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X_scaled)):
        X_seq.append(X_scaled[i-timesteps:i])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Update features_num
    features_num = X_seq.shape[2]

    # GRU Model
    model = Sequential()
    model.add(GRU(50, input_shape=(timesteps, features_num)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=100, batch_size=16, callbacks=[es], verbose=1)

    # Save Model
    model_path = os.path.join(MODEL_DIR, f"{asset_name}_gru.h5")
    model.save(model_path)
    print(f"Saved GRU model to {model_path}\n")