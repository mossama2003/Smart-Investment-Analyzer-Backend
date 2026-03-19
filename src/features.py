# src/features.py

"""
features.py

هذا الملف مسؤول عن:
1️⃣ حساب أهم المؤشرات المالية (Technical Indicators) لكل Asset
   - Moving Average (MA)
   - Relative Strength Index (RSI)
   - MACD
   - Daily Return
   - Rolling Volatility
2️⃣ إضافة الأعمدة دي على DataFrame جاهز للـ Machine Learning أو Reinforcement Learning
3️⃣ تجهيز البيانات لكل Asset بشكل مستقل

Usage:
from features import add_features

df_with_features = add_features(df)
"""

import pandas as pd
import numpy as np

def add_features(df, ma_windows=[5, 10, 20], rsi_period=14):
    df_feat = df.copy()

    # ----- Daily Return -----
    df_feat['Daily_Return'] = df_feat['Close'].pct_change()

    # ----- Moving Averages -----
    for window in ma_windows:
        df_feat[f'MA_{window}'] = df_feat['Close'].rolling(window=window).mean()

    # ----- Rolling Volatility -----
    for window in ma_windows:
        df_feat[f'Volatility_{window}'] = df_feat['Close'].rolling(window=window).std()

    # ----- RSI -----
    delta = df_feat['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_rolling = pd.Series(gain).rolling(window=rsi_period).mean()
    loss_rolling = pd.Series(loss).rolling(window=rsi_period).mean()

    rs = gain_rolling / loss_rolling
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    # ----- MACD -----
    ema_12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = ema_12 - ema_26

    # ----- Fill NaN values -----
    df_feat.fillna(0, inplace=True)

    return df_feat

# ===== Example Usage =====
if __name__ == "__main__":
    from data_loader import load_asset_data
    
    # ✅ مثال مع الملفات الجديدة
    test_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]
    
    for f in test_files:
        df = load_asset_data(f)
        if df is not None:
            df_features = add_features(df)
            print(f"===== {f} with Features =====")
            print(df_features.head())