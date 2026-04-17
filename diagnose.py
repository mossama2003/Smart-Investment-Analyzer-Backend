"""
diagnose.py  — run from the project ROOT:
    python diagnose.py

This script:
1. Lists all scaler / model files found in models/
2. For each asset, loads scaler_y and shows what inverse_transform(0.24) gives
   → this tells you if the scaler is correctly recovering real prices
3. Runs a full predict_asset() call and prints the result
4. Compares predicted price against last known Close so you can judge quality

Run BEFORE restarting the server to confirm the fix is working.
"""

import os, sys, joblib, numpy as np

# ── make sure src is on the path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_asset_data
from features    import add_features

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS    = ["ETEL", "COMI", "FWRY"]


def check_scalers():
    print("\n" + "="*60)
    print("SCALER DIAGNOSTIC")
    print("="*60)
    for asset in ASSETS:
        print(f"\n── {asset} ──")
        for folder, infix in [("xgboost",""), ("gru","gru_"), ("lstm","lstm_")]:
            d = os.path.join(MODEL_DIR, folder)
            sy_path = os.path.join(d, f"{asset}_{infix}scaler_y.pkl")
            # fallback
            if not os.path.exists(sy_path):
                sy_path = os.path.join(d, f"{asset}_scaler_y.pkl")

            if os.path.exists(sy_path):
                sy = joblib.load(sy_path)
                # show the real-price range the scaler was fit on
                min_p = float(sy.data_min_[0])
                max_p = float(sy.data_max_[0])
                # inverse-transform a mid-range scaled value → should be ~mid of price range
                sample = float(sy.inverse_transform([[0.5]])[0][0])
                print(f"  [{folder:8s}] scaler_y found | "
                      f"price range=[{min_p:.2f}, {max_p:.2f}] | "
                      f"inverse(0.5)={sample:.4f}")
            else:
                print(f"  [{folder:8s}] scaler_y NOT FOUND at {sy_path}")


def check_predictions():
    print("\n" + "="*60)
    print("PREDICTION DIAGNOSTIC  (model_type=xgboost)")
    print("="*60)

    # import the FIXED predict_asset
    try:
        from predict import predict_asset
    except ImportError:
        print("ERROR: could not import predict_asset — make sure you replaced src/predict.py")
        return

    for asset in ASSETS:
        filename = f"{asset}.csv"
        df = load_asset_data(filename)
        if df is None:
            print(f"  {asset}: could not load CSV")
            continue
        df = add_features(df)
        last_close = float(df['Close'].iloc[-1])
        last_date  = str(df['Date'].iloc[-1]) if 'Date' in df.columns else "?"

        price, action = predict_asset(filename, model_type="xgboost")

        if price is None:
            print(f"  {asset}: prediction returned None")
            continue

        pct_diff = abs(price - last_close) / last_close * 100
        ok = "✅" if pct_diff < 10 else "⚠️ " if pct_diff < 30 else "❌"
        print(f"  {ok} {asset} | last_date={last_date} | "
              f"last_close={last_close:.4f} | predicted={price:.4f} | "
              f"diff={pct_diff:.1f}% | action={action}")


if __name__ == "__main__":
    check_scalers()
    check_predictions()
    print("\nDone.")