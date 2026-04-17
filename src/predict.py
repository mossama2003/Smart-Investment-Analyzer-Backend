# src/predict.py

"""
predict.py — ENSEMBLE EDITION

For each asset, runs ALL available models (XGBoost, GRU, LSTM, RL) and combines:
  - Predicted price  → weighted average of models that return a price
  - Action           → weighted majority vote across all models

Model weights (higher = more trusted):
  XGBoost : 3
  GRU     : 3
  LSTM    : 2
  RL      : 1  (action-only — contributes to vote but not to price average)

Usage:
  predict_asset("COMI.csv")                    # ensemble (default)
  predict_asset("COMI.csv", model_type="gru")  # single model
"""

import os
import joblib
import numpy as np
import logging
from collections import Counter

from data_loader import load_asset_data
from features import add_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TIMESTEPS = 10

MODEL_WEIGHTS = {
    "xgboost": 3,
    "gru":     3,
    "lstm":    2,
    "rl":      1,
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_features(df):
    return [c for c in df.columns
            if c not in ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')]


def _get_action(price, last_close):
    if price > last_close * 1.01:
        return "Buy"
    elif price < last_close * 0.99:
        return "Sell"
    return "Hold"


def _load_scaler(path):
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


def _inverse(scaled, scaler_y):
    if scaler_y is None:
        return float(scaled)
    return float(scaler_y.inverse_transform([[scaled]])[0][0])


def _find(directory, *names):
    """Return first existing path, or None."""
    for name in names:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    return None


def _load_cols(path, fallback):
    if path and os.path.exists(path):
        return joblib.load(path)
    return fallback


# ──────────────────────────────────────────────
# Per-model runners
# Each returns (price | None, action | None)
# ──────────────────────────────────────────────

def _run_xgboost(asset, df, feature_cols, last_close):
    d = os.path.join(MODEL_DIR, "xgboost")
    mp = os.path.join(d, f"{asset}_xgboost.pkl")
    if not os.path.exists(mp):
        return None, None

    model    = joblib.load(mp)
    scaler_X = _load_scaler(_find(d, f"{asset}_scaler_X.pkl"))
    scaler_y = _load_scaler(_find(d, f"{asset}_scaler_y.pkl"))
    cols     = _load_cols(
        _find(d, f"{asset}_features.pkl", f"{asset}_xgboost.pkl_features.pkl"),
        feature_cols
    )

    X = df[cols].iloc[-1:].values.astype(np.float64)
    if scaler_X:
        X = scaler_X.transform(X)

    raw   = model.predict(X)[0]
    price = _inverse(raw, scaler_y)
    return price, _get_action(price, last_close)


def _run_gru(asset, df, feature_cols, last_close):
    d  = os.path.join(MODEL_DIR, "gru")
    mp = os.path.join(d, f"{asset}_gru.keras")
    if not os.path.exists(mp):
        return None, None

    import tensorflow as tf
    model    = tf.keras.models.load_model(mp, compile=False)
    scaler_X = _load_scaler(_find(d, f"{asset}_gru_scaler_X.pkl", f"{asset}_scaler_X.pkl"))
    scaler_y = _load_scaler(_find(d, f"{asset}_gru_scaler_y.pkl", f"{asset}_scaler_y.pkl"))

    X = df[feature_cols].iloc[-TIMESTEPS:].values.astype(np.float64)
    if scaler_X:
        X = scaler_X.transform(X)
    X = X.reshape((1, TIMESTEPS, len(feature_cols)))

    raw   = model.predict(X, verbose=0)[0][0]
    price = _inverse(raw, scaler_y)
    return price, _get_action(price, last_close)


def _run_lstm(asset, df, feature_cols, last_close):
    d  = os.path.join(MODEL_DIR, "lstm")
    mp = os.path.join(d, f"{asset}_lstm.keras")
    if not os.path.exists(mp):
        return None, None

    import tensorflow as tf
    model    = tf.keras.models.load_model(mp, compile=False)
    scaler_X = _load_scaler(_find(d, f"{asset}_lstm_scaler_X.pkl", f"{asset}_scaler_X.pkl"))
    scaler_y = _load_scaler(_find(d, f"{asset}_lstm_scaler_y.pkl", f"{asset}_scaler_y.pkl"))

    X = df[feature_cols].iloc[-TIMESTEPS:].values.astype(np.float64)
    if scaler_X:
        X = scaler_X.transform(X)
    X = X.reshape((1, TIMESTEPS, len(feature_cols)))

    raw   = model.predict(X, verbose=0)[0][0]
    price = _inverse(raw, scaler_y)
    return price, _get_action(price, last_close)


def _run_rl(asset, df, feature_cols, last_close):
    d  = os.path.join(MODEL_DIR, "rl")
    mp = os.path.join(d, f"{asset}_rl.zip")
    if not os.path.exists(mp):
        return None, None

    from stable_baselines3 import DQN
    model  = DQN.load(mp)
    state  = df[feature_cols].iloc[-1].values.astype(np.float32)
    code,_ = model.predict(state, deterministic=True)
    action = {0: "Hold", 1: "Buy", 2: "Sell"}.get(int(code), "Hold")
    return None, action   # RL contributes action only


RUNNERS = {
    "xgboost": _run_xgboost,
    "gru":     _run_gru,
    "lstm":    _run_lstm,
    "rl":      _run_rl,
}


# ──────────────────────────────────────────────
# Ensemble combiner
# ──────────────────────────────────────────────

def _combine(results: dict, last_close: float):
    """
    results = { model_name: (price_or_None, action_or_None) }
    Returns  (weighted_avg_price, majority_action, breakdown)
    """
    price_sum = 0.0
    price_w   = 0.0
    votes     = Counter()
    breakdown = {}

    for name, (price, action) in results.items():
        w = MODEL_WEIGHTS.get(name, 1)

        if price is not None and np.isfinite(price) and price > 0:
            price_sum += price * w
            price_w   += w

        if action is not None:
            votes[action] += w

        breakdown[name] = {
            "price":  round(float(price), 4) if price is not None else None,
            "action": action,
            "weight": w,
        }

    avg_price = price_sum / price_w if price_w > 0 else None

    if votes:
        top_score = votes.most_common(1)[0][1]
        winners   = [a for a, s in votes.items() if s == top_score]
        if len(winners) == 1:
            final_action = winners[0]
        else:
            # Tie-break using price
            final_action = _get_action(avg_price, last_close) if avg_price else "Hold"
    else:
        final_action = _get_action(avg_price, last_close) if avg_price else "Hold"

    return avg_price, final_action, breakdown


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def predict_asset(filename, model_type="ensemble"):
    """
    model_type = "ensemble" (default) | "xgboost" | "gru" | "lstm" | "rl"
    """
    df = load_asset_data(filename)
    if df is None:
        logging.error(f"Failed to load {filename}")
        return None, "Hold"

    df           = add_features(df)
    asset        = filename.replace(".csv", "")
    feature_cols = _get_features(df)
    last_close   = float(df['Close'].iloc[-1])

    try:
        # ── ENSEMBLE ──────────────────────────────────
        if model_type == "ensemble":
            results = {}
            for name, fn in RUNNERS.items():
                try:
                    price, action = fn(asset, df, feature_cols, last_close)
                    results[name] = (price, action)
                    logging.info(f"  [{asset}][{name}] price={price} action={action}")
                except Exception as e:
                    logging.warning(f"  [{asset}][{name}] skipped → {e}")
                    results[name] = (None, None)

            avg_price, final_action, breakdown = _combine(results, last_close)

            logging.info(
                f"[{asset}] ENSEMBLE → last_close={last_close:.4f} | "
                f"avg_price={f'{avg_price:.4f}' if avg_price else 'N/A'} | "
                f"action={final_action} | votes={dict(Counter(a for _,a in results.values() if a))}"
            )
            return (float(avg_price) if avg_price is not None else None), final_action

        # ── SINGLE MODEL ──────────────────────────────
        fn = RUNNERS.get(model_type.lower())
        if fn is None:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        price, action = fn(asset, df, feature_cols, last_close)

        if price is None and model_type != "rl":
            logging.warning(f"[{asset}][{model_type}] model file not found")
            return None, "Hold"

        action = action or _get_action(price, last_close)
        logging.info(
            f"[{asset}][{model_type}] last_close={last_close:.4f} | "
            f"predicted={f'{price:.4f}' if price else 'N/A'} | action={action}"
        )
        return (float(price) if price else None), action

    except Exception as e:
        logging.error(f"predict_asset failed for {filename}: {e}", exc_info=True)
        return None, "Hold"


def predict_portfolio(asset_files, model_type="ensemble"):
    return {
        f.replace(".csv", ""): {
            "predicted_price": p,
            "action": a,
        }
        for f in asset_files
        for p, a in [predict_asset(f, model_type=model_type)]
    }


if __name__ == "__main__":
    for f in ["ETEL.csv", "COMI.csv", "FWRY.csv"]:
        price, action = predict_asset(f, model_type="ensemble")
        logging.info(f"FINAL {f}: price={price}, action={action}")