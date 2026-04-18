"""
diagnose.py — run from the project ROOT:
    python diagnose.py

Tests ensemble prediction for each asset and shows a per-model breakdown
so you can see exactly how each model voted and what price it predicted.
"""

import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during diag

from data_loader import load_asset_data
from features    import add_features

ASSETS = ["ETEL", "COMI", "FWRY"]

def run():
    try:
        from predict import predict_asset, RUNNERS, MODEL_WEIGHTS, _combine, _get_features
        from features import add_features
        from data_loader import load_asset_data
    except ImportError as e:
        print(f"ERROR importing predict.py: {e}")
        return

    print("\n" + "="*70)
    print("ENSEMBLE PREDICTION DIAGNOSTIC")
    print("="*70)

    for asset in ASSETS:
        filename = f"{asset}.csv"
        df = load_asset_data(filename)
        if df is None:
            print(f"\n{asset}: ❌ could not load CSV")
            continue

        df           = add_features(df)
        feature_cols = _get_features(df)
        last_close   = float(df['Close'].iloc[-1])
        last_date    = str(df.index[-1]) if df.index.name == 'Date' else str(df['Date'].iloc[-1]) if 'Date' in df.columns else "?"

        print(f"\n── {asset}  (last_date={last_date}  last_close={last_close:.4f}) ──")

        results = {}
        for name, fn in RUNNERS.items():
            try:
                price, action = fn(asset, df, feature_cols, last_close)
                results[name] = (price, action)
                w = MODEL_WEIGHTS.get(name, 1)
                p_str = f"{price:.4f}" if price is not None else "N/A (action only)"
                ok = ""
                if price is not None:
                    diff = abs(price - last_close) / last_close * 100
                    ok = "✅" if diff < 10 else ("⚠️ " if diff < 30 else "❌")
                print(f"  {ok:3s} [{name:8s}] price={p_str:>12s}  action={action}  weight={w}")
            except Exception as e:
                results[name] = (None, None)
                print(f"  ❓  [{name:8s}] skipped → {e}")

        avg_price, final_action, _ = _combine(results, last_close)

        pct = abs(avg_price - last_close) / last_close * 100 if avg_price else None
        ok  = ("✅" if pct < 10 else ("⚠️ " if pct < 30 else "❌")) if pct is not None else "❓"
        print(f"\n  {ok} ENSEMBLE → predicted={f'{avg_price:.4f}' if avg_price else 'N/A'}  "
              f"diff={f'{pct:.1f}%' if pct else 'N/A'}  action={final_action}")

    print("\n" + "="*70)
    print("Done.")


if __name__ == "__main__":
    run()