import sys
from pathlib import Path
import logging
import pandas as pd

# إضافة src للـ path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from predict import predict_asset

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]


# ===== User Interaction =====
def ask_user():
    print("\n📊 Available Assets:")
    for i, asset in enumerate(asset_files, 1):
        print(f"{i}. {asset.replace('.csv', '')}")

    try:
        choice = int(input("\n👉 Choose asset number: "))
        selected_file = asset_files[choice - 1]
    except:
        print("❌ Invalid choice")
        return None, None

    model_type = input("🤖 Choose model (xgboost / lstm / gru / rl): ").lower()

    return selected_file, model_type


# ===== Main Flow =====
def main():
    selected_file, model_type = ask_user()

    if not selected_file:
        return

    logging.info("Running prediction...")

    price, action = predict_asset(selected_file, model_type=model_type)

    print("\n📈 Result:")
    print(f"Asset: {selected_file.replace('.csv','')}")
    print(f"Predicted Price: {price}")
    print(f"Recommended Action: {action}")

    result_df = pd.DataFrame([{
        "Asset": selected_file.replace(".csv",""),
        "Predicted_Price": price,
        "Action": action
    }])

    result_df.to_csv("prediction_result.csv", index=False)
    logging.info("Result saved to prediction_result.csv")


if __name__ == "__main__":
    main()