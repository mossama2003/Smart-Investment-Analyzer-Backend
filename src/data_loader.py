"""
data_loader.py
نسخة متطورة للتعامل مع ملفات CSV اللي ممكن يكون فيها عمود التاريخ باسم مختلف.
"""

import pandas as pd
import logging
from pathlib import Path
import os
from features import add_features  # استدعاء ملف features.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# قائمة أسماء الأعمدة اللي ممكن تمثل التاريخ
POSSIBLE_DATE_COLS = ['Date', 'date', 'Timestamp', 'timestamp']

def clean_yahoo_multiheader(df):
    """
    تنظيف ملفات Yahoo Finance اللي فيها multi-header
    """
    try:
        # لو الأعمدة MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # لو أول صف فيه كلمة Date (زي حالتك)
        if "Date" in df.iloc[0].values:
            df = df.drop(index=0).reset_index(drop=True)

        # إعادة تسمية الأعمدة لو اسمها غلط
        if "Price" in df.columns:
            df = df.rename(columns={"Price": "Date"})

        # ترتيب الأعمدة
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[[col for col in expected_cols if col in df.columns]]

        return df

    except Exception as e:
        logging.warning(f"Error cleaning Yahoo format: {e}")
        return df


def load_asset_data(filename):
    file_path = RAW_DATA_DIR / filename

    try:
        # نحاول نقرأ Multi-header الأول
        df = pd.read_csv(file_path, header=[0, 1])
    except:
        df = pd.read_csv(file_path)

    # 🔥 تنظيف الداتا (المهم هنا)
    df = clean_yahoo_multiheader(df)

    # البحث عن عمود التاريخ
    date_col = None
    for col in POSSIBLE_DATE_COLS:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"CSV file {filename} must contain a date column (one of {POSSIBLE_DATE_COLS})")

    # تحويل عمود التاريخ
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # التأكد من الأعمدة الرقمية
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file {filename} must contain '{col}' column")
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # ترتيب الأعمدة النهائي
    df = df[[date_col, 'Open', 'High', 'Low', 'Close', 'Volume']]

    # ترتيب البيانات حسب التاريخ
    df = df.sort_values(date_col).reset_index(drop=True)

    return df

def load_all_assets(asset_files):
    all_data = {}
    for file in asset_files:
        asset_name = os.path.splitext(file)[0]
        df = load_asset_data(file)
        if df is not None:
            try:
                df_with_features = add_features(df)
                all_data[asset_name] = df_with_features
                logging.info(f"Loaded and added features for {asset_name}")
            except Exception as e:
                logging.warning(f"Skipping {asset_name} due to error in feature calculation: {e}")
    return all_data

# ===== Example Usage =====
if __name__ == "__main__":
    asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]
    all_assets_data = load_all_assets(asset_files)

    CLEAN_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for asset, df in all_assets_data.items():
        save_path = CLEAN_DATA_DIR / f"{asset}_clean.csv"
        df.to_csv(save_path, index=False)
        logging.info(f"Saved cleaned file: {save_path}")

    for asset, df in all_assets_data.items():
        logging.info(f"===== {asset} =====\n{df.head()}")



    # مثال Asset واحد + Features
    df = load_asset_data("ETEL.csv")
    if df is not None:
        df_with_features = add_features(df)
        logging.info(f"===== ETEL with Features =====\n{df_with_features.head()}")