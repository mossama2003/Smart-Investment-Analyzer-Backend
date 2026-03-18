"""
data_loader.py

هذا الملف مسؤول عن:
1️⃣ تحميل بيانات الأسهم، صناديق الأسهم، وصناديق الذهب من ملفات CSV الموجودة في فولدر data/raw/
2️⃣ تحويل التاريخ لصيغة datetime للتعامل مع Time Series
3️⃣ ترتيب البيانات حسب التاريخ
4️⃣ إعادة البيانات على شكل DataFrame جاهز للـ Feature Engineering والتدريب
5️⃣ دمج الـ Features مباشرة باستخدام features.py
"""

import os
import pandas as pd
import logging
from features import add_features  # استدعاء ملف features.py

# إعداد Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# المسار الافتراضي لمجلد البيانات الخام
RAW_DATA_DIR = "../data/raw/"

def load_asset_data(filename):
    """
    وظيفة الدالة:
    ----------------
    - تقرأ CSV لأسهم أو صناديق
    - تحويل العمود 'Date' لصيغة datetime
    - ترتيب البيانات حسب التاريخ
    - إعادة DataFrame جاهز للاستخدام

    Parameters:
    -----------
    filename : str
        اسم الملف داخل فولدر data/raw/ (مثال: "AAPL.csv")

    Returns:
    --------
    pd.DataFrame
        بيانات مرتبة حسب التاريخ وجاهزة للـ Feature Engineering
    """
    file_path = os.path.join(RAW_DATA_DIR, filename)

    # قراءة الملف مع dtype لتقليل استهلاك الذاكرة
    try:
        df = pd.read_csv(file_path, dtype={'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

    # التأكد من وجود العمود Date
    if 'Date' not in df.columns:
        raise ValueError("CSV file must contain 'Date' column")

    # التأكد من وجود الأعمدة الأساسية قبل Features
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain '{col}' column for Features calculation")

    # تحويل العمود 'Date' لصيغة datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # ترتيب البيانات حسب التاريخ تصاعدياً
    df = df.sort_values('Date').reset_index(drop=True)

    return df

def load_all_assets(asset_files):
    """
    وظيفة الدالة:
    ----------------
    - تحميل قائمة من الملفات (أسهم + صناديق)
    - إعادة قاموس Dictionary بالاسم: DataFrame لكل Asset
    - يضيف Features لكل Asset باستخدام features.py

    Parameters:
    -----------
    asset_files : list
        قائمة أسماء الملفات داخل data/raw/

    Returns:
    --------
    dict
        مثال: {"AAPL": DataFrame, "GOLD_ETF": DataFrame, ...} مع Features مضافة
    """
    all_data = {}
    for file in asset_files:
        asset_name = os.path.splitext(file)[0]  # إزالة الامتداد للحصول على الاسم
        df = load_asset_data(file)
        if df is not None:
            try:
                df_with_features = add_features(df)  # إضافة Features مباشرة
                all_data[asset_name] = df_with_features
                logging.info(f"Loaded and added features for {asset_name}")
            except Exception as e:
                logging.warning(f"Skipping {asset_name} due to error in feature calculation: {e}")
    return all_data

# ===== Example Usage =====
if __name__ == "__main__":
    asset_files = ["AAPL.csv", "EGX30.csv", "GOLD_ETF.csv"]  # مثال
    all_assets_data = load_all_assets(asset_files)

    for asset, df in all_assets_data.items():
        logging.info(f"===== {asset} =====\n{df.head()}")

    # مثال Asset واحد + Features
    df = load_asset_data("AAPL.csv")
    if df is not None:
        df_with_features = add_features(df)
        logging.info(f"===== AAPL with Features =====\n{df_with_features.head()}")