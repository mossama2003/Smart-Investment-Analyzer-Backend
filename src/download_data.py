# src/download_data.py

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ===== المسار =====
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

# ===== الأسهم =====
symbols = ["ETEL.CA", "COMI.CA", "FWRY.CA"]


# ===== تجهيز الداتا (حل كل المشاكل) =====
def clean_dataframe(df):
    # ✅ لو MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ✅ تحويل Date من index لـ column
    df = df.reset_index()

    # ✅ تأكيد اسم Date
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # ✅ إزالة أي أعمدة غريبة
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # ✅ تحويل Date لـ datetime
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ===== آخر تاريخ =====
def get_last_date(file_path):
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    if df.empty or "Date" not in df.columns:
        return None

    return pd.to_datetime(df["Date"]).max()


# ===== تحديث سهم واحد =====
def update_symbol(sym):
    print(f"\n📥 Updating {sym}...")

    file_path = data_dir / f"{sym.split('.')[0]}.csv"

    last_date = get_last_date(file_path)

    # ===== تحديد بداية التحميل =====
    if last_date is None:
        print("🆕 First download...")
        df_new = yf.download(sym, period="max")
    else:
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")

        print(f"📅 Last date: {last_date.date()} → updating from {start_date}")

        df_new = yf.download(sym, start=start_date, end=end_date)

    # ===== لو فاضية =====
    if df_new.empty:
        print("⚠️ No new data")
        return

    # ===== تنظيف =====
    df_new = clean_dataframe(df_new)

    # ===== دمج =====
    if file_path.exists():
        df_old = pd.read_csv(file_path)
        df_old["Date"] = pd.to_datetime(df_old["Date"])

        df_all = pd.concat([df_old, df_new])
    else:
        df_all = df_new

    # ===== إزالة التكرار =====
    df_all.drop_duplicates(subset=["Date"], inplace=True)

    # ===== ترتيب =====
    df_all.sort_values(by="Date", inplace=True)

    # ===== حفظ =====
    df_all.to_csv(file_path, index=False)

    print(f"✅ Updated {sym} → {len(df_new)} rows")


# ===== تحديث الكل =====
def update_all_data():
    for sym in symbols:
        update_symbol(sym)


# ===== تشغيل مباشر =====
if __name__ == "__main__":
    update_all_data()