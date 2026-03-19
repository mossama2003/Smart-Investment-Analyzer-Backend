# src/download_data.py
import yfinance as yf
from pathlib import Path

# مجلد حفظ الملفات
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

# قائمة الأسهم
symbols = ["ETEL.CA", "COMI.CA", "FWRY.CA"]

for sym in symbols:
    print(f"Downloading {sym}...")
    df = yf.download(sym, start="2020-01-01", end="2026-03-20")
    
    # حفظ CSV
    file_path = data_dir / f"{sym.split('.')[0]}.csv"
    df.to_csv(file_path)
    
    print(f"{sym} saved to {file_path}")