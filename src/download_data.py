# src/download_data.py

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ===== Path =====
data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

# ===== Symbols =====
symbols = ["ETEL.CA", "COMI.CA", "FWRY.CA"]


# ===== Clean DataFrame =====
def clean_dataframe(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])

    # Strip timezone so concat with existing CSV doesn't break
    df["Date"] = df["Date"].dt.tz_localize(None)

    return df


# ===== Get last saved date =====
def get_last_date(file_path):
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    if df.empty or "Date" not in df.columns:
        return None
    return pd.to_datetime(df["Date"]).max()


# ===== Check if today is likely a trading day =====
def is_likely_trading_day():
    """EGX trades Sunday–Thursday. Closed Friday & Saturday."""
    # Monday=0, Tuesday=1, Wednesday=2, Thursday=3,
    # Friday=4, Saturday=5, Sunday=6
    return datetime.today().weekday() not in (4, 5)


# ===== Update a single symbol =====
def update_symbol(sym):
    print(f"\n📥 Updating {sym}...")

    file_path = data_dir / f"{sym.split('.')[0]}.csv"
    last_date = get_last_date(file_path)
    today     = datetime.today().date()

    if last_date is None:
        print("🆕 First download — fetching full history...")
        df_new = yf.download(sym, period="max", auto_adjust=True, progress=False)
    else:
        last_date_d = last_date.date()

        if last_date_d >= today:
            print(f"✅ Already up to date (last: {last_date_d})")
            return

        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"📅 Last saved: {last_date_d} → fetching {start_date} to {today}")
        df_new = yf.download(sym, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)

    # ===== Handle empty result =====
    if df_new is None or df_new.empty:
        if not is_likely_trading_day():
            day_name = datetime.today().strftime("%A")
            print(f"⚠️  No new data — today is {day_name}, EGX is closed (trades Sun–Thu).")
        else:
            print(f"⚠️  No new data returned for {sym}.")
            print(f"    Possible reasons:")
            print(f"      • Public holiday (EGX closed today)")
            print(f"      • yfinance data delay — try again in a few hours")
            print(f"      • Symbol may have changed — check finance.yahoo.com/quote/{sym}")
        return

    # ===== Clean =====
    df_new = clean_dataframe(df_new)

    # ===== Merge with existing =====
    if file_path.exists():
        df_old = pd.read_csv(file_path)
        df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.tz_localize(None)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # ===== Deduplicate + sort =====
    df_all.drop_duplicates(subset=["Date"], inplace=True)
    df_all.sort_values(by="Date", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # ===== Save =====
    df_all.to_csv(file_path, index=False)
    print(f"✅ {sym} updated — {len(df_new)} new rows added  "
          f"(total: {len(df_all)} rows, latest: {df_all['Date'].iloc[-1].date()})")


# ===== Update all =====
def update_all_data():
    print(f"🕐 Starting data update — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    for sym in symbols:
        try:
            update_symbol(sym)
        except Exception as e:
            print(f"❌ Failed to update {sym}: {e}")
    print("\n🏁 Data update complete.")


if __name__ == "__main__":
    update_all_data()