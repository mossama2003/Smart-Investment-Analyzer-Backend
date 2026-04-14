"""
main.py

ملف رئيسي لتشغيل المشروع كله:
1️⃣ تحميل البيانات من CSV
2️⃣ إضافة Features
3️⃣ (اختياري) تدريب الموديلات
4️⃣ سؤال المستخدم عن السهم والموديل
5️⃣ عمل Prediction وعرض النتيجة
"""

from fastapi import FastAPI
from typing import List
import sys
from pathlib import Path

# إضافة src للـ path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import predict_asset, predict_portfolio

from auth.database import SessionLocal
from auth.models import User, Asset
from auth.auth import hash_password, verify_password
from auth.models import Asset
from api.enums import AssetEnum
from src.predict import predict_asset

from fastapi import HTTPException

from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ===== Root =====
@app.get("/")
def home():
    return {"message": "Welcome to the Stock Prediction API! Use /predict for single asset or /predict-portfolio for multiple assets."}

@app.post("/register")
def register(username: str, email: str, password: str):
    db = SessionLocal()

    try:
        print("🔥 Register called")

        user = db.query(User).filter(User.email == email).first()
        print("✅ Checked existing user")

        if user:
            raise HTTPException(status_code=400, detail="Email already exists")

        hashed = hash_password(password)
        print("✅ Password hashed")

        new_user = User(
            username=username,
            email=email,
            password=hashed
        )

        db.add(new_user)
        db.commit()
        print("✅ User added")

        return {"message": "User created successfully"}

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

@app.post("/login")
def login(email: str, password: str):
    db = SessionLocal()

    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful"}

@app.get("/assets")
def get_assets():
    db = SessionLocal()
    assets = db.query(Asset).all()

    result = []
    for a in assets:
        result.append({
            "id": a.id,
            "name": a.name,
            "image": a.image_url
        })

    db.close()
    return result

@app.get("/asset/{asset_id}")
def get_asset(asset_id: int):
    db = SessionLocal()

    asset = db.query(Asset).filter(Asset.id == asset_id).first()

    if not asset:
        return {"error": "Asset not found"}

    db.close()

    return {
        "id": asset.id,
        "name": asset.name,
        "symbol": asset.symbol,
        "image": f"http://127.0.0.1:8000/static/images/{asset.image_url}"        
        }



@app.get("/predict/{asset_id}")
def predict(asset_id: int):
    db = SessionLocal()

    try:
        # ===== 1) نجيب السهم من الداتابيز =====
        asset = db.query(Asset).filter(Asset.id == asset_id).first()

        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        # ===== 2) نحول ل filename =====
        filename = f"{asset.name}.csv"

        # ===== 3) نعمل prediction =====
        price, action = predict_asset(filename)

        # ===== 4) نرجع النتيجة =====
        return {
            "asset_id": asset.id,
            "asset_name": asset.name,
            "predicted_price": price,
            "action": action
        }

    except Exception as e:
        print("❌ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

# ===== Predict Portfolio =====
@app.post("/predict-portfolio")
def predict_multiple(files: List[str], model_type: str = "xgboost"):
    results = predict_portfolio(files, model_type)

    return {
        "model": model_type,
        "results": results
    }


import os
import sys
from pathlib import Path
import logging
import pandas as pd

# ===== إضافة مجلد src للـ Python path =====
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_all_assets
from features import add_features
from predict import predict_asset

# ===== إعداد Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== ملفات البيانات =====
asset_files = ["ETEL.csv", "COMI.csv", "FWRY.csv"]

# ===== تحميل البيانات =====
logging.info("Loading assets...")
all_assets_data = load_all_assets(asset_files)


# ===== (اختياري) تدريب =====
def train_models():
    logging.info("Training XGBoost models...")
    try:
        import train_xgb
    except ImportError:
        logging.warning("train_xgb.py not found")

    logging.info("Training RL models...")
    try:
        import train_rl
    except ImportError:
        logging.warning("train_rl.py not found")

