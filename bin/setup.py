"""
setup.py

ملف إعداد المشروع (Cross-platform: Windows + Mac/Linux)
وظيفته:
1️⃣ التأكد من نسخة Python >= 3.10
2️⃣ إنشاء Virtual Environment
3️⃣ تثبيت المكتبات المطلوبة داخل الـ venv (إذا مش موجودة)
4️⃣ إنشاء مجلدات البيانات والموديلات
5️⃣ يعطي رسالة جاهزية للتشغيل

تشغيل:
python setup.py
"""

import os
import sys
import subprocess
import platform

# ===== إعدادات المشروع =====
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "tensorflow",
    "stable-baselines3",
    "gymnasium",
    "joblib",
    "yfinance"
]

MODEL_DIRS = ["models/xgboost", "models/lstm", "models/gru", "models/rl"]
DATA_DIR = "data/raw"
VENV_NAME = "env"

# ===== التحقق من نسخة Python =====
def check_python_version():
    if sys.version_info < (3, 10):
        print("Python 3.10 أو أعلى مطلوب لتشغيل المشروع.")
        sys.exit(1)

# ===== إنشاء مجلدات البيانات والموديلات =====
def create_folders():
    os.makedirs(DATA_DIR, exist_ok=True)
    for d in MODEL_DIRS:
        os.makedirs(d, exist_ok=True)

# ===== إنشاء Virtual Environment =====
def create_venv():
    if not os.path.exists(VENV_NAME):
        print("===== إنشاء Virtual Environment =====")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_NAME])
    else:
        print("Virtual Environment موجود مسبقاً.")

# ===== تثبيت مكتبة واحدة إذا مش موجودة =====
def install_package(pip_path, package):
    try:
        subprocess.check_call([pip_path, "show", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{package} مثبت بالفعل ✅")
    except subprocess.CalledProcessError:
        print(f"تثبيت {package} ...")
        subprocess.check_call([pip_path, "install", "--upgrade", package])

# ===== تثبيت جميع المكتبات داخل الـ venv =====
def install_packages():
    print("===== تثبيت المكتبات المطلوبة داخل الـ venv =====")
    if platform.system() == "Windows":
        pip_path = os.path.join(VENV_NAME, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(VENV_NAME, "bin", "pip")
    
    for pkg in REQUIRED_PACKAGES:
        install_package(pip_path, pkg)

# ===== Main =====
if __name__ == "__main__":
    print("===== بدء إعداد المشروع =====")
    check_python_version()
    create_venv()
    install_packages()
    create_folders()
    
    print("\n===== تم إعداد المشروع بنجاح! =====")
    if platform.system() == "Windows":
        print(f"لتفعيل البيئة: {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"لتفعيل البيئة: source {VENV_NAME}/bin/activate")
    print("لتشغيل المشروع: python main.py")