from apscheduler.schedulers.background import BackgroundScheduler
from src.download_data import update_all_data
from src.train_all import train_all_models
import logging

# لو عندك train_all بعد كده
def train_all_models():
    print("🤖 Training models...")
    # هنا هنضيف التدريب بعدين

scheduler = BackgroundScheduler()

# ===== Job 1: تحديث يومي =====
scheduler.add_job(
    update_all_data,
    trigger='cron',
    hour=1
)


# ===== Job 2: تدريب أسبوعي =====
scheduler.add_job(
    train_all_models,
    trigger='cron',
    day_of_week='sun',
    hour=3
)

def start_scheduler():
    scheduler.start()
    print("🚀 Scheduler started...")