from auth.database import SessionLocal
from auth.models import Asset

db = SessionLocal()

assets = [
    {"id": 1, "name": "COMI", "symbol": "COMI.CA", "image_url": "/static/images/comi.png"},
    {"id": 2, "name": "ETEL", "symbol": "ETEL.CA", "image_url": "/static/images/etel.png"},
    {"id": 3, "name": "FWRY", "symbol": "FWRY.CA", "image_url": "/static/images/fwry.png"},
]

for a in assets:
    asset = Asset(**a)
    db.add(asset)

db.commit()
db.close()

print("✅ Assets added")