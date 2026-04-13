from auth.database import engine, Base
import auth.models

Base.metadata.create_all(bind=engine)