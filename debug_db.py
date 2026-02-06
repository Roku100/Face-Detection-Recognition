import sys
from pathlib import Path
import pickle

db_path = Path("data/encodings/face_database.db")
if db_path.exists():
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    print("Keys in database:", list(db.keys()))
else:
    print("No database found at", db_path)
