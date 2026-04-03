"""Run once on VM after deploy if you need schema without starting the API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.face_schema import ensure_face_schema  # noqa: E402

if __name__ == "__main__":
    ensure_face_schema()
    print("face schema OK")
