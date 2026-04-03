"""Local face embedding via InsightFace (CPU). Model files download on first use."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import config

_app: Optional[Any] = None


def _model_root() -> str:
    root = config.INSIGHTFACE_ROOT or os.getenv("INSIGHTFACE_ROOT")
    if root:
        return str(Path(root).expanduser())
    return str(Path(__file__).resolve().parent.parent.parent / ".insightface")


def get_face_app():
    global _app
    if _app is not None:
        return _app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise RuntimeError(
            "insightface is not installed. On the VM: pip install insightface onnxruntime opencv-python-headless"
        ) from e

    providers = ["CPUExecutionProvider"]
    _app = FaceAnalysis(name="buffalo_l", root=_model_root(), providers=providers)
    _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Returns L2-normalized 512-d embedding (float32) from first detected face.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")

    app = get_face_app()
    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in image.")
    face = faces[0]
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
    out = np.asarray(emb, dtype=np.float32).reshape(-1)
    if out.shape[0] != 512:
        raise ValueError(f"Unexpected embedding dim {out.shape[0]}, expected 512.")
    return out
