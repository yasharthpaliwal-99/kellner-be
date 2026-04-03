"""
Phase 1: Face recognize (detect -> identify) using Azure Face API.

Goal:
  - Input: one image frame
  - Output: matched Azure personId (\"persons_id\") or null if no match

Notes:
  - This script uses Azure Face recognition operations (not generic Vision OCR/caption).
  - Identify is done against PersonDirectory using personIds=["*"] by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


FACE_API_VERSION = "v1.2"

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def detect_faces(
    *,
    endpoint: str,
    key: str,
    image_bytes: bytes,
    recognition_model: str,
) -> List[Dict[str, Any]]:
    # Some environments set HTTPS_PROXY/HTTP_PROXY; bypass to avoid tunnel failures.
    sess = requests.Session()
    sess.trust_env = False
    url = f"{endpoint.rstrip('/')}/face/{FACE_API_VERSION}/detect"
    params = {
        "returnFaceId": "true",
        "recognitionModel": recognition_model,
    }
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }
    resp = sess.post(url, params=params, headers=headers, data=image_bytes, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def identify_from_person_directory(
    *,
    endpoint: str,
    key: str,
    face_ids: List[str],
    person_ids: List[str],
    confidence_threshold: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    sess = requests.Session()
    sess.trust_env = False
    url = f"{endpoint.rstrip('/')}/face/{FACE_API_VERSION}/identify"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "faceIds": face_ids,
        "personIds": person_ids,
        "confidenceThreshold": confidence_threshold,
        "maxNumOfCandidatesReturned": max_candidates,
    }
    resp = sess.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure Face recognize: detect -> identify")
    parser.add_argument("--image", required=True, help="Path to an image file (JPG/PNG/etc).")
    parser.add_argument(
        "--debug-env",
        action="store_true",
        help="Print loaded endpoint + key length (never prints the key).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_FACE_ENDPOINT"),
        help="Face endpoint base URL, e.g. https://<resource>.cognitiveservices.azure.com/",
    )
    parser.add_argument(
        "--key",
        default=os.getenv("AZURE_FACE_KEY"),
        help="Face subscription key. Better via env var AZURE_FACE_KEY.",
    )
    parser.add_argument(
        "--recognition-model",
        default=os.getenv("AZURE_FACE_RECOGNITION_MODEL", "recognition_04"),
        help="Must match the recognition model used during enrollment (e.g. recognition_04).",
    )
    parser.add_argument(
        "--person-ids",
        default=os.getenv("AZURE_FACE_PERSON_IDS", "*"),
        help='Array of personIds to match against, or "*" to search entire directory. Default: "*".',
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.getenv("AZURE_FACE_CONFIDENCE_THRESHOLD", "0.7")),
        help="Identification confidence threshold in [0,1].",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=int(os.getenv("AZURE_FACE_MAX_CANDIDATES", "5")),
        help="Max candidate persons returned per face.",
    )
    args = parser.parse_args()

    endpoint = args.endpoint or _require_env("AZURE_FACE_ENDPOINT")
    key = args.key or _require_env("AZURE_FACE_KEY")
    if args.debug_env:
        print(json.dumps({"AZURE_FACE_ENDPOINT": endpoint, "AZURE_FACE_KEY_LEN": len(key or "")}))

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    faces = detect_faces(
        endpoint=endpoint,
        key=key,
        image_bytes=image_bytes,
        recognition_model=args.recognition_model,
    )
    face_ids = [f.get("faceId") for f in faces if isinstance(f, dict) and f.get("faceId")]

    if not face_ids:
        print(json.dumps({"matched": False, "personId": None, "reason": "no_face_detected"}))
        return

    person_ids: List[str]
    if (args.person_ids or "").strip() == "*":
        person_ids = ["*"]
    else:
        # Allow comma-separated list
        person_ids = [x.strip() for x in str(args.person_ids).split(",") if x.strip()]
        if not person_ids:
            person_ids = ["*"]

    ident_results = identify_from_person_directory(
        endpoint=endpoint,
        key=key,
        face_ids=face_ids,
        person_ids=person_ids,
        confidence_threshold=args.confidence_threshold,
        max_candidates=args.max_candidates,
    )

    # Choose the best candidate across all detected faces.
    best: Optional[Dict[str, Any]] = None
    for item in ident_results:
        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            continue
        for c in candidates:
            if not isinstance(c, dict):
                continue
            pid = c.get("personId")
            conf = c.get("confidence")
            if not pid:
                continue
            if best is None or (isinstance(conf, (int, float)) and conf > best.get("confidence", -1)):
                best = {"personId": pid, "confidence": conf}

    if not best:
        print(json.dumps({"matched": False, "personId": None, "reason": "no_match"}))
        return

    print(json.dumps({"matched": True, "personId": best["personId"], "confidence": best.get("confidence")}))


if __name__ == "__main__":
    main()

