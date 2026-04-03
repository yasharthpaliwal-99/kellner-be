"""Local face recognise (InsightFace + pgvector). Requires device session."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.services.device_auth_service import validate_device_session
from app.services.face_local_service import recognise_me

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face/local", tags=["face-local"])


def _hotel_from_session(sess: dict) -> int:
    raw = sess.get("hotel_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


@router.post("/recognise")
async def face_local_recognise(
    image: UploadFile = File(...),
    x_device_session: str = Header(...),
):
    """
    Single "Recognise me" call: identify if possible; otherwise enroll and return customer_id.
    Client always gets ok + customer_id on success; `matched` vs `created_new` tells returning vs new.
    """
    sess, err = validate_device_session(x_device_session.strip(), role="device")
    if err:
        logger.warning("face_recognise_auth_failed error=%s", err)
        raise HTTPException(status_code=401, detail=err)
    hotel_id = _hotel_from_session(sess or {})
    data = await image.read()
    if not data:
        logger.warning("face_recognise_empty_image hotel_id=%s", hotel_id)
        raise HTTPException(status_code=400, detail="Empty image.")
    cid, matched, dist, created_new, emsg = recognise_me(data, hotel_id)
    if emsg or cid is None:
        logger.warning(
            "face_recognise_failed hotel_id=%s error=%s",
            hotel_id,
            emsg or "no customer_id",
        )
        raise HTTPException(status_code=400, detail=emsg or "Recognise failed.")
    # print: always visible in uvicorn stdout (logger.info may be hidden by log level)
    print(
        "[FACE_RECOGNISE] HTTP 200 — response sent to frontend:\n"
        f"  ok: True\n"
        f"  customer_id: {cid}  — who this guest is (existing or new row id)\n"
        f"  matched: {matched}  — True: matched existing face; False: no match before enroll\n"
        f"  created_new: {created_new}  — True: new customers row + embedding; False: reused existing\n"
        f"  distance: {dist}",
        flush=True,
    )
    logger.info(
        "face_recognise_ok customer_id=%s matched=%s created_new=%s distance=%s",
        cid,
        matched,
        created_new,
        dist,
    )
    return {
        "ok": True,
        "customer_id": cid,
        "matched": matched,
        "created_new": created_new,
        "distance": dist,
    }
