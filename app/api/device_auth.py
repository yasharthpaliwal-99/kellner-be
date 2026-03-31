"""Backend endpoints for long-lived hotel device sessions."""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from app.services.device_auth_service import login_hotel_device, validate_device_session

router = APIRouter()


class DeviceLoginRequest(BaseModel):
    hotel_id: Any
    password: str = Field(min_length=1)
    role: Literal["device", "kitchen"]
    table_number: Optional[int] = None
    device_id: Optional[str] = None


@router.post("/device/login")
def device_login(payload: DeviceLoginRequest):
    if payload.role == "device" and payload.table_number is None:
        raise HTTPException(status_code=400, detail="table_number is required for role=device.")
    sess, err = login_hotel_device(
        hotel_id=payload.hotel_id,
        password=payload.password,
        role=payload.role,
        table_number=payload.table_number,
        device_id=(payload.device_id or "").strip() or None,
    )
    if err:
        raise HTTPException(status_code=401, detail=err)
    return {
        "ok": True,
        "session_id": sess["session_id"],
        "hotel_id": sess["hotel_id"],
        "role": sess["role"],
        "table_number": sess.get("table_number"),
        "device_id": sess.get("device_id"),
    }


@router.get("/device/session")
def device_session(x_device_session: str = Header(...)):
    sess, err = validate_device_session(x_device_session)
    if err:
        raise HTTPException(status_code=401, detail=err)
    return {
        "ok": True,
        "hotel_id": sess.get("hotel_id"),
        "role": sess.get("role"),
        "table_number": sess.get("table_number"),
        "device_id": sess.get("device_id"),
    }
