"""Staff menu: list dishes and update availability (Postgres menu_items.available)."""
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.device_auth_service import validate_device_session
from app.services.menu_management_service import (
    _same_hotel,
    apply_availability_updates,
    fetch_menu_rows,
)

router = APIRouter()


def _parse_hotel_id(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="hotel_id must be an integer.")


def _require_kitchen_session(x_device_session: str):
    sid = (x_device_session or "").strip()
    if not sid:
        raise HTTPException(status_code=401, detail="Missing X-Device-Session header.")
    sess, err = validate_device_session(sid, role="kitchen")
    if err:
        raise HTTPException(status_code=401, detail=err)
    return sess


class FetchMenuRequest(BaseModel):
    hotel_id: Any

    @field_validator("hotel_id", mode="before")
    @classmethod
    def coerce_hotel(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            raise ValueError("hotel_id is required")
        return v


class SaveMenuItem(BaseModel):
    dish_id: int = Field(..., ge=1)
    available: bool


class SaveMenuRequest(BaseModel):
    hotel_id: Any
    items: List[SaveMenuItem] = Field(default_factory=list)

    @field_validator("hotel_id", mode="before")
    @classmethod
    def coerce_hotel(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not str(v).strip()):
            raise ValueError("hotel_id is required")
        return v


@router.post("/fetch_menu")
def fetch_menu(
    payload: FetchMenuRequest,
    x_device_session: str = Header(..., alias="X-Device-Session"),
):
    """
    Returns all menu rows for the hotel: dish_id, name, price, available.
    Requires a valid **kitchen** device session; session hotel must match payload hotel_id.
    """
    sess = _require_kitchen_session(x_device_session)
    hid = _parse_hotel_id(payload.hotel_id)
    if not _same_hotel(sess.get("hotel_id"), hid):
        raise HTTPException(status_code=403, detail="Session hotel_id does not match request.")
    try:
        items = fetch_menu_rows(hid)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, "hotel_id": hid, "items": items}


@router.post("/save_menu")
def save_menu(
    payload: SaveMenuRequest,
    x_device_session: str = Header(..., alias="X-Device-Session"),
):
    """
    Updates `available` only for listed dishes. Requires **kitchen** session; hotel must match.
    Unknown dish_id or wrong hotel → listed in `failed`, not a hard error unless all fail.
    """
    sess = _require_kitchen_session(x_device_session)
    hid = _parse_hotel_id(payload.hotel_id)
    if not _same_hotel(sess.get("hotel_id"), hid):
        raise HTTPException(status_code=403, detail="Session hotel_id does not match request.")

    if not payload.items:
        return {"ok": True, "hotel_id": hid, "updated": [], "failed": []}

    updates = [(it.dish_id, it.available) for it in payload.items]
    try:
        updated, failed = apply_availability_updates(hid, updates)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {"ok": True, "hotel_id": hid, "updated": updated, "failed": failed}
