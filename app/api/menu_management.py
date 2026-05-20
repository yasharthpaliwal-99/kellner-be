"""Staff menu: list dishes and update availability (Postgres menu_items.available)."""
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.services.blob_storage_service import upload_menu_image_bytes
from app.services.device_auth_service import validate_device_session
from app.services.menu_management_service import (
    _same_hotel,
    apply_availability_updates,
    fetch_menu_rows,
    update_menu_image_url,
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
    x_device_session: Optional[str] = Header(None, alias="X-Device-Session"),
):
    """
    Returns all menu rows for the hotel: dish_id, name, price, available, image.
    Works without auth (conversation page) or with a kitchen session (kitchen dashboard).
    When a kitchen session is provided, hotel_id must match.
    """
    hid = _parse_hotel_id(payload.hotel_id)
    sid = (x_device_session or "").strip()
    if sid:
        sess = _require_kitchen_session(x_device_session)
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


@router.post("/upload_menu_image")
async def upload_menu_image(
    hotel_id: int = Form(...),
    dish_id: int = Form(...),
    image: UploadFile = File(...),
    x_device_session: str = Header(..., alias="X-Device-Session"),
):
    """
    Upload dish image to Azure Blob and update menu_items.image for (hotel_id, dish_id).
    Requires kitchen device session and matching hotel_id.
    """
    sess = _require_kitchen_session(x_device_session)
    hid = _parse_hotel_id(hotel_id)
    if dish_id < 1:
        raise HTTPException(status_code=400, detail="dish_id must be >= 1.")
    if not _same_hotel(sess.get("hotel_id"), hid):
        raise HTTPException(status_code=403, detail="Session hotel_id does not match request.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image.")
    ctype = (image.content_type or "").lower()
    if not ctype.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed.")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB).")

    try:
        blob_url = upload_menu_image_bytes(
            data,
            hotel_id=hid,
            dish_id=dish_id,
            filename=image.filename,
            content_type=image.content_type,
        )
        row = update_menu_image_url(hid, dish_id, blob_url)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if row is None:
        raise HTTPException(status_code=404, detail="dish_id not found for this hotel.")
    return {"ok": True, "hotel_id": hid, "item": row}
