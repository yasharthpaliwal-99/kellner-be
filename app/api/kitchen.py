"""Kitchen: GET orders; table POST order-ops; kitchen POST line-status (device session)."""
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from app.services.device_auth_service import validate_device_session
from app.services.kitchen_service import (
    DISH_STATUSES,
    fetch_kitchen_snapshot,
    hotel_id_query_variants,
    update_kitchen_line_dish_status,
    update_order_table_ops,
)

router = APIRouter()

SpiceLevel = Literal["mild", "low", "medium", "high"]
DishStatus = Literal["queued", "preparing", "cooking", "arriving", "ready", "served"]


def _session_hotel_matches(sess_hotel: object, body_hotel_id: str) -> bool:
    variants = hotel_id_query_variants(body_hotel_id.strip())
    sh = sess_hotel
    return sh in variants or str(sh).strip() == str(body_hotel_id).strip()


class TableOrderOpsBody(BaseModel):
    hotel_id: str
    table_number: int
    request_text: Optional[str] = Field(
        None,
        description="Service note (cutlery, table cleanup, etc.) → order.requests[]",
    )
    dish_name: Optional[str] = Field(None, description="Menu line name for spice (voice/order UI).")
    spice_level: Optional[SpiceLevel] = Field(None, description="Per-dish spice on matching line_items.")

    @model_validator(mode="after")
    def require_something(self) -> "TableOrderOpsBody":
        has_request = bool((self.request_text or "").strip())
        has_spice = self.spice_level is not None and bool((self.dish_name or "").strip())
        if not has_request and not has_spice:
            raise ValueError("Provide request_text and/or dish_name with spice_level.")
        if self.spice_level is not None and not (self.dish_name or "").strip():
            raise ValueError("dish_name is required when spice_level is set.")
        return self


class KitchenLineStatusBody(BaseModel):
    """Logged-in kitchen device: progress per dish for a table's active order."""

    hotel_id: str
    table_number: int
    dish_name: str
    dish_status: DishStatus = Field(
        ...,
        description=f"One of: {', '.join(DISH_STATUSES)}",
    )


@router.get("/kitchen")
def get_kitchen(
    hotel_id: str = Query(...),
    date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Kitchen board: orders for hotel.
    line_items may include spice_level (table order-ops) and dish_status (kitchen line-status).
    """
    hid = (hotel_id or "").strip()
    if not hid:
        raise HTTPException(status_code=400, detail="hotel_id is required.")
    data, err = fetch_kitchen_snapshot(hid, date_param=date)
    if err:
        if "not configured" in (err or "").lower():
            raise HTTPException(status_code=503, detail=err)
        raise HTTPException(status_code=400, detail=err)
    return data


@router.post("/kitchen/order-ops")
def post_kitchen_order_ops(body: TableOrderOpsBody):
    """Table device: service requests + per-dish spice on draft order for this table."""
    hid = body.hotel_id.strip()
    data, err = update_order_table_ops(
        hid,
        int(body.table_number),
        request_text=body.request_text,
        dish_name=body.dish_name,
        spice_level=body.spice_level,
    )
    if err:
        if "not configured" in (err or "").lower():
            raise HTTPException(status_code=503, detail=err)
        if "not found" in (err or "").lower() or "no draft" in (err or "").lower():
            raise HTTPException(status_code=404, detail=err)
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True, "order": data}


@router.post("/kitchen/line-status")
def post_kitchen_line_status(
    body: KitchenLineStatusBody,
    x_device_session: str = Header(..., alias="X-Device-Session"),
):
    """
    Kitchen device only: after POST /device/login with role=kitchen.
    Sets line_items[].dish_status by dish name on latest draft|confirmed order for hotel_id + table_number.
    Table/guest UI polls GET /kitchen and reads line_items for their table.
    """
    sess, serr = validate_device_session((x_device_session or "").strip(), role="kitchen")
    if serr:
        raise HTTPException(status_code=401, detail=serr)
    if not _session_hotel_matches(sess.get("hotel_id"), body.hotel_id):
        raise HTTPException(status_code=403, detail="hotel_id does not match kitchen session.")

    hid = body.hotel_id.strip()
    data, err = update_kitchen_line_dish_status(
        hid,
        int(body.table_number),
        body.dish_name.strip(),
        body.dish_status,
    )
    if err:
        if "not configured" in (err or "").lower():
            raise HTTPException(status_code=503, detail=err)
        if "not found" in (err or "").lower() or "no active order" in (err or "").lower():
            raise HTTPException(status_code=404, detail=err)
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True, "order": data}
