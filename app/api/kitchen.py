"""Kitchen read API: list orders + stats for a hotel from Mongo."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.kitchen_service import fetch_kitchen_snapshot

router = APIRouter()


@router.get("/kitchen")
def get_kitchen(
    hotel_id: str = Query(..., description="Hotel identifier (matches int or string in Mongo)."),
    date: Optional[str] = Query(
        None,
        description="UTC date YYYY-MM-DD. Restricts to orders whose created_at falls on that calendar day (reduces payload). Omit to return all orders for the hotel.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Read-only: exact order documents from Mongo for this hotel, plus stats for the same result set.
    Optional `date` filters by `created_at` (UTC day) to limit rows.
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
