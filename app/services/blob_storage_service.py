from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContentSettings

from app.config import config


def _guess_ext(filename: Optional[str], content_type: Optional[str]) -> str:
    name = (filename or "").strip().lower()
    ext = Path(name).suffix
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ext
    c = (content_type or "").lower()
    if c == "image/jpeg":
        return ".jpg"
    if c == "image/png":
        return ".png"
    if c == "image/webp":
        return ".webp"
    if c == "image/gif":
        return ".gif"
    return ".jpg"


def upload_menu_image_bytes(
    image_bytes: bytes,
    *,
    hotel_id: int,
    dish_id: int,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload menu image to Azure Blob Storage and return public blob URL.
    """
    conn_str = config.AZURE_BLOB_CONNECTION_STRING
    if not conn_str:
        raise ValueError("Set AZURE_BLOB_CONNECTION_STRING in .env")
    container = (config.AZURE_BLOB_CONTAINER_NAME or "menu-images").strip()
    if not container:
        raise ValueError("Set AZURE_BLOB_CONTAINER_NAME in .env")
    if not image_bytes:
        raise ValueError("Empty image bytes.")

    service = BlobServiceClient.from_connection_string(conn_str)
    container_client = service.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass

    ext = _guess_ext(filename, content_type)
    blob_name = f"hotel-{hotel_id}/dish-{dish_id}/{uuid.uuid4().hex}{ext}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        image_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type or "image/jpeg"),
    )
    return blob_client.url

