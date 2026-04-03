"""HTTP tests for POST /api/face/local/recognise (mocked auth + face logic)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class TestFaceRecognise(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_recognise_matched_ok(self) -> None:
        with patch(
            "app.api.face_local.validate_device_session",
            return_value=({"hotel_id": 1}, None),
        ):
            with patch(
                "app.api.face_local.recognise_me",
                return_value=(42, True, 0.22, False, None),
            ):
                r = self.client.post(
                    "/api/face/local/recognise",
                    headers={"x-device-session": "test-sid"},
                    files={"image": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")},
                )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["customer_id"], 42)
        self.assertTrue(data["matched"])
        self.assertFalse(data["created_new"])
        self.assertEqual(data["distance"], 0.22)

    def test_recognise_new_guest_ok(self) -> None:
        with patch(
            "app.api.face_local.validate_device_session",
            return_value=({"hotel_id": 1}, None),
        ):
            with patch(
                "app.api.face_local.recognise_me",
                return_value=(99, False, 0.55, True, None),
            ):
                r = self.client.post(
                    "/api/face/local/recognise",
                    headers={"x-device-session": "test-sid"},
                    files={"image": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")},
                )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["customer_id"], 99)
        self.assertFalse(data["matched"])
        self.assertTrue(data["created_new"])

    def test_recognise_unauthorized(self) -> None:
        with patch(
            "app.api.face_local.validate_device_session",
            return_value=(None, "invalid session"),
        ):
            r = self.client.post(
                "/api/face/local/recognise",
                headers={"x-device-session": "bad"},
                files={"image": ("x.jpg", b"x", "image/jpeg")},
            )
        self.assertEqual(r.status_code, 401)

    def test_recognise_empty_image(self) -> None:
        with patch(
            "app.api.face_local.validate_device_session",
            return_value=({"hotel_id": 1}, None),
        ):
            r = self.client.post(
                "/api/face/local/recognise",
                headers={"x-device-session": "test-sid"},
                files={"image": ("empty.jpg", b"", "image/jpeg")},
            )
        self.assertEqual(r.status_code, 400)

    def test_recognise_face_error(self) -> None:
        with patch(
            "app.api.face_local.validate_device_session",
            return_value=({"hotel_id": 1}, None),
        ):
            with patch(
                "app.api.face_local.recognise_me",
                return_value=(None, False, None, False, "no face"),
            ):
                r = self.client.post(
                    "/api/face/local/recognise",
                    headers={"x-device-session": "test-sid"},
                    files={"image": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")},
                )
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())


if __name__ == "__main__":
    unittest.main()
