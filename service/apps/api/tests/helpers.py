"""测试辅助：图片生成、注册/登录客户端工具。"""

from __future__ import annotations

import io
import random
import uuid

from fastapi.testclient import TestClient
from PIL import Image


def make_image_bytes(
    size: tuple[int, int] = (320, 240),
    image_format: str = "JPEG",
    *,
    texture: bool = True,
    color: tuple[int, int, int] | None = None,
) -> bytes:
    """生成图片字节：texture=True 时逐像素随机（高方差，通过模糊检查）。"""
    width, height = size
    if not texture:
        image = Image.new("RGB", size, color=color or (120, 120, 120))
    else:
        image = Image.new("RGB", size)
        pixels = image.load()
        rng = random.Random(7)
        for y in range(height):
            for x in range(width):
                pixels[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    stream = io.BytesIO()
    save_format = "JPEG" if image_format.upper() == "JPG" else image_format.upper()
    image.save(stream, format=save_format)
    return stream.getvalue()


def register(client: TestClient, email: str | None = None, password: str = "password123", name: str = "测试用户"):
    email = email or f"u{uuid.uuid4().hex[:10]}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "display_name": name},
    )
    assert response.status_code == 201, response.text
    data = response.json()
    return {
        "email": email,
        "password": password,
        "csrf_token": data["csrf_token"],
        "user": data["user"],
    }


def login(client: TestClient, email: str, password: str) -> dict:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    data = response.json()
    return {"email": email, "password": password, "csrf_token": data["csrf_token"], "user": data["user"]}


def csrf_headers(token: str) -> dict[str, str]:
    return {"X-CSRF-Token": token}


def upload_image(
    client: TestClient,
    case_id: str,
    view_type: str,
    account: dict,
    payload: bytes | None = None,
    filename: str = "photo.jpg",
) -> dict:
    payload = payload if payload is not None else make_image_bytes()
    response = client.post(
        f"/api/v1/cases/{case_id}/images",
        data={"view_type": view_type},
        files={"file": (filename, payload, "application/octet-stream")},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_case(client: TestClient, account: dict, research_consent: bool = False) -> dict:
    response = client.post(
        "/api/v1/cases",
        json={"claimed_herb": "骨碎补", "analysis_consent": True, "research_consent": research_consent},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 201, response.text
    return response.json()


def analyze_case(client: TestClient, case_id: str, account: dict) -> dict:
    response = client.post(
        f"/api/v1/cases/{case_id}/analyze",
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 200, response.text
    return response.json()
