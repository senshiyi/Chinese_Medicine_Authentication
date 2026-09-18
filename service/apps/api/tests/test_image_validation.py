import io
import re

import pytest
from PIL import Image

from drynaria_api.image_validation import (
    UploadValidationError,
    validate_upload_image,
)
from helpers import make_image_bytes


def test_empty_bytes_are_rejected():
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_image(b"")
    assert exc.value.code == "empty_file"


def test_corrupted_bytes_are_rejected():
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_image(b"this is definitely not an image \x00\x01")
    assert exc.value.code == "corrupted_image"


def test_payload_over_20mb_is_rejected():
    big = b"\x00" * (20 * 1024 * 1024 + 1)
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_image(big)
    assert exc.value.code == "file_too_large"


def test_pixels_over_40_million_are_rejected():
    # 7000 x 6000 = 4200 万像素，超过上限且仍可被 Pillow 打开（纯色 PNG 编码为 C 级快速路径）
    payload = make_image_bytes(size=(7000, 6000), image_format="PNG", texture=False)
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_image(payload)
    assert exc.value.code == "too_many_pixels"


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_valid_textured_image_passes(image_format):
    validated = validate_upload_image(make_image_bytes(image_format=image_format))
    assert validated.format == image_format
    assert validated.width == 320 and validated.height == 240
    assert re.fullmatch(r"[0-9a-f]{64}", validated.sha256)
    assert validated.quality_passes is True
    assert validated.quality_flags == []
    # sanitized 可再次被真实解码
    with Image.open(io.BytesIO(validated.sanitized_bytes)) as opened:
        opened.load()
        assert opened.size == (320, 240)


def test_gif_is_rejected_even_if_mime_claims_jpeg():
    payload = make_image_bytes(image_format="GIF")
    with pytest.raises(UploadValidationError) as exc:
        validate_upload_image(payload)
    assert exc.value.code == "unsupported_format"


def test_exif_is_stripped_before_save():
    payload = make_image_bytes()
    exif_bytes = Image.Exif()
    exif_bytes[0x010F] = "Camera Maker Test"
    raw = io.BytesIO()
    Image.open(io.BytesIO(payload)).save(raw, format="JPEG", exif=exif_bytes)
    validated = validate_upload_image(raw.getvalue())
    with Image.open(io.BytesIO(validated.sanitized_bytes)) as reopened:
        exif = reopened.getexif()
        assert not exif or exif.get(0x010F) is None


def test_flat_image_flagged_blur():
    payload = make_image_bytes(texture=False, color=(120, 120, 120))
    validated = validate_upload_image(payload)
    assert "blur" in validated.quality_flags
    assert validated.quality_passes is False


def test_dark_image_flagged_underexposed():
    payload = make_image_bytes(texture=False, color=(2, 2, 2))
    validated = validate_upload_image(payload)
    assert "underexposed" in validated.quality_flags


def test_bright_image_flagged_overexposed():
    payload = make_image_bytes(texture=False, color=(255, 255, 255))
    validated = validate_upload_image(payload)
    assert "overexposed" in validated.quality_flags
