"""上传图片服务端校验：真实解码、尺寸/像素/字节上限、EXIF 清理、质量标记。

策略：只相信 Pillow 实际解码结果；损坏/超限输入抛 UploadValidationError，
绝不创建预测记录。可测的模糊/曝光问题写进 quality_flags（质量策略）。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from statistics import fmean

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MB 压缩后字节上限
DEFAULT_MAX_PIXELS = 40_000_000  # 解码像素上限
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# 可解释质量规则（阶段 A 固定阈值，验证数据确定后在策略版本中登记）
BLUR_MIN_VARIANCE = 15.0
EXPOSURE_MIN = 30.0
EXPOSURE_MAX = 245.0

_DEBOMB = Image.DecompressionBombError


class UploadValidationError(ValueError):
    """业务输入校验错误：code 为稳定工程码，message 为面向调用方的中文提示。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)

    @property
    def message(self) -> str:  # type: ignore[override]
        return str(self)


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    sha256: str
    width: int
    height: int
    format: str
    size_bytes: int
    sanitized_bytes: bytes
    quality: dict
    quality_flags: list[str] = field(default_factory=list)

    @property
    def quality_passes(self) -> bool:
        return not self.quality_flags


def validate_upload_image(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> ValidatedImage:
    """解码并清洗上传图片；不满足策略即抛 UploadValidationError。"""
    if not payload:
        raise UploadValidationError("empty_file", "上传文件为空。")
    if len(payload) > max_bytes:
        raise UploadValidationError(
            "file_too_large",
            f"图片大小 {len(payload)} 字节超过上限 {max_bytes} 字节。",
        )

    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image_format = (opened.format or "").upper()
            width, height = opened.size
            pixels = width * height
            if width <= 0 or height <= 0:
                raise UploadValidationError("invalid_image", "图片尺寸无效。")
            if pixels > max_pixels:
                raise UploadValidationError(
                    "too_many_pixels",
                    f"图片像素数 {pixels} 超过上限 {max_pixels}。",
                )
            if image_format not in ALLOWED_FORMATS:
                raise UploadValidationError(
                    "unsupported_format",
                    "仅支持 JPEG、PNG、WebP（以实际解码格式为准）。",
                )
            opened.load()
            rgb = opened.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise UploadValidationError("decompression_bomb", "图片解压体积异常。") from exc
    except UploadValidationError:
        # 校验失败已在 try 内抛出的业务错误，不要落入“损坏图片”分类
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadValidationError("corrupted_image", "图片损坏或无法解码。") from exc

    # 质量测量（灰度拉普拉斯方差 = 模糊；平均亮度 = 曝光）
    gray = rgb.convert("L")
    edges = gray.filter(ImageFilter.Kernel((3, 3), [0, -1, 0, -1, 4, -1, 0, -1, 0], scale=1))
    # 3x3 卷积在图像边缘会引入假边缘（缺失邻居按 0 计算），裁掉边界再统计方差
    inner = edges.crop((2, 2, edges.width - 2, edges.height - 2)) if edges.width > 4 and edges.height > 4 else edges
    try:
        blur_score = float(ImageStat.Stat(inner).var[0])
    except (ValueError, ZeroDivisionError):
        blur_score = float("inf")
    luminance = float(fmean(ImageStat.Stat(gray).mean or [128.0]))

    flags: list[str] = []
    if blur_score < BLUR_MIN_VARIANCE:
        flags.append("blur")
    if luminance < EXPOSURE_MIN:
        flags.append("underexposed")
    if luminance > EXPOSURE_MAX:
        flags.append("overexposed")

    # 保存前清除非必要 EXIF：重新编码为无元数据内容
    sanitized = io.BytesIO()
    save_format = image_format if image_format != "JPEG" else "JPEG"
    if save_format == "JPEG":
        rgb.save(sanitized, format="JPEG", quality=90)
    elif save_format == "WEBP":
        rgb.save(sanitized, format="WEBP", quality=90)
    else:
        rgb.save(sanitized, format="PNG")
    sanitized_bytes = sanitized.getvalue()

    quality = {
        "blur_score": round(blur_score, 3),
        "luminance": round(luminance, 3),
        "width": width,
        "height": height,
    }
    return ValidatedImage(
        sha256=hashlib.sha256(sanitized_bytes).hexdigest(),
        width=width,
        height=height,
        format=save_format,
        size_bytes=len(sanitized_bytes),
        sanitized_bytes=sanitized_bytes,
        quality=quality,
        quality_flags=flags,
    )
