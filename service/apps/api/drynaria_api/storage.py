"""对象存储抽象：MinIO/S3 与本地磁盘两种后端。

对象键使用随机 UUID（不含用户标识或原文件名）。图片访问通过鉴权后的
API 代理端点读取；S3 后端额外提供短期签名 URL 供未来直连使用。
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from .settings import Settings


class StorageError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class StorageBackend(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def delete_bytes(self, key: str) -> None: ...

    @abstractmethod
    def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str | None: ...

    def exists(self, key: str) -> bool:
        try:
            self.get_bytes(key)
            return True
        except StorageError:
            return False


class LocalStorageBackend(StorageBackend):
    """本地磁盘后端：测试与无 MinIO 的本地开发使用。"""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = Path(key).name
        return self.base_dir / safe

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(f"object not found: {key}")
        return path.read_bytes()

    def delete_bytes(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str | None:
        return None  # 由 API 鉴权代理端点提供


class S3StorageBackend(StorageBackend):
    """S3/MinIO 后端（boto3）。"""

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        import boto3
        from botocore.client import Config

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            # 开发环境自动建桶；生产由运维预创建并开启版本控制
            self.client.create_bucket(Bucket=bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to read object {key}") from exc

    def delete_bytes(self, key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to delete object {key}") from exc

    def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str | None:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )


def build_storage(settings: Settings) -> StorageBackend:
    backend = settings.effective_storage_backend
    if backend == "s3":
        return S3StorageBackend(
            endpoint_url=settings.s3_endpoint_url,
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
    return LocalStorageBackend(settings.storage_local_dir)


def new_object_key() -> str:
    """随机 UUID 对象键；不包含用户标识或原文件名。"""
    return f"images/{uuid.uuid4()}.img"


def content_type_for(image_format: str) -> str:
    mapping = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    return mapping.get(image_format.upper(), "application/octet-stream")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
