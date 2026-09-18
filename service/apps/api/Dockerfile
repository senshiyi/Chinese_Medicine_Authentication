# 阶段 A API / Worker 镜像（单 Dockerfile，构建参数控制是否安装 torch）
# - api 服务：不安装 torch、不挂载模型 —— 进程不加载模型。
# - worker 服务：INSTALL_TORCH=1（CPU 版 torch），模型与原项目只读挂载。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv/api/apps/api \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api ./apps/api

ARG INSTALL_TORCH=0
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

EXPOSE 8000
CMD ["uvicorn", "drynaria_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
