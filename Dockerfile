# Paper Factor System API Dockerfile（多阶段构建）
# 用法：docker build -t paper-factor-api . && docker run -p 8000:8000 paper-factor-api

# ===========================================
# Stage 1: 构建依赖
# ===========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# 系统依赖（lizard 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===========================================
# Stage 2: 运行时
# ===========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 仅安装运行时必需的系统包
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        && rm -rf /var/lib/apt/lists/*

# 复制 Python 依赖（从 builder）
COPY --from=builder /install /usr/local

# 复制应用代码
COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]