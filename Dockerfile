# 使用官方 Python 运行时作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖（OpenCV 和字体）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    fonts-liberation \
    curl \
    && rm -rf /var/lib/apt/lists/*


# 复制依赖文件
COPY backend/requirements.txt /app/backend/

# 安装 Python 依赖
# 使用国内镜像加速（可选，如果在国外可以删除 -i 参数）
RUN pip install --no-cache-dir -r /app/backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY backend /app/backend
COPY frontend /app/frontend

# 创建必要的目录
RUN mkdir -p /app/backend/uploads \
    /app/backend/outputs \
    /app/backend/temp_processing \
    /app/logs


    
# 暴露端口
EXPOSE 5001

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 切换到 backend 目录
WORKDIR /app/backend

# 使用 gunicorn 运行应用（使用 Docker 专用配置）
CMD ["gunicorn", "-c", "gunicorn_docker.conf.py", "api:app"]

