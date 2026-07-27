# ============================================================
#  Edge AI - YOLO 目标检测服务
#  职责：从摄像头读取画面 → YOLOv8n 检测 → MQTT 发布告警
#  依赖：tflite_runtime + opencv-python-headless + paho-mqtt
#  镜像优势：不需要 PyTorch(~2GB)，总大小约 400MB
# ============================================================

FROM python:3.10-slim

# ---- 设置工作目录 ----
WORKDIR /app

# ---- 安装系统依赖（OpenCV 需要的 libgl） ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ---- 复制依赖清单并安装 Python 包 ----
COPY requirements-yolo.txt .
RUN pip install --no-cache-dir -r requirements-yolo.txt

# ---- 复制模型文件和检测脚本 ----
COPY yolov8n_int8.tflite ./models/
COPY yolo_service.py .

# ---- 默认命令：启动检测服务 ----
CMD ["python", "yolo_service.py"]
