# ============================================================
#  Edge AI - Flask 告警仪表盘
#  职责：订阅 MQTT 告警 → Web 页面实时展示
#  依赖：flask + waitress + paho-mqtt
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# ---- 复制依赖清单并安装 ----
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# ---- 复制 Web 服务脚本 ----
COPY dashboard_service.py .

# ---- 默认命令：启动仪表盘 ----
CMD ["python", "dashboard_service.py"]
