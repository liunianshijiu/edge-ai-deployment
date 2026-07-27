"""
============================================================
 Edge AI - 统一服务器 (API + Dashboard)
 
 功能:
   REST API:  /detect, /health, /history, /docs
   Web面板:  /dashboard  (MQTT实时告警可视化)
   
 启动方式:
   cd 项目根目录
   .\edgeai_env\Scripts\Activate.ps1
   pip install -r api/requirements-api.txt
   python api/edgeai_server.py
   
 访问地址:
   http://localhost:8080/docs       ← Swagger API 文档
   http://localhost:8080/dashboard ← Web 可视化面板
   http://localhost:8080/detect    ← 检测端点
============================================================
"""

import os
import io
import time
import json
import base64
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import cv2
import numpy as np


# ==================== 配置 ====================
class Config:
    HOST = os.getenv('API_HOST', '0.0.0.0')
    PORT = int(os.getenv('API_PORT', '8080'))
    MODEL_PATH = os.getenv('MODEL_PATH', 'yolov8n.pt')
    CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.5'))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))
    MAX_HISTORY = int(os.getenv('MAX_HISTORY', '100'))
    IMAGE_MAX_SIZE = int(os.getenv('IMAGE_MAX_SIZE', '1920'))
    
    # MQTT 配置（Dashboard 用）
    MQTT_BROKER = os.getenv('MQTT_BROKER', 'test.mosquitto.org')
    MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
    MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'edgeai/alerts')


# ==================== 数据模型 ====================
class DetectionResult(BaseModel):
    class_name: str = Field(..., description="类别名称")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1,y1,x2,y2]")


class HealthResponse(BaseModel):
    status: str
    model: str
    model_size_mb: float
    conf_threshold: float
    uptime_seconds: float
    total_detections: int
    history_count: int
    mqtt_alerts_count: int


# ==================== 全局状态 ====================
app = FastAPI(
    title="EdgeAI Unified Server",
    version="2.0.0",
    description="""
## 边缘 AI 统一服务 (API + Dashboard)

### API 端点
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/detect` | 上传图片，返回检测结果 |
| GET | `/health` | 健康检查 |
| GET | `/history` | 检测历史记录 |
| GET | `/dashboard` | **Web 可视化面板** (MQTT 实时告警) |

### 使用方式
1. 访问 `/docs` 打开 Swagger UI 测试 API
2. 访问 `/dashboard` 查看 MQTT 实时告警面板
3. 或用 curl：
```bash
curl -X POST "http://localhost:8080/detect" -F "file=@bus.jpg"
```
""",
)

model = None
start_time = time.time()
detection_history: List[dict] = []      # API 检测历史
mqtt_alerts: List[dict] = []             # MQTT 实时告警缓存
total_detections_count = 0
mqtt_connected = False                    # MQTT 连接状态标志
lock = threading.Lock()                   # 保护共享数据


# ==================== HTML 模板 (Dashboard 面板) ====================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeAI 告警面板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }

        /* 头部 */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px 30px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .header h1 { font-size: 22px; color: #fff; }
        .header h1 span { color: #4CAF50; }
        .status-bar {
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 13px;
            color: #aaa;
        }
        .status-dot {
            width: 10px; height: 10px; border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.online { background: #4CAF50; box-shadow: 0 0 10px #4CAF50; }
        .status-dot.offline { background: #f44336; box-shadow: 0 0 10px #f44336; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        /* 统计卡片 */
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 25px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 14px;
            padding: 22px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            transition: transform 0.2s, background 0.2s;
        }
        .stat-card:hover { transform: translateY(-3px); background: rgba(255,255,255,0.08); }
        .stat-number { font-size: 32px; font-weight: 700; }
        .stat-label { font-size: 12px; color: #888; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
        .c-total { color: #4CAF50; }
        .c-person { color: #f44336; }
        .c-car { color: #2196F3; }
        .c-alerts { color: #FF9800; }

        /* 表格 */
        .table-card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .table-header {
            padding: 18px 24px;
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        table { width: 100%; border-collapse: collapse; }
        th {
            text-align: left;
            padding: 12px 20px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            font-weight: 600;
            background: rgba(0,0,0,0.15);
        }
        td { padding: 14px 20px; font-size: 13px; border-top: 1px solid rgba(255,255,255,0.03); }
        tr:hover td { background: rgba(255,255,255,0.04); }
        
        /* 类别标签 */
        .tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .tag-person { background: rgba(244,67,54,0.15); color: #f44336; }
        .tag-car { background: rgba(33,150,243,0.15); color: #2196F3; }
        .tag-bus { background: rgba(76,175,80,0.15); color: #4CAF50; }
        .tag-dog, .tag-cat { background: rgba(233,30,99,0.15); color: #e91e63; }
        .tag-bicycle { background: rgba(56,142,60,0.15); color: #388E3C; }
        .tag-default { background: rgba(158,158,158,0.15); color: #9e9e9e; }

        .conf-bar {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .conf-track {
            width: 60px;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .conf-fill { height: 100%; border-radius: 2px; background: #4CAF50; }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #555;
        }
        .empty-state .icon { font-size: 48px; margin-bottom: 15px; }

        /* 顶部导航栏 */
        .top-nav {
            display: flex;
            gap: 4px;
            padding: 10px 0;
            margin-bottom: 16px;
        }
        .nav-link {
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            color: #888;
            font-size: 13px;
            font-weight: 500;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            transition: all 0.2s;
        }
        .nav-link:hover {
            color: #fff;
            background: rgba(76,175,80,0.15);
            border-color: rgba(76,175,80,0.3);
        }
        .nav-link.active {
            color: #4CAF50;
            background: rgba(76,175,80,0.1);
            border-color: rgba(76,175,80,0.3);
        }

        footer {
            text-align: center;
            margin-top: 30px;
            color: #444;
            font-size: 12px;
        }
        footer a { color: #4CAF50; text-decoration: none; }
    </style>
</head>
<body>
<div class="container">

    <!-- 顶部导航 -->
    <nav class="top-nav">
        <a href="/monitor" class="nav-link">🎥 实时监控</a>
        <a href="/dashboard" class="nav-link active">📊 告警面板</a>
        <a href="/docs" class="nav-link">📖 API 文档</a>
        <a href="/health" class="nav-link">❤️ 健康检查</a>
    </nav>

    <!-- 头部 -->
    <div class="header">
        <h1><span>⚡</span> EdgeAI 边缘告警面板</h1>
        <div class="status-bar">
            <span class="status-dot {{'online' if mqtt_status else 'offline'}}"></span>
            <span>{{'MQTT 已连接' if mqtt_status else 'MQTT 未连接'}} · {{ broker }}:{{ port }}</span>
            <span id="timer"></span>
        </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number c-total">{{ total_alerts }}</div>
            <div class="stat-label">总告警数</div>
        </div>
        <div class="stat-card">
            <div class="stat-number c-person">{{ person_count }}</div>
            <div class="stat-label">人员检测</div>
        </div>
        <div class="stat-card">
            <div class="stat-number c-car">{{ vehicle_count }}</div>
            <div class="stat-label">车辆检测</div>
        </div>
        <div class="stat-card">
            <div class="stat-number c-alerts">{{ alerts | length }}</div>
            <div class="stat-label">缓存条目</div>
        </div>
    </div>

    <!-- 告警表格 -->
    <div class="table-card">
        <div class="table-header">
            <span>📋 最近告警记录</span>
            <span style="font-size:12px;color:#666;font-weight:400;">
                自动刷新: 每 3 秒 | 显示最近 {{ alerts | length if alerts | length <= 20 else 20 }} 条
            </span>
        </div>
        {% if alerts %}
        <table>
            <thead><tr><th>#</th><th>时间</th><th>目标类型</th><th>置信度</th></tr></thead>
            <tbody>
                {% for a in reversed(alerts[-20:]) %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td style="color:#aaa;">{{ a.time }}</td>
                    <td>
                        {% set tag_class = 'tag-' + a.type if a.type in ['person','car','bus','dog','cat','bicycle'] else 'tag-default' %}
                        <span class="{{ tag_class }}">{{ a.type | upper }}</span>
                    </td>
                    <td>
                        <div class="conf-bar">
                            <div class="conf-track"><div class="conf-fill" style="width:{{ '%.0f' | format(a.conf * 100) }}%"></div></div>
                            <span>{{ '%.0f' | format(a.conf * 100) }}%</span>
                        </div>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="empty-state">
            <div class="icon">📡</div>
            <p>暂无告警数据</p>
            <p style="font-size:12px;margin-top:8px;">等待 YOLO 检测服务发送消息到 MQTT Broker...</p>
        </div>
        {% endif %}
    </div>

    <footer>
        Edge AI Deployment v2.0 · Unified Server (API + Dashboard) · 
        <a href="/docs">Swagger API</a> · 
        <a href="/health">Health Check</a>
    </footer>

</div>

<script>
    // 时钟更新
    function updateClock() {
        document.getElementById('timer').textContent = new Date().toLocaleTimeString('zh-CN');
    }
    updateClock();
    setInterval(updateClock, 1000);

    // 页面自动刷新
    setTimeout(function() { location.reload(); }, 3000);
</script>
</body>
</html>
"""


# ==================== 模型加载 ====================
def load_model():
    """启动时加载 YOLO 模型"""
    global model
    
    model_path = Config.MODEL_PATH
    if not Path(model_path).exists():
        print(f"[错误] 模型不存在: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"[模型] 加载成功: {model_path} ({size_mb:.2f}MB)")
        return True
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return False


# ==================== 工具函数 ====================
def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码上传的图片")
    return image


def resize_if_needed(image: np.ndarray, max_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


def run_detection(image: np.ndarray):
    start = time.perf_counter()
    results = model(image, conf=Config.CONF_THRESHOLD, iou=Config.IOU_THRESHOLD, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results[0], elapsed_ms


def parse_results(result, inference_ms: float):
    global total_detections_count
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detections.append(DetectionResult(
                class_name=result.names[cls_id],
                confidence=round(conf, 4),
                bbox=[round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
            ))
            total_detections_count += 1
    
    return {
        "success": True,
        "model": Config.MODEL_PATH,
        "inference_ms": round(inference_ms, 2),
        "image_size": [result.orig_shape[1], result.orig_shape[0]],
        "total_objects": len(detections),
        "detections": detections
    }


def save_history(image_size: List[int], parsed: dict):
    with lock:
        summary = {}
        for det in parsed["detections"]:
            name = det.class_name
            summary[name] = summary.get(name, 0) + 1
        
        record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "image_size": image_size,
            "total_objects": parsed["total_objects"],
            "detections_summary": [{"class_name": k, "count": v} for k, v in sorted(summary.items())]
        }
        detection_history.append(record)
        if len(detection_history) > Config.MAX_HISTORY:
            detection_history.pop(0)


# ==================== MQTT 后台订阅者 ====================
def mqtt_listener():
    """后台线程：连接 MQTT Broker 并持续监听告警"""
    global mqtt_connected
    
    try:
        import paho.mqtt.client as mqtt_client_lib
    except ImportError:
        print("[MQTT] paho-mqtt 未安装，跳过订阅")
        return
    
    def on_connect(client, userdata, flags, rc):
        global mqtt_connected
        if rc == 0:
            mqtt_connected = True
            client.subscribe(Config.MQTT_TOPIC)
            print(f"[MQTT] 已连接 {Config.MQTT_BROKER}:{Config.MQTT_PORT} | Topic: {Config.MQTT_TOPIC}")
        else:
            mqtt_connected = False
            print(f"[MQTT] 连接失败，错误码: {rc}")
    
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            with lock:
                mqtt_alerts.append({
                    "time": data.get("timestamp", "--:--:--"),
                    "type": data.get("type", "unknown"),
                    "conf": data.get("confidence", 0)
                })
                if len(mqtt_alerts) > Config.MAX_HISTORY:
                    mqtt_alerts.pop(0)
            print(f"[MQTT 收到] {data.get('type')} ({data.get('confidence',0):.0%})")
        except json.JSONDecodeError:
            pass
    
    client = mqtt_client_lib.Client(client_id='unified_server')
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    
    while True:
        try:
            client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            mqtt_connected = False
            print(f"[MQTT] 连接异常: {e}, 5秒后重试...")
            time.sleep(5)


# ==================== 启动事件 ====================
@app.on_event("startup")
async def startup():
    load_model()
    threading.Thread(target=mqtt_listener, daemon=True).start()


# ==================== API 端点 ====================

@app.get("/", tags=["首页"])
async def root():
    return {
        "service": "EdgeAI Unified Server",
        "version": "2.0.0",
        "endpoints": {
            "POST /detect": "图片检测",
            "GET  /health": "健康检查",
            "GET  /history": "检测历史",
            "GET  /monitor": "**实时摄像头监控** (浏览器端)",
            "GET  /dashboard": "Web 可视化面板 (MQTT 实时告警)",
            "GET  /docs": "Swagger 文档"
        },
        "monitor_url": f"http://localhost:{Config.PORT}/monitor",
        "dashboard_url": f"http://localhost:{Config.PORT}/dashboard",
        "swagger_url": f"http://localhost:{Config.PORT}/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    model_size = 0.0
    if Path(Config.MODEL_PATH).exists():
        model_size = Path(Config.MODEL_PATH).stat().st_size / (1024*1024)
    
    return HealthResponse(
        status="ok" if model else "error",
        model=Config.MODEL_PATH,
        model_size_mb=round(model_size, 2),
        conf_threshold=Config.CONF_THRESHOLD,
        uptime_seconds=round(time.time()-start_time, 1),
        total_detections=total_detections_count,
        history_count=len(detection_history),
        mqtt_alerts_count=len(mqtt_alerts),
    )


@app.post("/detect", tags=["检测"])
async def detect_objects(
    file: UploadFile = File(..., description="待检测图片"),
    draw_boxes: bool = Query(False, description="返回标注图 base64")
):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传有效图片")
    
    try:
        file_bytes = await file.read()
        image = decode_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    original_size = [image.shape[1], image.shape[0]]
    image_processed = resize_if_needed(image, Config.IMAGE_MAX_SIZE)
    result, inference_ms = run_detection(image_processed)
    parsed = parse_results(result, inference_ms)
    save_history(original_size, parsed)
    
    response_data = {
        "success": parsed["success"],
        "model": parsed["model"],
        "inference_ms": parsed["inference_ms"],
        "image_size": parsed["image_size"],
        "total_objects": parsed["total_objects"],
        "detections": [det.model_dump() for det in parsed["detections"]]
    }
    
    if draw_boxes and result.boxes is not None:
        try:
            annotated = result.plot()
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            response_data["annotated_image"] = base64.b64encode(buffer).decode()
        except Exception:
            pass
    
    return JSONResponse(content=response_data)


@app.get("/history", tags=["查询"])
async def get_history(limit: int = Query(20, ge=1, le=100), class_filter: Optional[str] = None):
    records = list(detection_history)
    if class_filter:
        records = [r for r in records
                   if any(s["class_name"] == class_filter for s in r.get("detections_summary", []))]
    return {"total": len(records), "records": records[-limit:]}


@app.get("/dashboard", response_class=HTMLResponse, tags=["面板"])
async def dashboard():
    """
    Web 可视化面板 — 显示 MQTT 实时告警数据
    
    特性:
    - 深色主题，毛玻璃效果
    - 统计卡片（总数/人员/车辆）
    - 3秒自动刷新获取最新告警
    - 置信度进度条可视化
    """
    from jinja2 import Template as JTemplate
    
    with lock:
        alerts_copy = list(mqtt_alerts)
    
    # 统计各类别数量
    person_count = sum(1 for a in alerts_copy if a.get("type") == "person")
    vehicle_types = ["car", "bus", "truck", "motorcycle", "bicycle"]
    vehicle_count = sum(1 for a in alerts_copy if a.get("type") in vehicle_types)
    
    html = JTemplate(DASHBOARD_HTML).render(
        alerts=alerts_copy,
        total_alerts=len(alerts_copy),
        person_count=person_count,
        vehicle_count=vehicle_count,
        mqtt_status=mqtt_connected,
        broker=Config.MQTT_BROKER,
        port=Config.MQTT_PORT,
    )
    return HTMLResponse(html)


# ==================== 实时监控页面 HTML ====================
MONITOR_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeAI 实时监控</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0e17;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
        }

        /* 头部 */
        .header {
            width: 100%;
            max-width: 900px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .header h1 { font-size: 18px; }
        .header h1 span { color: #4CAF50; }

        /* 状态指示 */
        .status-row {
            display: flex;
            gap: 18px;
            align-items: center;
            font-size: 13px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 8px; height: 8px; border-radius: 50%;
        }
        .dot.green { background: #4CAF50; box-shadow: 0 0 6px #4CAF50; animation: blink 1.5s infinite; }
        .dot.red { background: #f44336; box-shadow: 0 0 6px #f44336; }
        .dot.yellow { background: #FF9800; box-shadow: 0 0 6px #FF9800; animation: blink 1s infinite; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

        /* 视频容器 */
        .video-container {
            position: relative;
            width: 900px;
            max-width: 95vw;
            border-radius: 14px;
            overflow: hidden;
            background: #111;
            border: 2px solid rgba(76,175,80,0.3);
            box-shadow: 0 0 30px rgba(76,175,80,0.1);
        }

        video {
            display: block;
            width: 100%;
            height: auto;
            transform: scaleX(-1);  /* 镜像翻转，更自然 */
        }

        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        /* 底部控制栏 */
        .controls {
            width: 100%;
            max-width: 900px;
            margin-top: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.06);
        }

        .btn-group { display: flex; gap: 10px; }

        button {
            padding: 8px 22px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-start {
            background: linear-gradient(135deg, #4CAF50, #388E3C);
            color: white;
        }
        .btn-start:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(76,175,80,0.4); }
        .btn-stop {
            background: linear-gradient(135deg, #f44336, #c62828);
            color: white;
        }
        .btn-stop:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(244,67,54,0.4); }
        button:disabled { opacity: 0.4; cursor: not-allowed; transform: none !important; }

        /* 统计信息 */
        .stats-mini {
            display: flex;
            gap: 16px;
            font-size: 13px;
        }
        .stat-item {
            background: rgba(255,255,255,0.04);
            padding: 5px 12px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-weight: 700; font-size: 16px; color: #4CAF50; }
        .stat-label { font-size: 10px; color: #777; text-transform: uppercase; letter-spacing: 0.5px; }

        /* 检测结果侧边栏 */
        .detect-list {
            width: 100%;
            max-width: 900px;
            margin-top: 12px;
            padding: 12px 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.04);
        }
        .detect-list-title {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .detect-tags { display: flex; flex-wrap: wrap; gap: 8px; }
        .detect-tag {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .tag-person { background: rgba(244,67,54,0.2); color: #f44336; border: 1px solid rgba(244,67,54,0.3); }
        .tag-car { background: rgba(33,150,243,0.2); color: #2196F3; border: 1px solid rgba(33,150,243,0.3); }
        .tag-bus { background: rgba(76,175,80,0.2); color: #4CAF50; border: 1px solid rgba(76,175,80,0.3); }
        .tag-default { background: rgba(158,158,158,0.15); color: #aaa; border: 1px solid rgba(158,158,158,0.2); }

        /* 提示文字 */
        .hint {
            text-align: center;
            color: #555;
            font-size: 14px;
            margin-top: 40px;
        }
        .hint .icon { font-size: 48px; margin-bottom: 12px; }

        /* 顶部导航栏 */
        .top-nav {
            display: flex;
            gap: 4px;
            padding: 10px 0;
            margin-bottom: 12px;
        }
        .nav-link {
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            color: #888;
            font-size: 13px;
            font-weight: 500;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            transition: all 0.2s;
        }
        .nav-link:hover {
            color: #fff;
            background: rgba(76,175,80,0.15);
            border-color: rgba(76,175,80,0.3);
        }
        .nav-link.active {
            color: #4CAF50;
            background: rgba(76,175,80,0.1);
            border-color: rgba(76,175,80,0.3);
        }

        footer {
            margin-top: 20px;
            color: #333;
            font-size: 11px;
        }
        footer a { color: #4CAF50; text-decoration: none; }
    </style>
</head>
<body>

    <!-- 顶部导航 -->
    <nav class="top-nav">
        <a href="/monitor" class="nav-link active">🎥 实时监控</a>
        <a href="/dashboard" class="nav-link">📊 告警面板</a>
        <a href="/docs" class="nav-link">📖 API 文档</a>
        <a href="/health" class="nav-link">❤️ 健康检查</a>
    </nav>

    <!-- 头部 -->
    <div class="header">
        <h1><span>⚡</span> EdgeAI 实时目标检测</h1>
        <div class="status-row">
            <div class="status-item">
                <span class="dot {{'green' if is_detecting else 'red'}}" id="statusDot"></span>
                <span id="statusText">{{'检测中...' if is_detecting else '待机'}}</span>
            </div>
            <div class="status-item" id="fpsDisplay" style="color:#888;">
                FPS: --
            </div>
            <div class="status-item" id="latencyDisplay" style="color:#888;">
                延迟: -- ms
            </div>
        </div>
    </div>

    <!-- 视频区域 -->
    <div class="video-container" id="videoContainer">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
    </div>

    <!-- 控制栏 -->
    <div class="controls">
        <div class="btn-group">
            <button id="btnStart" class="btn-start" onclick="startCamera()">📷 开启摄像头</button>
            <button id="btnDetect" class="btn-start" onclick="toggleDetect()" disabled>🎯 开始检测</button>
            <button id="btnStop" class="btn-stop" onclick="stopAll()" disabled>⏹ 停止</button>
        </div>
        <div class="stats-mini">
            <div class="stat-item">
                <div class="stat-value" id="objectCount">0</div>
                <div class="stat-label">当前目标</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="totalDetections">0</div>
                <div class="stat-label">累计检测</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="intervalValue">{{ interval }}</div>
                <div class="stat-label">间隔(ms)</div>
            </div>
        </div>
    </div>

    <!-- 当前检测结果 -->
    <div class="detect-list" id="detectList" style="display:none;">
        <div class="detect-list-title">当前帧检测结果</div>
        <div class="detect-tags" id="detectTags"></div>
    </div>

    <footer>
        EdgeAI Real-Time Monitor · YOLOv8n INT8 · 
        <a href="/docs">API 文档</a> · 
        <a href="/dashboard">告警面板</a>
    </footer>

<script>
// ==================== 全局状态 ====================
const API_URL = '';
const DETECT_INTERVAL = {{ interval }} || 500;   // 检测频率（毫秒）
let video, overlay, ctx;
let stream = null;
let detecting = false;
let detectTimer = null;
let totalDetections = 0;
let frameCount = 0;
let lastFpsTime = Date.now();
let lastLatency = 0;

// 颜色映射（不同类别不同颜色）
const COLOR_MAP = {
    person: { stroke: '#f44336', fill: 'rgba(244,67,54,0.15)' },
    bicycle: { stroke: '#388E3C', fill: 'rgba(56,142,60,0.15)' },
    car: { stroke: '#2196F3', fill: 'rgba(33,150,243,0.15)' },
    bus: { stroke: '#4CAF50', fill: 'rgba(76,175,80,0.15)' },
    truck: { stroke: '#FF9800', fill: 'rgba(255,152,0,0.15)' },
    motorcycle: { stroke: '#9C27B0', fill: 'rgba(156,39,176,0.15)' },
    dog: { stroke: '#e91e63', fill: 'rgba(233,30,99,0.15)' },
    cat: { stroke: '#00BCD4', fill: 'rgba(0,188,212,0.15)' },
};
const DEFAULT_COLOR = { stroke: '#fff', fill: 'rgba(255,255,255,0.1)' };

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    overlay = document.getElementById('overlay');
    ctx = overlay.getContext('2d');

    // Canvas 尺寸跟随视频
    video.addEventListener('loadedmetadata', () => {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
    });
});

// ==================== 摄像头控制 ====================
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
            audio: false
        });

        // 先播放视频，等元数据加载
        video.srcObject = stream;
        await video.play();

        // 设置 canvas 尺寸（需要等视频开始播放后）
        setTimeout(() => {
            overlay.width = video.videoWidth || 1280;
            overlay.height = video.videoHeight || 720;
        }, 300);

        document.getElementById('btnStart').disabled = true;
        document.getElementById('btnDetect').disabled = false;
        document.getElementById('btnStop').disabled = false;

    } catch (err) {
        alert('无法访问摄像头：\\n' + err.message + '\\n\\n请确保：\\n1. 浏览器允许了摄像头权限\\n2. 使用 HTTPS 或 localhost');
        console.error('Camera error:', err);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;
}

// ==================== 检测控制 ====================
async function toggleDetect() {
    const btn = document.getElementById('btnDetect');
    
    if (!detecting) {
        // 开始检测
        detecting = true;
        btn.textContent = '⏸ 暂停检测';
        btn.classList.remove('btn-start');
        btn.classList.add('btn-stop');
        updateStatus(true);
        
        detectTimer = setInterval(runDetection, DETECT_INTERVAL);
        runDetection(); // 立即执行第一次
        
    } else {
        // 暂停检测
        detecting = false;
        btn.textContent = '🎯 开始检测';
        btn.classList.remove('btn-stop');
        btn.classList.add('btn-start');
        updateStatus(false);
        
        if (detectTimer) clearInterval(detectTimer);
        clearCanvas();
    }
}

async function runDetection() {
    if (!video.videoWidth) return;

    const t0 = performance.now();

    try {
        // 截图 → base64
        const blob = await captureFrame();
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        // 发送到 /detect 接口
        const resp = await fetch(API_URL + '/detect', {
            method: 'POST',
            body: formData
        });

        const data = await resp.json();
        const latency = performance.now() - t0;
        lastLatency = Math.round(latency);

        // 更新 FPS
        frameCount++;
        updateFPS();

        // 更新统计
        if (data.success && data.detections) {
            totalDetections += data.detections.length;
            
            // 画检测框
            drawDetections(data.detections, data.image_size);
            
            // 更新 UI
            updateDetectUI(data.detections, data.inference_ms);
        } else {
            document.getElementById('objectCount').textContent = '0';
        }

        document.getElementById('latencyDisplay').textContent = `延迟: ${lastLatency}ms`;

    } catch (err) {
        console.error('Detect error:', err);
        document.getElementById('latencyDisplay').textContent = `错误`;
    }
}

// ==================== 截图 ====================
function captureFrame() {
    return new Promise((resolve) => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        tempCanvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.7);
    });
}

// ==================== 绘制检测框 ====================
function drawDetections(detections, imageSize) {
    // 清空画布
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    if (!detections || detections.length === 0) return;

    const scaleX = overlay.width / (imageSize ? imageSize[0] : video.videoWidth);
    const scaleY = overlay.height / (imageSize ? imageSize[1] : video.videoHeight);

    detections.forEach((det, i) => {
        const [x1, y1, x2, y2] = det.bbox;
        const sx = x1 * scaleX;
        const sy = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        // 获取颜色
        const colors = COLOR_MAP[det.class_name] || DEFAULT_COLOR;

        // 画填充矩形（半透明）
        ctx.fillStyle = colors.fill;
        ctx.fillRect(sx, sy, sw, sh);

        // 画边框
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx, sy, sw, sh);

        // 标签背景 + 文字
        const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
        ctx.font = 'bold 13px -apple-system, sans-serif';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = colors.stroke;
        ctx.fillRect(sx, sy - 20, textWidth + 10, 20);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, sx + 5, sy - 5);
    });
}

function clearCanvas() {
    if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    document.getElementById('objectCount').textContent = '0';
    document.getElementById('detectList').style.display = 'none';
}

// ==================== UI 更新 ====================
function updateDetectUI(detections, inferenceMs) {
    document.getElementById('objectCount').textContent = detections.length;
    document.getElementById('totalDetections').textContent = totalDetections;

    // 显示标签列表
    const listEl = document.getElementById('detectList');
    const tagsEl = document.getElementById('detectTags');
    
    if (detections.length > 0) {
        listEl.style.display = 'block';
        tagsEl.innerHTML = detections.map(det => {
            const cls = det.class_name in COLOR_MAP ? 'tag-' + det.class_name : 'tag-default';
            return `<span class="detect-tag ${cls}">${det.class_name.toUpperCase()} ${Math.round(det.confidence*100)}%</span>`;
        }).join('');
    } else {
        tagsEl.innerHTML = '<span style="color:#555;font-size:12px;">未检测到目标</span>';
    }
}

function updateStatus(isOn) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    dot.className = 'dot ' + (isOn ? 'green' : 'red');
    text.textContent = isOn ? '检测中...' : '已暂停';
}

function updateFPS() {
    const now = Date.now();
    const elapsed = now - lastFpsTime;
    if (elapsed >= 1000) {
        const fps = Math.round(frameCount * 1000 / elapsed);
        document.getElementById('fpsDisplay').textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastFpsTime = now;
    }
}

function stopAll() {
    stopCamera();
    if (detecting) toggleDetect();
    
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnDetect').disabled = true;
    document.getElementById('btnStop').disabled = true;
}
</script>

</body>
</html>
"""


@app.get("/monitor", response_class=HTMLResponse, tags=["监控"])
async def monitor_page():
    """
    浏览器端实时摄像头监控页面
    
    功能:
    - 调用本地摄像头 (getUserMedia)
    - 定时截图发送到 /detect 接口进行 YOLO 检测
    - 在视频画面上叠加彩色检测框
    - 实时统计：FPS、延迟、累计检测数
    
    注意: 需要 HTTPS 或 localhost 才能调用摄像头
    """
    from jinja2 import Template as JTemplate

    html = JTemplate(MONITOR_HTML).render(
        is_detecting=False,
        interval=500  # 默认每 500ms 检测一次（约 2 FPS）
    )
    return HTMLResponse(html)


@app.get("/api/alerts", tags=["查询"])
async def get_mqtt_alerts():
    """返回当前缓存的 MQTT 告警数据（给前端 JS 轮询用）"""
    with lock:
        return {"count": len(mqtt_alerts), "alerts": list(mqtt_alerts)}


# ==================== 入口 ====================
if __name__ == '__main__':
    if not load_model():
        print("[警告] 模型加载失败\n")
    
    print(f"\n{'='*50}")
    print(f"  EdgeAI Unified Server v2.0")
    print(f"  Monitor:  http://{Config.HOST}:{Config.PORT}/monitor")
    print(f"  Panel:    http://{Config.HOST}:{Config.PORT}/dashboard")
    print(f"  API Docs: http://{Config.HOST}:{Config.PORT}/docs")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
