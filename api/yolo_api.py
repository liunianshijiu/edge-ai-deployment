"""
============================================================
 Edge AI - YOLO REST API 服务
 功能：将目标检测能力暴露为 HTTP 接口

 接口列表:
   POST /detect      上传图片 -> 返回检测结果（JSON）
   GET  /health       健康检查 + 模型信息
   GET  /history      最近 N 条检测记录
   GET  /docs         Swagger 交互式文档（FastAPI 自动生成）

 启动方式:
   cd 项目根目录
   .\edgeai_env\Scripts\Activate.ps1
   pip install -r api/requirements-api.txt
   python api/yolo_api.py

 访问地址:
   http://localhost:8080/docs    <- Swagger 文档界面（可直接测试）
   http://localhost:8080/detect  <- 接口端点
============================================================
"""

import os
import io
import time
import json
import base64
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import cv2
import numpy as np
from PIL import Image

# ==================== 配置 ====================
class Config:
    """全局配置：环境变量 + 默认值"""
    HOST: str = os.getenv('API_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('API_PORT', '8080'))
    MODEL_PATH: str = os.getenv('MODEL_PATH', 'yolov8n.pt')
    CONF_THRESHOLD: float = float(os.getenv('CONF_THRESHOLD', '0.5'))  # 置信度阈值
    IOU_THRESHOLD: float = float(os.getenv('IOU_THRESHOLD', '0.45'))   # NMS IOU 阈值
    MAX_HISTORY: int = int(os.getenv('MAX_HISTORY', '100'))            # 最大历史记录数
    IMAGE_MAX_SIZE: int = int(os.getenv('IMAGE_MAX_SIZE', '1920'))     # 上传图片最大边长


# ==================== 数据模型（Pydantic 自动校验） ====================
class DetectionResult(BaseModel):
    """单个检测目标的结果"""
    class_name: str = Field(..., description="类别名称，如 person/car/dog")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0~1")
    bbox: List[float] = Field(..., description="边界框 [x1, y1, x2, y2] 像素坐标")


class DetectResponse(BaseModel):
    """POST /detect 的响应体"""
    success: bool
    model: str
    inference_ms: float = Field(..., description="推理耗时（毫秒）")
    image_size: List[int] = Field(..., description="输入图片尺寸 [宽, 高]")
    total_objects: int = Field(..., description="检测到的目标总数")
    detections: List[DetectionResult]


class HealthResponse(BaseModel):
    """GET /health 的响应体"""
    status: str
    model: str
    model_size_mb: float
    conf_threshold: float
    uptime_seconds: float
    total_detections: int
    history_count: int


class HistoryRecord(BaseModel):
    """历史记录单条"""
    timestamp: str
    image_size: List[int]
    total_objects: int
    detections_summary: List[Dict[str, Any]]  # [{class_name, count}]


# ==================== 全局状态 ====================
app = FastAPI(
    title="EdgeAI YOLO Detection API",
    version="1.0.0",
    description="""
## 边缘 AI 目标检测 REST API

基于 **YOLOv8n** 的轻量级目标检测服务。

### 特性
- 🚀 INT8 量化模型支持（3.27MB）
- ⚡ 单张推理 < 10ms (CPU)
- 📊 自动记录检测历史
- 📝 Swagger 交互式文档

### 使用方式
1. 访问 `/docs` 打开 Swagger UI
2. 点击 `POST /detect` → `Try it out` → `Choose file` → `Execute`
3. 或用 curl：
```bash
curl -X POST "http://localhost:8080/detect" -F "file=@test.jpg"
```
""",
)

# 检测模型实例（启动时加载）
model = None
start_time = time.time()
detection_history: List[dict] = []  # 线程安全（GIL 保护）
total_detections_count = 0


# ==================== 模型加载 ====================
def load_model():
    """
    加载 YOLO 模型（应用启动时调用一次）
    
    为什么在 startup 而不是 import 时加载？
    - 避免模块导入失败时整个服务起不来
    - 方便热替换模型（重启即可）
    - 启动日志能明确显示加载结果
    """
    global model
    
    model_path = Config.MODEL_PATH
    
    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print(f"[提示] 请确认路径正确或先运行 day9_yolo_export.py 导出模型")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # 显示模型信息
        info = {
            'task': model.task,
            'names': model.names,
            'nc': len(model.names),
        }
        
        # 计算模型文件大小（MB）
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        
        print(f"{'='*50}")
        print(f"[模型] 加载成功: {model_path}")
        print(f"[模型] 类型: {info['task']} | 类别数: {info['nc']}")
        print(f"[模型] 大小: {size_mb:.2f} MB")
        print(f"[模型] 类别: {list(info['names'].values())[:10]}...")
        print(f"{'='*50}")
        return True
        
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return False


# ==================== 工具函数 ====================
def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    将上传的字节数据解码为 OpenCV 图像数组
    支持: JPG/PNG/BMP/WebP 等常见格式
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码上传的图片")
    return image


def resize_if_needed(image: np.ndarray, max_size: int) -> np.ndarray:
    """
    如果图片过大，按比例缩小以加速推理
    保持宽高比，长边缩放到 max_size
    """
    h, w = image.shape[:2]
    if max(w, h) <= max_size:
        return image
    
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def run_detection(image: np.ndarray) -> tuple:
    """
    执行 YOLO 推理
    返回: (原始结果对象, 推理耗时ms)
    """
    start = time.perf_counter()
    results = model(
        image,
        conf=Config.CONF_THRESHOLD,
        iou=Config.IOU_THRESHOLD,
        verbose=False  # 不打印推理日志到终端
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return results[0], elapsed_ms


def parse_results(result, inference_ms: float) -> dict:
    """
    将 YOLO 原始输出解析为标准 JSON 结构
    返回格式与 DetectResponse 一致
    """
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
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            ))
            
            total_detections_count += 1
    
    return {
        "success": True,
        "model": Config.MODEL_PATH,
        "inference_ms": round(inference_ms, 2),
        "image_size": [result.orig_shape[1], result.orig_shape[0]],  # [W, H]
        "total_objects": len(detections),
        "detections": detections
    }


def save_history(image_size: List[int], parsed: dict):
    """
    保存检测记录到历史缓存
    只存摘要信息（不含完整边界框），节省内存
    """
    # 统计各类别数量
    summary = {}
    for det in parsed["detections"]:
        name = det.class_name  # DetectionResult 是 Pydantic 对象，用 . 访问
        summary[name] = summary.get(name, 0) + 1
    
    record = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "image_size": image_size,
        "total_objects": parsed["total_objects"],
        "detections_summary": [
            {"class_name": k, "count": v} for k, v in sorted(summary.items())
        ]
    }
    
    detection_history.append(record)
    
    # 防止内存无限增长
    if len(detection_history) > Config.MAX_HISTORY:
        detection_history.pop(0)


# ==================== 启动事件 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动时自动加载模型"""
    load_model()


# ==================== API 端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查端点
    
    用途:
    - Docker/K8s 就绪探针 (readiness probe)
    - 监控系统心跳检测
    - 快速查看服务状态和模型信息
    
    返回: 服务状态、模型信息、运行统计
    """
    model_size = 0.0
    if Path(Config.MODEL_PATH).exists():
        model_size = Path(Config.MODEL_PATH).stat().st_size / (1024 * 1024)
    
    return HealthResponse(
        status="ok" if model else "error",
        model=Config.MODEL_PATH,
        model_size_mb=round(model_size, 2),
        conf_threshold=Config.CONF_THRESHOLD,
        uptime_seconds=round(time.time() - start_time, 1),
        total_detections=total_detections_count,
        history_count=len(detection_history),
    )


@app.post("/detect", response_model=DetectResponse, tags=["检测"])
async def detect_objects(
    file: UploadFile = File(..., description="待检测的图片文件"),
    draw_boxes: bool = Query(False, description="是否返回带标注框的 base64 图片")
):
    """
    目标检测接口
    
    上传一张图片，返回所有检测到的目标及其位置。
    
    - **file**: 图片文件（支持 JPG/PNG/BMP/WebP）
    - **draw_boxes**: 可选，设为 true 时额外返回标注后的 base64 图
    
    返回示例:
    ```json
    {
      "success": true,
      "inference_ms": 15.32,
      "total_objects": 3,
      "detections": [
        {"class_name": "person", "confidence": 0.92, "bbox": [100, 50, 300, 500]},
        {"class_name": "car", "confidence": 0.87, "bbox": [400, 200, 800, 600]}
      ]
    }
    ```
    """
    global model
    
    # ---- 校验 ----
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请检查模型文件")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件")
    
    # ---- 读取并解码图片 ----
    try:
        file_bytes = await file.read()
        image = decode_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图片处理异常: {e}")
    
    # ---- 尺寸限制 ----
    original_size = [image.shape[1], image.shape[0]]  # [W, H]
    image_processed = resize_if_needed(image, Config.IMAGE_MAX_SIZE)
    
    # ---- 执行推理 ----
    result, inference_ms = run_detection(image_processed)
    
    # ---- 解析结果 ----
    parsed = parse_results(result, inference_ms)
    
    # ---- 保存历史 ----
    save_history(original_size, parsed)

    # ---- 构建可序列化的响应（Pydantic 对象转字典） ----
    response_data = {
        "success": parsed["success"],
        "model": parsed["model"],
        "inference_ms": parsed["inference_ms"],
        "image_size": parsed["image_size"],
        "total_objects": parsed["total_objects"],
        "detections": [det.model_dump() for det in parsed["detections"]]
    }

    # ---- 可选: 返回标注图 ----
    if draw_boxes and result.boxes is not None:
        try:
            annotated = result.plot()
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            response_data["annotated_image"] = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            # 标注图生成失败不影响主结果
            response_data["_draw_error"] = str(e)

    return JSONResponse(content=response_data)


@app.get("/history", tags=["查询"])
async def get_history(
    limit: int = Query(20, ge=1, le=100, description="返回条数上限"),
    class_filter: Optional[str] = Query(None, description="按类别过滤，如 person/car")
):
    """
    查询检测历史记录
    
    返回最近的检测摘要（不含完整边界框数据）
    支持按目标类型过滤
    """
    records = list(detection_history)
    
    # 按类别过滤
    if class_filter:
        records = [
            r for r in records
            if any(s["class_name"] == class_filter for s in r["detections_summary"])
        ]
    
    return {
        "total": len(records),
        "records": records[-limit:]  # 最近 N 条
    }


@app.get("/", tags=["首页"])
async def root():
    """
    API 根路径 — 返回快速使用指南
    """
    return {
        "service": "EdgeAI YOLO Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect": "上传图片进行目标检测",
            "GET  /health": "服务健康检查",
            "GET  /history": "查询检测历史",
            "GET  /docs": "Swagger 交互式文档"
        },
        "swagger_ui": "http://localhost:" + str(Config.PORT) + "/docs",
        "_tip": "访问 /docs 可以直接在线测试接口！"
    }


# ==================== 入口 ====================
if __name__ == '__main__':
    # 先尝试加载模型
    if not load_model():
        print("\n[警告] 模型加载失败，部分功能不可用\n")
    
    print(f"\n[API] 启动中... http://{Config.HOST}:{Config.PORT}")
    print(f"[API] Swagger 文档: http://{Config.HOST}:{Config.PORT}/docs\n")
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
