markdown
# Edge AI 模型部署实验项目

从零学习 AI 模型边缘部署：模型转换、量化优化、实时检测、IoT 通信、监控面板。

## 系统架构

摄像头 → YOLOv8 检测 → MQTT 告警 → Flask 面板 ↓ TFLite 量化模型（13MB → 3.8MB）


## 核心技术栈

| 技术 | 用途 |
|------|------|
| TensorFlow Lite | 模型转换 + INT8 量化 |
| ONNX Runtime | 跨框架部署 + 性能对比 |
| YOLOv8 (Ultralytics) | 实时目标检测 |
| MQTT (paho-mqtt) | 设备间告警通信 |
| Flask | Web 监控面板 |

## 关键实验数据

### 量化对比（MobileNetV2）

| 格式 | 大小 | 推理延迟 |
|------|------|------|
| FP32 原始 | 13.35 MB | 5.75 ms |
| TFLite INT8 | 3.84 MB (-71%) | 3.39 ms (-41%) |

### 双框架对比（同模型 MobileNetV2）

| 格式 | 大小 | 延迟 |
|------|------|------|
| TFLite FP32 | 13.35 MB | 5.75 ms |
| ONNX FP32 | 13.38 MB | 2.13 ms |

结论：ONNX 在 FP32 推理上更快，TFLite 在 INT8 量化部署上更成熟。

### YOLOv8 检测

| 格式 | 大小 |
|------|------|
| PyTorch 原始 | 6.2 MB |
| TFLite FP32 | 12.3 MB |
| TFLite INT8 | 3.3 MB (-73%) |

## 快速开始

```bash
# 1. 安装依赖
pip install tensorflow opencv-python numpy matplotlib onnx onnxruntime torch torchvision ultralytics paho-mqtt flask

# 2. 摄像头实时分类（MobileNetV2）
python day5_camera.py

# 3. INT8 量化实验
python day4_quantize.py

# 4. TFLite vs ONNX 对比
python day8_compare.py

# 5. YOLO + MQTT 实时告警
python day12_yolo_mqtt.py

# 6. Flask 告警面板
python day13_dashboard.py
# 浏览器打开 http://127.0.0.1:5000
项目文件
文件	说明
day1_convert.py	Keras → TFLite 模型转换
day3_inference.py	TFLite 推理延迟测试
day4_quantize.py	INT8 量化对比实验
day5_camera.py	摄像头实时分类
day6_onnx.py	PyTorch → ONNX 导出
day7_onnx_quant.py	ONNX 量化实验
day8_compare.py	TFLite vs ONNX 公平对比
day9_yolo_export.py	YOLOv8 多格式导出
day10_yolo_camera.py	YOLO 摄像头检测
day11_mqtt_test.py	MQTT 发布/订阅测试
day12_yolo_mqtt.py	YOLO + MQTT 实时告警
day13_dashboard.py	Flask 告警面板
api/edgeai_server.py	**FastAPI 统一服务器**（API + Dashboard + 实时监控）
.github/workflows/ci.yml	**GitHub Actions CI/CD**（代码检查 + 单元测试 + Docker 部署）
CHANGELOG.md	项目变更日志（用于 Release 生成）