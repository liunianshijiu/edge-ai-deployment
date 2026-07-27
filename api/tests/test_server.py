"""
EdgeAI Unified Server 单元测试
================================

使用方法（本地）:
    cd 项目根目录
    .\\edgeai_env\\Scripts\\Activate.ps1
    pytest api/tests/ -v

设计说明:
    - 用 FastAPI TestClient 在内存中发起 HTTP 请求，不需要启动真实服务器
    - 用 unittest.mock 把 YOLO 模型替换成假对象，不需要 yolov8n.pt 模型文件
    - 不连接真实 MQTT Broker，避免 CI 环境网络依赖
    - 这样测试可以在任何环境（含 GitHub Actions ubuntu）稳定通过
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

# 把 api/ 目录加入搜索路径，才能 import edgeai_server
sys.path.insert(0, str(Path(__file__).parent.parent))

import edgeai_server
from fastapi.testclient import TestClient


def _make_mock_model():
    """构造一个假的 YOLO 模型，run_detection 调用它返回固定的检测结果"""
    fake_box = MagicMock()
    fake_box.cls = [0]                       # 类别 id = 0
    fake_box.conf = [0.9]                    # 置信度 0.9
    fake_box.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]])  # 边界框坐标

    fake_result = MagicMock()
    fake_result.boxes = [fake_box]           # 有 1 个检测框
    fake_result.names = {0: "person"}        # 类别 id 0 -> person
    fake_result.orig_shape = (100, 100)      # 原图尺寸 (h, w)，供 image_size 使用

    fake_model = MagicMock()
    fake_model.return_value = [fake_result]   # model(img) -> [fake_result]
    return fake_model


@pytest.fixture
def client():
    """提供 TestClient，并把模型和 MQTT 监听都 mock 掉

    关键：直接 patch load_model，让它装上假模型。
    否则在本地（yolov8n.pt 真实存在）startup 会加载真实模型覆盖 mock，
    导致测试结果依赖环境。patch load_model 后本地/CI 行为一致。
    """
    def _fake_load_model():
        edgeai_server.model = _make_mock_model()
        return True

    with patch.object(edgeai_server, "load_model", _fake_load_model), \
         patch.object(edgeai_server, "mqtt_listener", lambda: None):
        with TestClient(edgeai_server.app) as c:
            yield c


def _dummy_image_bytes():
    """生成一张 100x100 的黑色测试图（JPEG 编码后的字节）"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health(client):
    """健康检查接口应返回 200 且 status=ok"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_detect_success(client):
    """上传图片检测应返回 200，且解析出 1 个 person 目标"""
    image_bytes = _dummy_image_bytes()
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    resp = client.post("/detect", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_objects"] >= 1
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["class_name"] == "person"
    assert 0.0 <= data["detections"][0]["confidence"] <= 1.0


def test_detect_rejects_non_image(client):
    """上传非图片文件应被拒绝（400）"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    resp = client.post("/detect", files=files)
    assert resp.status_code == 400


def test_history_after_detect(client):
    """检测后历史记录应包含刚才的检测"""
    image_bytes = _dummy_image_bytes()
    files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
    client.post("/detect", files=files)

    resp = client.get("/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


def test_dashboard_renders(client):
    """Dashboard 面板应返回 HTML 页面"""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "html" in resp.text.lower()


def test_root_endpoint(client):
    """根路径应返回服务信息 JSON"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()
