"""
YOLO 目标检测服务 - 容器化版本
功能：读取摄像头 → YOLOv8n INT8 推理 → MQTT 发布告警

与 day12_yolo_mqtt.py 的区别：
1. Broker 地址/端口通过环境变量配置
2. 使用 TFLite INT8 模型（不需要 PyTorch，镜像更轻量）
3. 摄像头设备路径可配置（Docker 中通常为 /dev/video0）
"""

import os
import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ==================== 环境变量配置 ====================
MQTT_BROKER = os.getenv('MQTT_BROKER', 'mqtt-broker')  # docker-compose 服务名
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'edgeai/alerts')
CAMERA_DEVICE = os.getenv('CAMERA_DEVICE', '/dev/video0')  # Linux 设备路径
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/yolov8n_int8.tflite')
COOLDOWN_SEC = int(os.getenv('COOLDOWN_SEC', '5'))  # 同类目标告警冷却时间

# 告警目标类型（COCO 数据集类别名子集）
TARGET_CLASSES = ['person', 'car', 'dog', 'cat', 'bicycle']


# ==================== TFLite 推理引擎封装 ====================
class TFLiteDetector:
    """封装 TFLite INT8 模型的加载和推理流程"""
    
    def __init__(self, model_path):
        import tflite_runtime.interpreter as tflite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入/输出详情
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 输入尺寸
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        print(f"[模型] 加载成功: {model_path}")
        print(f"[模型] 输入尺寸: {self.input_width}x{self.input_height}")
    
    def preprocess(self, frame):
        """图像预处理：resize + 归一化 + 维度扩展(HWC→NHWC)"""
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        # 转换为 float32 并归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # 添加 batch 维度: (H,W,C) → (1,H,W,C)
        return np.expand_dims(normalized, axis=0)
    
    def detect(self, frame):
        """
        执行目标检测
        返回: detections列表, 每个元素为 {class_name, confidence, box: [x1,y1,x2,y2]}
        """
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # 获取输出: shape=(1, 84, 8400) → 转置为 (1, 8400, 84)
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = output.transpose((0, 2, 1))  # (1, 8400, 84)
        
        # 解析检测结果（简化版后处理）
        detections = []
        predictions = output[0]  # (8400, 84)
        
        # 置信度阈值
        conf_threshold = 0.5
        
        for pred in predictions:
            # 前4个是边界框坐标 [cx, cy, w, h]
            # 后80个是各类别概率 (COCO 80类)
            class_probs = pred[4:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]
            
            if confidence > conf_threshold:
                # COCO 类别名映射（常用 subset）
                coco_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'bus',
                    'truck', 'dog', 'cat', 'bird', 'cow'
                ]
                name = coco_names[class_id] if class_id < len(coco_names) else f'class_{class_id}'
                
                cx, cy, w, h = pred[:4]
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                
                detections.append({
                    'name': name,
                    'confidence': float(confidence),
                    'box': [x1, y1, x2, y2]
                })
        
        return detections


# ==================== MQTT 发布客户端 ====================
def create_mqtt_client():
    """创建并连接 MQTT 客户端"""
    client = mqtt.Client(client_id='yolo_detector')
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    print(f"[MQTT] 已连接到 {MQTT_BROKER}:{MQTT_PORT}")
    return client


# ==================== 主循环 ====================
def main():
    # 加载 TFLite 模型
    detector = TFLiteDetector(MODEL_PATH)
    
    # 连接 MQTT
    mqtt_client = create_mqtt_client()
    
    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头: {CAMERA_DEVICE}")
        return
    
    print("[摄像头] 启动成功")
    print(f"[告警] 监控目标: {TARGET_CLASSES}")
    print(f"[告警] 冷却时间: {COOLDOWN_SEC}秒")
    print("="*50)
    
    last_alert_time = {}  # 防止同类目标频繁告警
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # YOLO 推理
            detections = detector.detect(frame)
            
            # 在画面上绘制检测框
            for det in detections:
                x1, y1, x2, y2 = det['box']
                label = f"{det['name']} {det['confidence']:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 只对目标类型发布 MQTT 告警
                if det['name'] in TARGET_CLASSES:
                    now = time.time()
                    if now - last_alert_time.get(det['name'], 0) > COOLDOWN_SEC:
                        alert = {
                            "type": det['name'],
                            "confidence": round(det['confidence'], 2),
                            "timestamp": time.strftime('%H:%M:%S')
                        }
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(alert), qos=1)
                        print(f"  [MQTT →] {det['name']} ({det['confidence']:.0%})")
                        last_alert_time[det['name']] = now
            
            # 显示状态信息
            status = f"MQTT: {MQTT_BROKER}:{MQTT_PORT} | Topic: {MQTT_TOPIC}"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('EdgeAI-YOLO Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
    
    finally:
        cap.release()
        mqtt_client.disconnect()
        cv2.destroyAllWindows()
        print("[退出] 资源已释放")


if __name__ == '__main__':
    main()
