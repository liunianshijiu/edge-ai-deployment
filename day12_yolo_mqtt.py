from ultralytics import YOLO
import paho.mqtt.client as mqtt
import cv2
import json
import time

# ==================== 连 MQTT Broker ====================
pub = mqtt.Client(client_id='yolo_cam')
pub.connect('test.mosquitto.org', 1883)

# ==================== 加载 YOLO ====================
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

last_alert_time = {}  # 防止同一物体频繁告警

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # 遍历检测到的每个物体
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])

        # 只对兴趣目标发告警，且至少隔 5 秒
        if name in ['person', 'car', 'dog', 'cat', 'bicycle']:
            now = time.time()
            if now - last_alert_time.get(name, 0) > 5:
                alert = {
                    "type": name,
                    "confidence": round(conf, 2),
                    "timestamp": time.strftime('%H:%M:%S')
                }
                pub.publish('edgeai/alerts', json.dumps(alert))
                print(f"[MQTT] {name} (置信度: {conf:.2f})")
                last_alert_time[name] = now

    cv2.putText(annotated, f'MQTT: edgeai/alerts', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('YOLO + MQTT Alert', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pub.disconnect()
cv2.destroyAllWindows()
