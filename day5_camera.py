import cv2
import numpy as np
import tensorflow as tf
import json
import urllib.request
import time

# ==================== 加载模型和标签 ====================
print("加载 INT8 量化模型...")
interpreter = tf.lite.Interpreter(model_path='mobilenet_v2_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 下载 ImageNet 1000类标签（只需要一次）
print("下载分类标签...")
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as f:
    labels_raw = json.load(f)
labels = {int(k): v[1] for k, v in labels_raw.items()}
print(f"加载了 {len(labels)} 个类别")

# ==================== 打开摄像头 ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

print("\n按 Q 退出")
print("=" * 40)

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理：缩放 + BGR转RGB
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    # 推理
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_id = int(np.argmax(output[0]))
    label = labels.get(pred_id, f"未知({pred_id})")

    # 显示结果
    cv2.putText(frame, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow('Edge AI Demo - PC Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 算帧率
    frame_count += 1
    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - prev_time)
        prev_time = now

cap.release()
cv2.destroyAllWindows()
print("退出")
