from ultralytics import YOLO
import os, shutil

# ========== 下载 YOLOv8n ==========
print("加载 YOLOv8n...")
model = YOLO('yolov8n.pt')

# ========== 测试 ==========
print("测试原始模型...")
results = model('https://ultralytics.com/images/bus.jpg')
print(f"检测到 {len(results[0].boxes)} 个目标")

# ========== 导出 TFLite FP32 ==========
print("\n导出 TFLite FP32...")
model.export(format='tflite')
src = 'yolov8n_saved_model/yolov8n_float32.tflite'
shutil.copy(src, 'yolov8n_fp32.tflite')
print(f"  TFLite FP32: {os.path.getsize('yolov8n_fp32.tflite')/1024/1024:.2f} MB")

# ========== 导出 TFLite INT8 ==========
print("\n导出 TFLite INT8（稍等）...")
model.export(format='tflite', int8=True)
src = 'yolov8n_saved_model/yolov8n_integer_quant.tflite'
shutil.copy(src, 'yolov8n_int8.tflite')
print(f"  TFLite INT8: {os.path.getsize('yolov8n_int8.tflite')/1024/1024:.2f} MB")

# ========== 导出 ONNX ==========
print("\n导出 ONNX...")
model.export(format='onnx')
print(f"  ONNX: {os.path.getsize('yolov8n.onnx')/1024/1024:.2f} MB")

print("\n导出完成！TF / INT8 / ONNX 三格式齐全")
