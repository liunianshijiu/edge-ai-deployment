import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import time
import os

# ========== 加载同一个 MobileNetV2 模型 ==========
print("加载 MobileNetV2 模型...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# ========== 导出 ONNX ==========
print("导出 ONNX 格式...")
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name='input'),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open('mobilenet_v2_onnx.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())
print("ONNX 导出完成")

# ========== 大小对比 ==========
print("\n" + "=" * 50)
print("模型大小对比（同一个 MobileNetV2）：")
print("-" * 50)
tflite_size = os.path.getsize('mobilenet_v2.tflite') / 1024 / 1024
onnx_size = os.path.getsize('mobilenet_v2_onnx.onnx') / 1024 / 1024
int8_size = os.path.getsize('mobilenet_v2_int8.tflite') / 1024 / 1024
print(f"  TFLite FP32:  {tflite_size:.2f} MB")
print(f"  ONNX FP32:    {onnx_size:.2f} MB")
print(f"  TFLite INT8:  {int8_size:.2f} MB")

# ========== 推理速度对比 ==========
print("\n推理延迟对比（100次取平均，同一批假数据）：")
print("-" * 50)

dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)

# TFLite FP32
interpreter = tf.lite.Interpreter(model_path='mobilenet_v2.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

start = time.time()
for _ in range(100):
    interpreter.set_tensor(inp['index'], dummy)
    interpreter.invoke()
    _ = interpreter.get_tensor(out['index'])
tflite_fp32 = (time.time() - start) / 100 * 1000
print(f"  TFLite FP32:  {tflite_fp32:.2f} ms")

# ONNX FP32
session = ort.InferenceSession('mobilenet_v2_onnx.onnx')
start = time.time()
for _ in range(100):
    _ = session.run(None, {'input': dummy})
onnx_fp32 = (time.time() - start) / 100 * 1000
print(f"  ONNX FP32:    {onnx_fp32:.2f} ms")

# TFLite INT8
interpreter = tf.lite.Interpreter(model_path='mobilenet_v2_int8.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
dummy_u8 = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)

start = time.time()
for _ in range(100):
    interpreter.set_tensor(inp['index'], dummy_u8)
    interpreter.invoke()
    _ = interpreter.get_tensor(out['index'])
tflite_int8 = (time.time() - start) / 100 * 1000
print(f"  TFLite INT8:  {tflite_int8:.2f} ms")

# ========== 总结 ==========
print("\n" + "=" * 50)
print("对比结论：")
print(f"  FP32 速度：ONNX / TFLite = {onnx_fp32/tflite_fp32:.2f}x")
print(f"  量化收益：TFLite INT8 比 FP32 快 {tflite_fp32/tflite_int8:.1f}x，大小缩 {tflite_size/int8_size:.0f}%")
print(f"  ONNX 适合跨框架交换模型，TFLite 在量化部署上更成熟")
