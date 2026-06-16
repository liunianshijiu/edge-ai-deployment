import tensorflow as tf
import numpy as np
import time

# ==================== 准备：重新加载模型 ====================
print("=" * 50)
print("加载 MobileNetV2 预训练模型...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# ==================== 方法1：动态范围量化 ====================
print("\n【方法1】动态范围量化")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_dynamic = converter.convert()

with open('mobilenet_v2_dynamic.tflite', 'wb') as f:
    f.write(quant_dynamic)
print(f"  动态量化后: {len(quant_dynamic) / 1024:.1f} KB")

# ==================== 方法2：INT8 全整数量化 ====================
print("\n【方法2】INT8 全整数量化（需要校准数据，稍等...）")

def representative_dataset():
    for _ in range(100):
        data = np.random.randn(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
quant_int8 = converter.convert()

with open('mobilenet_v2_int8.tflite', 'wb') as f:
    f.write(quant_int8)
print(f"  INT8量化后: {len(quant_int8) / 1024:.1f} KB")

# ==================== 大小对比 ====================
print("\n" + "=" * 50)
print("模型大小对比：")
print("-" * 50)
original_size = 13.35  # MB (之前跑出来的)
print(f"  FP32 原始:     {original_size:>8.2f} MB   (100%)")
print(f"  动态量化:      {len(quant_dynamic)/1024/1024:>8.2f} MB   ({len(quant_dynamic)/1024/1024/original_size*100:.0f}%)")
print(f"  INT8 量化:     {len(quant_int8)/1024/1024:>8.2f} MB   ({len(quant_int8)/1024/1024/original_size*100:.0f}%)")

# ==================== 推理速度对比 ====================
print("\n推理速度对比（100次取平均）：")
print("-" * 50)

def benchmark_latency(model_path, runs=100):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 根据模型类型准备输入
    if input_details[0]['dtype'] == np.uint8:
        dummy = np.random.randint(0, 255, input_details[0]['shape'], dtype=np.uint8)
    else:
        dummy = np.random.randn(*input_details[0]['shape']).astype(np.float32)

    start = time.time()
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    return (time.time() - start) / runs * 1000

print(f"  FP32 原始:     {6.23:>7.2f} ms   (100%)")
dyn_lat = benchmark_latency('mobilenet_v2_dynamic.tflite')
print(f"  动态量化:      {dyn_lat:>7.2f} ms   ({dyn_lat/6.23*100:.0f}%)")
int8_lat = benchmark_latency('mobilenet_v2_int8.tflite')
print(f"  INT8 量化:     {int8_lat:>7.2f} ms   ({int8_lat/6.23*100:.0f}%)")

print("\n" + "=" * 50)
print("量化实验完成！生成文件：")
print("  - mobilenet_v2_dynamic.tflite（动态量化）")
print("  - mobilenet_v2_int8.tflite（INT8 全整数量化）")
