import onnxruntime as ort
import numpy as np
import time
import os

# ==================== ONNX 动态量化 ====================
print("ONNX 动态量化中...")
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input='resnet18.onnx',
    model_output='resnet18_quant.onnx',
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul']
)

# ==================== 大小对比 ====================
orig_size = os.path.getsize('resnet18.onnx') + os.path.getsize('resnet18.onnx.data')
quant_size = os.path.getsize('resnet18_quant.onnx')

print("\n模型大小对比：")
print(f"  FP32:  {orig_size / 1024 / 1024:.2f} MB")
print(f"  量化后: {quant_size / 1024 / 1024:.2f} MB ({quant_size / orig_size * 100:.0f}%)")

# ==================== 推理速度对比 ====================
def benchmark(model_path, runs=100):
    session = ort.InferenceSession(model_path)
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    start = time.time()
    for _ in range(runs):
        session.run(None, {'input': input_data})
    return (time.time() - start) / runs * 1000

print("\n推理速度对比（100次取平均）：")
fp32_lat = benchmark('resnet18.onnx')
print(f"  FP32:  {fp32_lat:.2f} ms")
quant_lat = benchmark('resnet18_quant.onnx')
print(f"  INT8:  {quant_lat:.2f} ms ({quant_lat/fp32_lat*100:.0f}%)")

print("\n量化完成！生成文件：resnet18_quant.onnx")
