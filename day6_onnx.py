import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
import time

# ==================== Step 1: 加载 PyTorch 预训练模型 ====================
print("加载 ResNet18 预训练模型...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# ==================== Step 2: 导出 ONNX ====================
print("导出 ONNX 格式...")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, 'resnet18.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    dynamo=False
)

# ==================== Step 3: 验证 ONNX 模型 ====================
onnx_model = onnx.load('resnet18.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证通过！")

# ==================== Step 4: ONNX Runtime 推理 ====================
print("ONNX Runtime 推理测试...")
session = ort.InferenceSession('resnet18.onnx')
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'input': input_data})
print(f"输出形状: {outputs[0].shape}")  # 应该输出 (1, 1000)

# ==================== Step 5: 测延迟 ====================
print("推理延迟测试（100次取平均）...")
start = time.time()
for _ in range(100):
    session.run(None, {'input': input_data})
latency = (time.time() - start) / 100 * 1000

# ==================== 汇总 ====================
import os
print("\n" + "=" * 50)
print(f"ONNX 模型大小: {os.path.getsize('resnet18.onnx') / 1024 / 1024:.2f} MB")
print(f"ONNX Runtime 延迟: {latency:.2f} ms")
print(f"\n之前 TFLite 的数据：")
print(f"  FP32:  13.35 MB, 6.23 ms")
print(f"  INT8:   3.84 MB, 3.46 ms")
