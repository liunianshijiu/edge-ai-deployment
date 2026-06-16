import numpy as np
import tensorflow as tf
import time

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='mobilenet_v2.tflite')
interpreter.allocate_tensors()

# 获取输入输出信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"输入形状: {input_details[0]['shape']}")
print(f"输出形状: {output_details[0]['shape']}")

# 假数据推理 100 次，测延迟
dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

start = time.time()
for _ in range(100):
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end = time.time()

print(f"100次推理总耗时: {(end-start)*1000:.0f} ms")
print(f"平均推理延迟: {(end-start)/100*1000:.2f} ms")
