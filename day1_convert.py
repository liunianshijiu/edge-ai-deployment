import tensorflow as tf

# 下载 MobileNetV2 预训练模型
print("下载 MobileNetV2 模型...")
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# 转换成 TFLite
print("转换为 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存
with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"转换完成！模型大小: {len(tflite_model) / 1024 / 1024:.2f} MB")
