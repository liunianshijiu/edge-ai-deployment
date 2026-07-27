import paho.mqtt.client as mqtt
import json
import time
import threading

# ====== 统计 ======
sent = 0
received = 0
lock = threading.Lock()

# ====== 订阅端 ======
def on_message(client, userdata, msg):
    global received
    with lock:
        received += 1

def subscriber():
    client = mqtt.Client(client_id='stress_sub')
    client.on_message = on_message
    client.connect('test.mosquitto.org', 1883)
    client.subscribe('edgeai/stress_test')
    client.loop_forever()

threading.Thread(target=subscriber, daemon=True).start()
time.sleep(1)

# ====== 压测：每秒100条连发10秒 ======
pub = mqtt.Client(client_id='stress_pub')
pub.connect('test.mosquitto.org', 1883)

print("开始压测：每秒100条，持续10秒...")
for i in range(10):
    batch_start = time.time()
    for _ in range(100):
        pub.publish('edgeai/stress_test', json.dumps({"msg": sent}))
        sent += 1
    elapsed = time.time() - batch_start
    time.sleep(max(0, 1.0 - elapsed))

time.sleep(3)
pub.disconnect()

print(f"\n发送: {sent} 条")
print(f"收到: {received} 条")
print(f"丢包率: {(sent - received) / sent * 100:.2f}%")
