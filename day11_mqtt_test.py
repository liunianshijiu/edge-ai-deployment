import paho.mqtt.client as mqtt
import threading
import json
import time

# ======= 用一个线程模拟 Subscriber =======
received = []

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    received.append(data)
    print(f"[订阅者收到] {data['type']}: {data['message']}")

def subscriber():
    client = mqtt.Client(client_id='sub_01')
    client.on_message = on_message
    client.connect('test.mosquitto.org', 1883)
    client.subscribe('edgeai/alerts')
    client.loop_forever()

sub_thread = threading.Thread(target=subscriber, daemon=True)
sub_thread.start()
time.sleep(1)

# ======= 主线程当 Publisher =======
pub = mqtt.Client(client_id='pub_01')
pub.connect('test.mosquitto.org', 1883)

alerts = [
    {"type": "person", "message": "检测到行人", "confidence": 0.92},
    {"type": "car",    "message": "检测到车辆", "confidence": 0.88},
    {"type": "dog",    "message": "检测到动物", "confidence": 0.75},
]

for alert in alerts:
    pub.publish('edgeai/alerts', json.dumps(alert))
    print(f"[发布者发送] {alert['type']}: {alert['message']}")
    time.sleep(1)

pub.disconnect()
time.sleep(2)

print(f"\n共收到 {len(received)} 条消息")
for r in received:
    print(f"  - {r['type']}: {r['message']} (置信度: {r['confidence']})")
