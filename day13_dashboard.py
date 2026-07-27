from flask import Flask, jsonify
import paho.mqtt.client as mqtt
import threading
import json
from waitress import serve

# ======= 共享数据：列表，存最近 50 条告警 =======
alerts = []

# ======= Flask：提供网页 =======
app = Flask(__name__)

@app.route('/')
def index():
    # 返回 HTML 页面（直接写在字符串里）
    html = '''
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f8f9fa; }
            h2 { color: #333; }
            table { border-collapse: collapse; width: 100%%; }
            th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
            th { background: #4CAF50; color: white; }
            tr:nth-child(even) { background: #f2f2f2; }
            .person { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>边缘AI 告警面板</h2>
        <table>
            <tr><th>时间</th><th>类型</th><th>置信度</th></tr>
    '''
    for a in reversed(alerts[-20:]):
        cls = 'person' if a.get('type') == 'person' else ''
        html += f'<tr><td>{a["time"]}</td><td class="{cls}">{a["type"]}</td><td>{a["conf"]:.0%}</td></tr>'
    html += '</table></body></html>'
    return html

# ======= MQTT：后台监听告警 =======
def mqtt_listener():
    def on_message(client, userdata, msg):
        data = json.loads(msg.payload)
        alerts.append({
            'time': data['timestamp'],
            'type': data['type'],
            'conf': data['confidence']
        })

    client = mqtt.Client(client_id='dash_sub')
    client.on_message = on_message
    client.connect('test.mosquitto.org', 1883)
    client.subscribe('edgeai/alerts')
    client.loop_forever()

# 启动 MQTT 监听线程
threading.Thread(target=mqtt_listener, daemon=True).start()

# 启动 Flask 服务器
if __name__ == '__main__':
    print("Dashboard 启动: http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000, threads=4)
