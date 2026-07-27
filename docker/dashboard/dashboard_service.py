"""
Flask 告警仪表盘 - 容器化版本
功能：订阅 MQTT → 缓存告警 → Web 页面展示

与 day13_dashboard.py 的区别：
1. Broker 地址/端口通过环境变量配置
2. 端口可配置
3. 代码结构更清晰（HTML 模板分离）
"""

import os
import json
import threading
from flask import Flask, render_template_string, jsonify
from waitress import serve
import paho.mqtt.client as mqtt

# ==================== 环境变量配置 ====================
MQTT_BROKER = os.getenv('MQTT_BROKER', 'mqtt-broker')  # docker-compose 服务名
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'edgeai/alerts')
WEB_PORT = int(os.getenv('WEB_PORT', '5000'))
MAX_ALERTS = int(os.getenv('MAX_ALERTS', '50'))  # 最大缓存条数

# ==================== 共享数据 ====================
alerts = []  # 告警缓存列表，线程安全（GIL 保护）

# HTML 模板（内嵌，避免额外的文件复制）
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="2">
    <title>EdgeAI 告警面板</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f5f5f5; }
        h2 { color: #333; display: flex; align-items: center; gap: 10px; }
        .status { font-size: 14px; color: #666; font-weight: normal; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #4CAF50; display: inline-block; }
        table { border-collapse: collapse; width: 100%%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px 16px; text-align: left; }
        th { background: #2196F3; color: white; text-transform: uppercase; font-size: 13px; letter-spacing: 0.5px; }
        tr:nth-child(even) { background: #f9f9f9; }
        tr:hover { background: #e3f2fd; }
        .person { color: #d32f2f; font-weight: bold; }
        .car { color: #1976D2; font-weight: bold; }
        .dog, .cat { color: #e91e63; font-weight: bold; }
        .bicycle { color: #388E3C; font-weight: bold; }
        .empty { text-align: center; color: #999; padding: 40px; }
        footer { margin-top: 30px; color: #999; font-size: 13px; text-align: center; }
    </style>
</head>
<body>
    <h2><span class="status-dot"></span> EdgeAI 边缘告警面板 <span class="status">| MQTT: {{ broker }}:{{ port }} | 自动刷新: 2秒</span></h2>
    <table>
        <thead>
            <tr><th>#</th><th>时间</th><th>目标类型</th><th>置信度</th></tr>
        </thead>
        <tbody>
            {% if alerts %}
                {% for a in reversed(alerts[-20:]) %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ a['time'] }}</td>
                    <td class="{{ a['type'] }}">{{ a['type'] | upper }}</td>
                    <td>{{ '%.0f' | format(a['conf'] * 100) }}%</td>
                </tr>
                {% endfor %}
            {% else %}
                <tr><td colspan="4" class="empty">暂无告警数据，等待 YOLO 检测服务...</td></tr>
            {% endif %}
        </tbody>
    </table>
    <footer>Edge AI Deployment Project | Containerized with Docker | {{ alerts | length }} 条缓存</footer>
</body>
</html>
'''

# ==================== Flask App ====================
app = Flask(__name__)


@app.route('/')
def index():
    """主页：显示最近 20 条告警"""
    return render_template_string(
        HTML_TEMPLATE,
        alerts=alerts[-20:],
        broker=MQTT_BROKER,
        port=MQTT_PORT
    )


@app.route('/health')
def health():
    """健康检查端点（给监控/Docker 使用）"""
    return jsonify({
        'status': 'ok',
        'alerts_count': len(alerts),
        'broker': f'{MQTT_BROKER}:{MQTT_PORT}'
    })


# ==================== MQTT 订阅者 ====================
def mqtt_listener():
    """后台线程：连接 MQTT Broker 并持续监听告警"""

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] 已连接到 {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(MQTT_TOPIC)
            print(f"[MQTT] 订阅主题: {MQTT_TOPIC}")
        else:
            print(f"[MQTT] 连接失败，错误码: {rc}")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload)
            alerts.append({
                'time': data.get('timestamp', '--:--:--'),
                'type': data.get('type', 'unknown'),
                'conf': data.get('confidence', 0)
            })

            # 防止内存无限增长
            if len(alerts) > MAX_ALERTS:
                alerts.pop(0)

            print(f"[收到] {data.get('type')} ({data.get('confidence', 0):.0%})")

        except json.JSONDecodeError:
            print("[错误] 无效的 JSON 数据")

    client = mqtt.Client(client_id='dashboard_sub')
    client.on_connect = on_connect
    client.on_message = on_message

    # 支持断线重连
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except Exception as e:
        print(f"[MQTT] 连接异常: {e}")
        # 5 秒后重试
        time.sleep(5)
        mqtt_listener()


# ==================== 启动入口 ====================
if __name__ == '__main__':
    import time

    # 启动 MQTT 监听线程
    mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
    mqtt_thread.start()
    print(f"[Dashboard] 启动中... 端口: {WEB_PORT}")

    # 启动 Waitress 生产服务器
    serve(app, host='0.0.0.0', port=WEB_PORT, threads=4)
