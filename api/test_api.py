"""
EdgeAI Unified Server 测试脚本
用法: python api/test_api.py

功能:
1. 检查 API 是否在线
2. 上传图片检测
3. 查看历史记录
4. 检查 Dashboard 面板
5. 检查 MQTT 告警状态
"""

import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8080"


def print_header(num, title):
    print(f"\n{'='*55}")
    print(f"  {num}. {title}")
    print(f"{'-'*55}")


def test_health():
    print_header(1, "健康检查 (GET /health)")
    
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        
        print(f"  状态码: {resp.status_code}")
        print(f"  服务状态: {data.get('status')}")
        print(f"  模型文件: {data.get('model')} ({data.get('model_size_mb')} MB)")
        print(f"  已运行: {data.get('uptime_seconds')} 秒")
        print(f"  总检测次数: {data.get('total_detections')}")
        print(f"  检测历史: {data.get('history_count')} 条")
        print(f"  MQTT告警: {data.get('mqtt_alerts_count')} 条")
        return data.get('status') == 'ok'
    except Exception as e:
        print(f"  [错误] {e}")
        return False


def test_detect(image_path: str):
    print_header(2, "目标检测 (POST /detect)")
    
    if not Path(image_path).exists():
        print(f"  [跳过] 图片不存在: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            resp = requests.post(
                f"{BASE_URL}/detect",
                files=files,
                params={'draw_boxes': True},
                timeout=30
            )
        
        data = resp.json()
    except Exception as json_err:
        print(f"  状态码: {resp.status_code}")
        print(f"  Content-Type: {resp.headers.get('content-type', 'unknown')}")
        print(f"  原始响应: {resp.text[:500]}")
        print(f"  [JSON解析失败] {json_err}")
        return None
    
    try:
        if not data.get('success'):
            print(f"  [失败] {data}")
            return None
        
        print(f"  状态码: {resp.status_code} | 推理耗时: {data.get('inference_ms')} ms")
        print(f"  图片尺寸: {data.get('image_size')} | 目标数: {data.get('total_objects')}")
        
        for i, det in enumerate(data.get('detections', []), 1):
            bbox = det['bbox']
            print(f"    {i}. {det['class_name']:10s} | {det['confidence']:.0%} | [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        if 'annotated_image' in data:
            import base64
            out_path = Path(image_path).stem + '_detected.jpg'
            with open(out_path, 'wb') as out:
                out.write(base64.b64decode(data['annotated_image']))
            print(f"\n  ✅ 标注图已保存: {out_path}")
        
        return data
        
    except Exception as e:
        print(f"  [错误] {e}")
        return None


def test_history():
    print_header(3, "检测历史 (GET /history)")
    
    try:
        resp = requests.get(f"{BASE_URL}/history?limit=5", timeout=5)
        data = resp.json()
        
        print(f"  总记录数: {data.get('total')}")
        for r in data.get('records', []):
            summary = ", ".join(
                f"{s['class_name']}x{s['count']}" 
                for s in r.get('detections_summary', [])
            )
            print(f"    [{r['timestamp']}] {r['total_objects']}个 -> {summary}")
            
    except Exception as e:
        print(f"  [错误] {e}")


def test_dashboard():
    print_header(4, "Dashboard 面板 (GET /dashboard)")
    
    try:
        resp = requests.get(f"{BASE_URL}/dashboard", timeout=5)
        
        if resp.status_code == 200 and '<html' in resp.text.lower():
            # 统计页面内容
            has_mqtt_status = 'MQTT' in resp.text or 'mqtt' in resp.text.lower()
            has_table = '<table>' in resp.text
            has_stats = 'stat-card' in resp.text
            
            print(f"  状态码: {resp.status_code}")
            print(f"  页面大小: {len(resp.text)} 字符")
            print(f"  MQTT状态显示: {'✅' if has_mqtt_status else '❌'}")
            print(f"  告警表格: {'✅' if has_table else '❌'}")
            print(f"  统计卡片: {'✅' if has_stats else '❌'}")
            print(f"\n  📎 面板地址: {BASE_URL}/dashboard")
            return True
        else:
            print(f"  [异常] 状态码: {resp.status_code}, 非 HTML 响应")
            return False
            
    except Exception as e:
        print(f"  [错误] {e}")
        return False


def test_alerts():
    print_header(5, "MQTT 告警数据 (GET /api/alerts)")
    
    try:
        resp = requests.get(f"{BASE_URL}/api/alerts", timeout=5)
        data = resp.json()
        
        count = data.get('count', 0)
        alerts = data.get('alerts', [])
        
        print(f"  缓存告警数: {count}")
        if alerts:
            latest = alerts[-1]
            print(f"  最新告警: {latest.get('type')} ({latest.get('conf',0):.0%}) @ {latest.get('time')}")
            
            # 统计各类别
            types = {}
            for a in alerts:
                t = a.get('type', '?')
                types[t] = types.get(t, 0) + 1
            summary = ", ".join(f"{k}x{v}" for k, v in sorted(types.items()))
            print(f"  类别分布: {summary}")
        else:
            print(f"  暂无告警（需 YOLO+MQTT 服务发送消息）")
        
        return True
        
    except Exception as e:
        print(f"  [错误] {e}")
        return False


def main():
    print("\n" + "="*55)
    print("   🔍 EdgeAI Unified Server v2.0 测试工具")
    print("="*55)
    
    # Step 1
    if not test_health():
        print("\n  ⚠️  API 未启动！请先运行:")
        print("     python api/edgeai_server.py")
        sys.exit(1)
    
    # Step 2: Detect
    test_images = ['bus.jpg', 'test.jpg']
    image_path = next((str(p) for p in map(Path, test_images) if p.exists()), None)
    
    if image_path:
        test_detect(image_path)
    else:
        print("\n  [跳过] 无测试图片")
    
    # Step 3-5
    test_history()
    test_dashboard()
    test_alerts()
    
    print(f"\n{'='*55}")
    print("  ✅ 全部测试完成!")
    print(f"     Swagger: {BASE_URL}/docs")
    print(f"     Panel:   {BASE_URL}/dashboard")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
