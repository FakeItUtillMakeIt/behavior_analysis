from flask import Flask, request, jsonify, Response
from detect_behavior import detect_video_combined_stream, load_config, download_videos, is_rtsp_url
from scenarios import ALARM_NAMES, YOLO_SCENARIO_IDS, VIDEOMAE_SCENARIO_IDS
import time
import os
import json
import queue
from logger import outter_logger, inner_logger


app = Flask(__name__)

# 加载配置
config = load_config()
YOLO_MODEL_PATH = config.get('yolo_model', './models/yolo/sevnce_cls13_1280_20260604.pt')


@app.route('/AI/behavior_analysis/v1/detect', methods=['POST'])
def handle_detect():
    """
    统一检测接口（仅支持视频输入）
    
    请求格式:
    {
        "video": "path/rtsp://...",           // 视频路径（必填）
        "scenario_ids": [0,1,2,3,4,5,6,10,11],  // 可选，默认全部
        "detect_areas": [[x1,y1,x2,y2,x3,y3,x4,y4], ...]  // 可选，检测区域
    }
    
    场景ID说明:
        0-7:  YOLO + 场景检测（安全帽、工装等）
        10:   VideoMAE 行为识别（攀爬）
        11:   VideoMAE 行为识别（打架）
    
    detect_areas说明:
        - 为空或不传：全域检测
        - 格式：多边形顶点坐标列表，每个多边形8个值 [x1,y1,x2,y2,x3,y3,x4,y4]
        - 检测框中心点必须在区域内才会参与场景检测
    """
    try:
        if not request.is_json:
            return jsonify({
                "cost": 0,
                "statusCode": 1,
                "statusMsg": "输入必须是json格式",
                "result": {}
            }), 400
        
        data = request.get_json()
        outter_logger.info(f"Detect request: {data}")
        
        video = data.get("video")
        if not video:
            return jsonify({
                "cost": 0,
                "statusCode": 1,
                "statusMsg": "请提供 video 参数",
                "result": {}
            }), 400
        
        # 默认检测所有场景
        scenario_ids = data.get("scenario_ids", list(ALARM_NAMES.keys()))
        
        # 检测区域参数
        detect_areas = data.get("detect_areas", [])
        
        start_time = time.time()
        
        # 处理视频路径
        video_path = video
        if video.startswith(('http://', 'https://')):
            video_path = download_videos(video)
            if video_path is None:
                return jsonify({
                    "cost": 0,
                    "statusCode": 1,
                    "statusMsg": "视频下载失败",
                    "result": {}
                }), 500
        
        # 检查本地文件（RTSP流不需要检查文件存在）
        if not is_rtsp_url(video_path) and not os.path.exists(video_path):
            return jsonify({
                "cost": 0,
                "statusCode": 1,
                "statusMsg": f"视频文件不存在: {video_path}",
                "result": {}
            }), 400
        
        # 分离场景ID
        yolo_ids = [sid for sid in scenario_ids if sid in YOLO_SCENARIO_IDS]
        videomae_ids = [sid for sid in scenario_ids if sid in VIDEOMAE_SCENARIO_IDS]
        
        inner_logger.info(f"场景分配: YOLO={yolo_ids}, VideoMAE={videomae_ids}")
        
        # 并行流式检测
        detect_result = detect_video_combined_stream(
            video_path, YOLO_MODEL_PATH, yolo_ids, videomae_ids, detect_areas=detect_areas
        )
        
        cost_time = time.time() - start_time
        
        # 构建响应
        result = {
            "array": [5] if detect_result['has_alarm'] else [4],
            "yolo_scenarios": detect_result['yolo_scenarios'],
            "behavior": detect_result['behavior'],
            "detections_count": detect_result['detections_count']
        }
        
        if detect_result['image_view_base64']:
            result["image_view_base64"] = detect_result['image_view_base64']
        
        response = {
            "cost": round(cost_time, 4),
            "statusCode": 0,
            "statusMsg": "success",
            "result": result
        }
        
        outter_logger.info(f"Detect response: cost={cost_time:.3f}s, alarm={detect_result['has_alarm']}")
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        outter_logger.error(error_msg)
        print(f"Error: {error_msg}")
        return jsonify({
            "cost": 0,
            "statusCode": 1,
            "statusMsg": f"请求数据异常: {str(e)}",
            "result": {}
        }), 500


@app.route('/AI/behavior_analysis/v1/detect_stream', methods=['POST'])
def handle_detect_stream():
    """
    实时流式检测接口 - SSE (Server-Sent Events)
    
    请求格式:
    {
        "video": "rtsp://...",              // 视频流地址（必填）
        "scenario_ids": [0,1,2,3,4,5,6,10,11],  // 可选
        "detect_areas": [[x1,y1,x2,y2,x3,y3,x4,y4], ...]  // 可选，检测区域
    }
    
    返回: SSE流，每个告警事件实时推送
    """
    try:
        if not request.is_json:
            return jsonify({
                "cost": 0,
                "statusCode": 1,
                "statusMsg": "输入必须是json格式",
                "result": {}
            }), 400
        
        data = request.get_json()
        outter_logger.info(f"Stream detect request: {data}")
        
        video = data.get("video")
        if not video:
            return jsonify({
                "cost": 0,
                "statusCode": 1,
                "statusMsg": "请提供 video 参数",
                "result": {}
            }), 400
        
        scenario_ids = data.get("scenario_ids", list(ALARM_NAMES.keys()))
        
        # 检测区域参数
        detect_areas = data.get("detect_areas", [])
        
        # 分离场景ID
        yolo_ids = [sid for sid in scenario_ids if sid in YOLO_SCENARIO_IDS]
        videomae_ids = [sid for sid in scenario_ids if sid in VIDEOMAE_SCENARIO_IDS]
        
        # 使用队列进行线程间通信
        alert_queue = queue.Queue()
        
        def alert_callback(frame_info):
            """告警回调函数"""
            alert_queue.put(frame_info)
        
        def generate():
            """SSE 生成器"""
            # 启动检测线程
            import threading
            
            def detect_worker():
                detect_video_combined_stream(
                    video, YOLO_MODEL_PATH, yolo_ids, videomae_ids, 
                    callback=alert_callback, detect_areas=detect_areas
                )
                # 检测完成后发送结束信号
                alert_queue.put(None)
            
            detect_thread = threading.Thread(target=detect_worker, daemon=True)
            detect_thread.start()
            
            # 实时推送告警
            while True:
                try:
                    data = alert_queue.get(timeout=1)
                    if data is None:
                        # 检测完成
                        #yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                        break
                    
                    # 只推送有告警的数据
                    if isinstance(data, dict):
                        # VideoMAE告警（流式检测）
                        if data.get('type') == 'videomae_alarm':
                            event_data = {
                                'type': 'videomae_alarm',
                                'behavior': data.get('behavior', {}),
                                'frame': data.get('frame', 0),
                                'timestamp': data.get('timestamp', time.time())
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        
                        # YOLO告警（已包含图片）
                        elif data.get('has_alarm', False):
                            event_data = {
                                'type': 'yolo_alarm',
                                'frame': data.get('frame', 0),
                                'scenarios': data.get('scenarios', {}),
                                'detection_count': data.get('detection_count', 0),
                                'timestamp': data.get('timestamp', time.time()),
                                'image_base64': data.get('image_base64', '')
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                    
                except queue.Empty:
                    # 超时，发送心跳
                    #yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    pass
                except Exception as e:
                    inner_logger.error(f"SSE推送异常: {e}")
                    break
        pass
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        outter_logger.error(error_msg)
        return jsonify({
            "cost": 0,
            "statusCode": 1,
            "statusMsg": f"请求数据异常: {str(e)}",
            "result": {}
        }), 500


@app.route('/AI/behavior_analysis/v1/scenarios', methods=['GET'])
def get_scenarios():
    """获取支持的场景列表"""
    scenarios = []
    for scenario_id, name in ALARM_NAMES.items():
        scenarios.append({
            "id": scenario_id,
            "name": name,
            "type": "yolo" if scenario_id in YOLO_SCENARIO_IDS else "videomae"
        })
    return jsonify({
        "statusCode": 0,
        "result": {
            "scenarios": scenarios
        }
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10235, debug=False, threaded=True)
