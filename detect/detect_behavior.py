
from ultralytics import YOLO
import json,numpy as np
import base64
import cv2
import os
import torch
import threading,time,requests
from PIL import Image
from logger import inner_logger
from drawing_utils import draw_alarm_frame
from scenarios import load_scenario_config

# VideoMAE 相关导入
try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    VIDEOMAE_AVAILABLE = True
except ImportError:
    VIDEOMAE_AVAILABLE = False
    inner_logger.warning("transformers 未安装，VideoMAE 功能不可用")

# 全局配置和模型缓存
_config = None
_videomae_model = None
_videomae_processor = None
_videomae_class_mapping = None


def load_config():
    """加载配置文件"""
    global _config
    if _config is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config', 'config.json')
        try:
            with open(config_path, 'r') as f:
                _config = json.load(f)
            inner_logger.info(f"加载配置文件: {config_path}")
        except Exception as e:
            inner_logger.error(f"加载配置文件失败: {e}")
            _config = {
                "yolo_model": "./models/yolo/sevnce_cls13_1280_20260604.pt",
                "videomae_model": "./models/videomae",
                "videomae": {
                    "num_frames": 16,
                    "threshold": 0.7,
                    "device": "auto"
                }
            }
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key in ['yolo_model', 'videomae_model']:
        if key in _config and not os.path.isabs(_config[key]):
            _config[key] = os.path.join(project_root, _config[key].lstrip('./'))
    
    # 加载场景检测配置（聚集、睡岗、离岗阈值）
    load_scenario_config(_config)
    
    return _config


def load_videomae_model():
    """加载 VideoMAE 模型"""
    global _videomae_model, _videomae_processor, _videomae_class_mapping
    
    if _videomae_model is not None:
        return _videomae_model, _videomae_processor, _videomae_class_mapping
    
    if not VIDEOMAE_AVAILABLE:
        inner_logger.error("transformers 未安装，无法加载 VideoMAE 模型")
        return None, None, None
    
    config = load_config()
    model_dir = config.get('videomae_model', './models/videomae')
    
    if not os.path.exists(model_dir):
        inner_logger.error(f"VideoMAE 模型目录不存在: {model_dir}")
        return None, None, None
    
    try:
        device_config = config.get('videomae', {}).get('device', 'auto')
        if device_config == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        inner_logger.info(f"加载 VideoMAE 模型: {model_dir}")
        _videomae_model = VideoMAEForVideoClassification.from_pretrained(model_dir)
        _videomae_model.to(device)
        _videomae_model.eval()
        
        try:
            _videomae_processor = VideoMAEImageProcessor.from_pretrained(model_dir)
        except:
            _videomae_processor = None
            inner_logger.warning("VideoMAEImageProcessor 加载失败，使用手动预处理")
        
        class_mapping_path = os.path.join(model_dir, "class_mapping.json")
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                _videomae_class_mapping = json.load(f)
                _videomae_class_mapping = {int(k): v for k, v in _videomae_class_mapping.items()}
        else:
            _videomae_class_mapping = {0: "climb", 1: "fight"}
        
        inner_logger.info(f"VideoMAE 模型加载成功，类别: {_videomae_class_mapping}")
        return _videomae_model, _videomae_processor, _videomae_class_mapping
        
    except Exception as e:
        inner_logger.error(f"VideoMAE 模型加载失败: {e}")
        return None, None, None


def load_video_frames(video_path, num_frames=16):
    """从视频中均匀采样帧（支持 mp4 和 rtsp）"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        inner_logger.error(f"无法打开视频: {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        frames = []
        frame_count = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % max(1, int(fps / 10)) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            frame_count += 1
        cap.release()
        
        if len(frames) == 0:
            inner_logger.error(f"无法从 RTSP 流读取帧: {video_path}")
            return None, None
        
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return frames[:num_frames], fps
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
        
        return frames[:num_frames], fps


def predict_videomae(video_path, num_frames=16, threshold=0.7):
    """使用 VideoMAE 预测视频行为"""
    model, processor, class_mapping = load_videomae_model()
    
    if model is None:
        inner_logger.error("VideoMAE 模型未加载")
        return None
    
    config = load_config()
    device_config = config.get('videomae', {}).get('device', 'auto')
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    frames, fps = load_video_frames(video_path, num_frames)
    if frames is None:
        inner_logger.error(f"视频帧加载失败: {video_path}")
        return None
    
    if processor:
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        processed_frames = [transform(f) for f in frames]
        pixel_values = torch.stack(processed_frames, dim=1).unsqueeze(0).to(device)
        inputs = {'pixel_values': pixel_values}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(-1).item()
    confidence = probs[0][pred_idx].item()
    
    if confidence < threshold:
        pred_class = "unknown"
    else:
        pred_class = class_mapping.get(pred_idx, f"class_{pred_idx}")
    
    all_probs = {class_mapping.get(i, f"class_{i}"): probs[0][i].item() 
                 for i in range(len(class_mapping))}
    
    result = {
        "class": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in all_probs.items()},
        "fps": fps,
        "frame_obj": frames[-1] if frames else None
    }
    
    inner_logger.info(f"VideoMAE 预测结果: {result}")
    return result


def download_videos(url):
    """下载视频文件"""
    current_thread = threading.current_thread().ident
    save_path = 'video'
    filename = 'temp_' + str(current_thread) + ".mp4"
    video_path = save_path + '/' + filename

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    try:
        res = requests.get(url, stream=True, verify=False, timeout=30)
        res_result = res.status_code
    except Exception as e:
        inner_logger.error(f"下载视频请求失败: {e}")
        return None
    
    if res_result != 200:
        inner_logger.error(f"下载视频失败, 状态码: {res_result}")
        return None
    
    try:
        with open(video_path, 'wb') as f1:
            t1 = time.time()
            past_time = 1
            for chunk in res.iter_content(chunk_size=1024*4):
                t = int(time.time() - t1)
                if int(t) % 5 == 0 and past_time != t:
                    past_time = t
                    inner_logger.info("下载中:" + str(t) + ' s')
                f1.write(chunk)
        inner_logger.info(f"下载视频成功, 路径为: {video_path}")
        return video_path
    except Exception as e:
        inner_logger.error(f"保存视频失败: {e}")
        return None


def detect_videomae_stream(video_path, threshold=0.7, callback=None):
    """
    VideoMAE 流式检测 - 持续从RTSP流读取帧并预测行为
    
    Args:
        video_path: RTSP流地址
        threshold: 置信度阈值
        callback: 回调函数 callback(result)
    
    Returns:
        dict: 最终检测结果
    """
    model, processor, class_mapping = load_videomae_model()
    if model is None:
        inner_logger.error("VideoMAE 模型未加载")
        return None
    
    config = load_config()
    videomae_config = config.get('videomae', {})
    num_frames = videomae_config.get('num_frames', 16)
    device_config = videomae_config.get('device', 'auto')
    
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    inner_logger.info(f"VideoMAE 开始流式检测: {video_path}, device={device}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        inner_logger.error(f"无法打开视频流: {video_path}")
        return None
    
    # RTSP流参数
    sample_interval = 2.0  # 每2秒采样一次
    frame_buffer = []
    last_sample_time = time.time()
    last_alarm_time = 0  # 上次告警时间
    alarm_cooldown = 5.0  # 告警冷却时间（秒）
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    result = {
        'has_alarm': False,
        'behavior': None,
        'all_predictions': []
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp_url(video_path):
                inner_logger.warning("VideoMAE: RTSP流断开，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(video_path)
                continue
            else:
                break
        
        frame_count += 1
        current_time = time.time()
        
        # 每隔一定帧数保存一帧到缓冲区
        if frame_count % max(1, int(fps / 5)) == 0:  # 每秒5帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(Image.fromarray(frame_rgb))
            if len(frame_buffer) > num_frames:
                frame_buffer.pop(0)
        
        # 检查是否到达采样间隔
        if current_time - last_sample_time >= sample_interval and len(frame_buffer) >= num_frames:
            last_sample_time = current_time
            
            # 使用缓冲帧进行预测
            try:
                if processor:
                    inputs = processor(frame_buffer.copy(), return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    processed_frames = [transform(f) for f in frame_buffer.copy()]
                    pixel_values = torch.stack(processed_frames, dim=1).unsqueeze(0).to(device)
                    inputs = {'pixel_values': pixel_values}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(-1).item()
                confidence = probs[0][pred_idx].item()
                
                pred_class = class_mapping.get(pred_idx, f"class_{pred_idx}")
                
                all_probs = {class_mapping.get(i, f"class_{i}"): probs[0][i].item() 
                             for i in range(len(class_mapping))}
                
                prediction = {
                    "class": pred_class,
                    "confidence": round(confidence, 4),
                    "probabilities": {k: round(v, 4) for k, v in all_probs.items()},
                    "frame": frame_count,
                    "timestamp": current_time
                }
                
                inner_logger.info(f"VideoMAE 预测: {pred_class} ({confidence:.4f})")
                result['all_predictions'].append(prediction)
                result['behavior'] = prediction
                
                # 如果检测到告警行为（climb或fight）且冷却时间已过
                if pred_class in ['climb', 'fight'] and confidence >= threshold:
                    # 检查冷却时间
                    if current_time - last_alarm_time >= alarm_cooldown:
                        result['has_alarm'] = True
                        last_alarm_time = current_time
                        
                        # 获取告警帧
                        alarm_frame_obj = None
                        if frame_buffer:
                            last_frame = np.array(frame_buffer[-1])
                            if last_frame is None:
                                inner_logger.warning("VideoMAE: 告警帧为空，无法保存告警图片")
                            alarm_frame_obj = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
                        
                        if callback:
                            callback({
                                'type': 'videomae_alarm',
                                'has_alarm': True,
                                'behavior': prediction,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'frame_obj': alarm_frame_obj
                            })
                    else:
                        inner_logger.debug(f"VideoMAE 告警冷却中，跳过本次告警保存")
                
                # 清空缓冲区，只保留最后几帧用于下次预测
                frame_buffer = frame_buffer[num_frames//2:]
                
            except Exception as e:
                inner_logger.error(f"VideoMAE 预测异常: {e}")
    
    cap.release()
    inner_logger.info(f"VideoMAE 流式检测完成: {len(result['all_predictions'])} 次预测")
    return result


def detect_videomae(video_url):
    """使用 VideoMAE 检测视频行为"""
    if video_url.lower().startswith('rtsp://'):
        video_type = "rtsp"
    elif video_url.lower().endswith('.mp4'):
        video_type = "mp4"
    else:
        video_type = "unknown"
    
    inner_logger.info(f"VideoMAE 检测 - 视频类型: {video_type}, URL: {video_url}")
    
    if video_url.startswith(('http://', 'https://')):
        video_path = download_videos(video_url)
        if video_path is None:
            return video_type, None
    else:
        video_path = video_url
    
    # RTSP流跳过文件存在检查，直接通过cv2.VideoCapture读取
    if not video_path.startswith('rtsp://'):
        if not os.path.exists(video_path):
            inner_logger.error(f"视频文件不存在: {video_path}")
            return video_type, None
    
    config = load_config()
    videomae_config = config.get('videomae', {})
    num_frames = videomae_config.get('num_frames', 16)
    threshold = videomae_config.get('threshold', 0.7)
    
    result = predict_videomae(video_path, num_frames=num_frames, threshold=threshold)
    
    if result is None:
        return video_type, None
    
    return video_type, result


def is_rtsp_url(url):
    """判断是否为RTSP视频流"""
    return url.lower().startswith('rtsp://')


def detect_video_yolo_stream(video_path, yolo_model_path, scenario_ids, callback=None, detect_areas=None):
    """
    YOLO 场景检测 - 流式处理，不退出，实时输出告警
    
    Args:
        video_path: 视频路径或RTSP流地址
        yolo_model_path: YOLO模型路径
        scenario_ids: 场景检测ID列表
        callback: 回调函数，用于实时输出告警信息 callback(frame_info)
        detect_areas: 检测区域列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]，为空则全域检测
    
    Returns:
        dict: 最终检测结果
    """
    from scenarios import run_scenario_checks, get_alarm_summary, check_det_in_area
    from boxmot.trackers.tracker_zoo import create_tracker
    
    config = load_config()
    tracker_config = config.get('tracker', {})
    
    yolo_model = YOLO(yolo_model_path)
    
    # 初始化跟踪器
    tracker = create_tracker('ocsort', evolve_param_dict=tracker_config)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        inner_logger.error(f"无法打开视频: {video_path}")
        return None
    
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_rtsp = is_rtsp_url(video_path)
    
    skip_frame = 3
    frame_number = 0
    
    # 滑动窗口用于时序检测
    alarm_history = []
    window_size = 20
    alarm_rate_thr = 0.5
    
    # 告警冷却时间
    last_alarm_time = 0
    alarm_cooldown = 5.0  # 告警冷却时间（秒）
    
    last_annotated_img = None
    total_detections = []
    all_frame_alarms = []  # 存储所有帧的告警信息
    all_alarm_scenarios = {}  # 记录所有告警帧的场景统计
    
    inner_logger.info(f"开始 YOLO 流式检测: is_rtsp={is_rtsp}, frames={video_frame_count}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp:
                # RTSP流断开重连
                inner_logger.warning("RTSP流断开，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(video_path)
                continue
            else:
                break
        
        if frame_number % skip_frame == 0:
            results = yolo_model.predict(frame, device="0")
            
            # 获取检测结果
            det = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    det.append([
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                        float(confs[i]), int(classes[i])
                    ])
            
            det = np.array(det) if len(det) > 0 else np.empty((0, 6))
            
            # 跟踪
            tracks = tracker.update(det, frame)
            
            # 将跟踪结果转换为场景检测格式
            # tracks格式: (N, 8) [x1,y1,x2,y2,track_id,conf,cls,track_length]
            all_dets = []
            track_info = {}  # {track_id: track_length}
            
            if len(tracks) > 0:
                for track in tracks:
                    if len(track) >= 8:
                        x1, y1, x2, y2 = track[:4]
                        track_id = int(track[4])
                        conf = float(track[5])
                        cls = int(track[6])
                        track_length = int(track[7])
                        
                        all_dets.append([x1, y1, x2, y2, conf, cls])
                        track_info[track_id] = {
                            'track_length': track_length,
                            'bbox': [x1, y1, x2, y2],
                            'class': cls
                        }
            
            # 记录帧信息
            frame_has_alarm = False
            frame_scenarios = {}
            if all_dets:
                # 检测区域过滤
                if detect_areas:
                    all_dets, error_msg = check_det_in_area(all_dets, detect_areas)
                    if error_msg:
                        inner_logger.debug(f"检测区域过滤: {error_msg}")
                
                # 过滤后填充total_detections
                for det in all_dets:
                    x1, y1, x2, y2, conf, cls = det
                    total_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': yolo_model.names[cls],
                        'track_id': None,  # 区域过滤后track_id可能已不匹配
                        'track_length': None
                    })
                
                scenario_results = run_scenario_checks(
                    all_dets, 
                    scenario_ids, 
                    track_info=track_info,
                    frame_id=frame_number,
                    tracks=tracks
                )
                alarm_summary = get_alarm_summary(scenario_results)
                frame_scenarios = alarm_summary
                if alarm_summary:
                    frame_has_alarm = True
            
            # 记录当前帧告警信息
            frame_info = {
                'frame': frame_number,
                'timestamp': time.time(),
                'has_alarm': frame_has_alarm,
                'scenarios': frame_scenarios,
                'detection_count': len(all_dets),
                'track_info': track_info,
                'frame_obj': frame.copy() if frame_has_alarm else None,
                'detections': all_dets.copy() if frame_has_alarm else [],
                'alarm_counts': scenario_results if frame_has_alarm else {}
            }
            all_frame_alarms.append(frame_info)
            
            # 如果有告警，更新全局场景统计
            if frame_has_alarm:
                for name, count in frame_scenarios.items():
                    all_alarm_scenarios[name] = all_alarm_scenarios.get(name, 0) + count
            
            alarm_history.append({
                'frame': frame_number,
                'has_alarm': frame_has_alarm,
                'scenarios': frame_scenarios
            })
            
            if len(alarm_history) > window_size:
                alarm_history.pop(0)
            
            # 有告警时才回调
            if frame_has_alarm and callback:
                current_time = time.time()
                # 检查冷却时间
                if current_time - last_alarm_time >= alarm_cooldown:
                    last_alarm_time = current_time
                    
                    # 绘制告警帧（检测框 + 统计面板）
                    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource', 'logo.png')
                    if not os.path.exists(logo_path):
                        logo_path = None
                    alarm_img = draw_alarm_frame(
                        frame, 
                        all_dets, 
                        all_dets,  # 所有检测框都作为告警框
                        frame_scenarios, 
                        scenario_results,
                        logo_path=logo_path,
                        scenario_ids=scenario_ids
                    )
                    _, buffer = cv2.imencode('.jpg', alarm_img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    frame_info['image_base64'] = img_base64
                    
                    # 记录告警日志
                    inner_logger.warning(f"[YOLO告警] 帧={frame_number}, 场景={frame_scenarios}, 跟踪ID={list(track_info.keys())}")
                    
                    callback(frame_info)
                else:
                    inner_logger.debug(f"YOLO 告警冷却中，跳过本次告警保存")
            
            last_annotated_img = results[0].plot()
        
        frame_number += 1
        
        # 实时流：每100帧输出一次状态
        if is_rtsp and frame_number % 100 == 0:
            inner_logger.info(f"RTSP流检测中: frame={frame_number}")
    
    cap.release()
    
    # 计算最终结果
    has_alarm = False
    
    if len(alarm_history) > 0:
        alarm_count = sum(1 for h in alarm_history if h['has_alarm'])
        alarm_rate = alarm_count / len(alarm_history)
        
        inner_logger.info(f"YOLO 检测完成: 告警帧数={alarm_count}/{len(alarm_history)}, 告警率={alarm_rate:.2%}")
        
        if alarm_rate >= alarm_rate_thr:
            has_alarm = True
    
    annotated_base64 = None
    if last_annotated_img is not None:
        _, buffer = cv2.imencode('.jpg', last_annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'has_alarm': has_alarm,
        'scenarios': all_alarm_scenarios,
        'detections': total_detections,
        'image_view_base64': annotated_base64,
        'video_frame_count': frame_number,
        'all_frame_alarms': all_frame_alarms
    }


def detect_video_combined_stream(video_path, yolo_model_path, yolo_scenario_ids, videomae_scenario_ids, callback=None, detect_areas=None):
    """
    并行流式检测 - 实时输出告警，融合YOLO和VideoMAE告警
    
    Args:
        video_path: 视频路径或RTSP流地址
        yolo_model_path: YOLO模型路径
        yolo_scenario_ids: YOLO场景检测ID列表
        videomae_scenario_ids: VideoMAE行为识别ID列表
        callback: 回调函数 callback(frame_info)
        detect_areas: 检测区域列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]，为空则全域检测
    
    Returns:
        dict: 综合检测结果
    """
    result = {
        'has_alarm': False,
        'yolo_scenarios': {},
        'behavior': None,
        'image_view_base64': None,
        'detections_count': 0,
        'all_frame_alarms': []
    }
    
    result_lock = threading.Lock()
    
    # 告警状态（用于监控线程融合）
    alarm_state = {
        'yolo_has_alarm': False,
        'yolo_scenarios': {},
        'yolo_frame_obj': None,
        'yolo_frame_num': 0,
        'yolo_detections': [],
        'yolo_alarm_counts': {},
        'videomae_has_alarm': False,
        'videomae_behavior': None,
        'videomae_frame_num': 0,
        'videomae_frame_obj': None,
        'last_save_time': 0  # 上次保存时间
    }
    state_lock = threading.Lock()
    
    # 告警图片保存目录
    alert_img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alert_images')
    os.makedirs(alert_img_dir, exist_ok=True)
    
    def save_alarm_image(frame_obj, scenarios, behavior, frame_num, detections=None, alarm_counts=None):
        """保存告警图片并记录日志"""
        try:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')
            
            # 构建告警信息
            alarm_info = []
            if scenarios:
                for name, count in scenarios.items():
                    alarm_info.append(f"{name}x{count}")
            if behavior:
                alarm_info.append(f"{behavior.get('class', 'unknown')}({behavior.get('confidence', 0):.2f})")
            
            alarm_desc = "+".join(alarm_info) if alarm_info else "unknown"
            img_filename = f"alarm_{timestamp_str}_f{frame_num}_{alarm_desc}.jpg"
            img_path = os.path.join(alert_img_dir, img_filename)
            
            if frame_obj is not None:
                # 绘制告警框和统计面板
                logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource', 'logo.png')
                if not os.path.exists(logo_path):
                    logo_path = None
                
                # 如果有VideoMAE行为告警，添加到alarm_counts
                merged_alarm_counts = dict(alarm_counts) if alarm_counts else {}
                if behavior:
                    behavior_class = behavior.get('class', '')
                    # 映射行为类别到场景ID
                    behavior_to_id = {'climb': 10, 'fight': 11}
                    if behavior_class in behavior_to_id:
                        merged_alarm_counts[behavior_to_id[behavior_class]] = 1
                
                # 获取场景ID列表
                all_scenario_ids = list(merged_alarm_counts.keys()) if merged_alarm_counts else None
                
                frame_obj = draw_alarm_frame(
                    frame_obj, 
                    detections if detections else [], 
                    detections if detections else [],  # 所有检测框都作为告警框
                    scenarios, 
                    merged_alarm_counts,
                    logo_path=logo_path,
                    scenario_ids=all_scenario_ids
                )
                cv2.imwrite(img_path, frame_obj)
                inner_logger.info(f"告警图片已保存: {img_path}, frame_obj shape={frame_obj.shape if hasattr(frame_obj, 'shape') else 'N/A'}")
            else:
                inner_logger.warning(f"告警图片保存失败: frame_obj is None")
            
            # 记录告警日志
            log_msg = f"[告警] 帧={frame_num}, 场景={scenarios}, 行为={behavior.get('class') if behavior else 'None'}, 图片={img_path}"
            inner_logger.warning(log_msg)
            
            return img_path, True
            
        except Exception as e:
            inner_logger.error(f"保存告警图片失败: {e}")
            return None, False
    
    def alarm_monitor():
        """告警监控线程 - 每5秒检查一次，融合保存告警"""
        nonlocal alarm_state
        check_interval = 5.0  # 检查间隔（秒）
        
        while True:
            time.sleep(check_interval)
            
            with state_lock:
                current_time = time.time()
                
                # 检查是否有告警
                has_yolo_alarm = alarm_state['yolo_has_alarm']
                has_videomae_alarm = alarm_state['videomae_has_alarm']
                
                if has_yolo_alarm or has_videomae_alarm:
                    # 检查冷却时间
                    if current_time - alarm_state['last_save_time'] < check_interval:
                        continue
                    
                    # 融合告警信息
                    scenarios = alarm_state['yolo_scenarios'] if has_yolo_alarm else {}
                    behavior = alarm_state['videomae_behavior'] if has_videomae_alarm else None
                    frame_obj = alarm_state['yolo_frame_obj'] if has_yolo_alarm else alarm_state['videomae_frame_obj']
                    frame_num = alarm_state['yolo_frame_num'] if has_yolo_alarm else alarm_state['videomae_frame_num']
                    detections = alarm_state['yolo_detections'] if has_yolo_alarm else []
                    alarm_counts = alarm_state['yolo_alarm_counts'] if has_yolo_alarm else {}
                    
                    inner_logger.info(f"监控线程: 告警检测 - has_yolo={has_yolo_alarm}, has_videomae={has_videomae_alarm}, frame_obj={frame_obj is not None}")
                    
                    # 保存告警图片
                    save_alarm_image(frame_obj, scenarios, behavior, frame_num, detections, alarm_counts)
                    
                    # 更新保存时间
                    alarm_state['last_save_time'] = current_time
                    
                    # 重置告警状态
                    alarm_state['yolo_has_alarm'] = False
                    alarm_state['yolo_scenarios'] = {}
                    alarm_state['yolo_frame_obj'] = None
                    alarm_state['yolo_detections'] = []
                    alarm_state['yolo_alarm_counts'] = {}
                    alarm_state['videomae_has_alarm'] = False
                    alarm_state['videomae_behavior'] = None
                    alarm_state['videomae_frame_obj'] = None
                    
                    inner_logger.info(f"监控线程: 告警已保存, scenarios={scenarios}, behavior={behavior.get('class') if behavior else 'None'}")
    
    def yolo_worker():
        """YOLO 场景检测线程"""
        nonlocal alarm_state
        try:
            def yolo_callback(frame_info):
                """YOLO帧回调 - 只在有告警时调用"""
                if not frame_info.get('has_alarm', False):
                    return
                
                with result_lock:
                    result['all_frame_alarms'].append(frame_info)
                    result['has_alarm'] = True
                    # 更新场景统计
                    for name, count in frame_info['scenarios'].items():
                        result['yolo_scenarios'][name] = result['yolo_scenarios'].get(name, 0) + count
                
                # 更新告警状态
                with state_lock:
                    alarm_state['yolo_has_alarm'] = True
                    alarm_state['yolo_scenarios'] = frame_info.get('scenarios', {})
                    alarm_state['yolo_frame_obj'] = frame_info.get('frame_obj')
                    alarm_state['yolo_frame_num'] = frame_info.get('frame', 0)
                    alarm_state['yolo_detections'] = frame_info.get('detections', [])
                    alarm_state['yolo_alarm_counts'] = frame_info.get('alarm_counts', {})
                
                # 传递给外部回调
                if callback:
                    callback(frame_info)
            
            yolo_result = detect_video_yolo_stream(
                video_path, yolo_model_path, yolo_scenario_ids, callback=yolo_callback, detect_areas=detect_areas
            )
            if yolo_result:
                with result_lock:
                    result['yolo_scenarios'] = yolo_result['scenarios']
                    result['detections_count'] = len(yolo_result['detections'])
                    if yolo_result['has_alarm']:
                        result['has_alarm'] = True
                    if yolo_result['image_view_base64']:
                        result['image_view_base64'] = yolo_result['image_view_base64']
            inner_logger.info(f"YOLO 线程完成: scenarios={result['yolo_scenarios']}")
        except Exception as e:
            inner_logger.error(f"YOLO 线程异常: {e}")
    
    def videomae_worker():
        """VideoMAE 行为识别线程 - 支持流式检测"""
        nonlocal alarm_state
        try:
            config = load_config()
            videomae_config = config.get('videomae', {})
            threshold = videomae_config.get('threshold', 0.7)
            
            def videomae_callback(info):
                """VideoMAE回调 - 更新告警状态"""
                if not info.get('has_alarm', False):
                    return
                
                # 更新告警状态
                with state_lock:
                    alarm_state['videomae_has_alarm'] = True
                    alarm_state['videomae_behavior'] = info.get('behavior')
                    alarm_state['videomae_frame_num'] = info.get('frame', 0)
                    alarm_state['videomae_frame_obj'] = info.get('frame_obj')
                
                # 传递给外部回调
                if callback:
                    callback(info)
            
            if is_rtsp_url(video_path):
                # RTSP流：持续流式检测
                videomae_result = detect_videomae_stream(video_path, threshold=threshold, callback=videomae_callback)
                if videomae_result:
                    with result_lock:
                        result['behavior'] = videomae_result.get('behavior')
                        if videomae_result.get('has_alarm'):
                            result['has_alarm'] = True
                inner_logger.info(f"VideoMAE 线程完成 (流式): behavior={result['behavior']}")
            else:
                # MP4文件：一次性检测
                video_type, behavior_result = detect_videomae(video_path)
                if behavior_result:
                    with result_lock:
                        result['behavior'] = {
                            'class': behavior_result['class'],
                            'confidence': behavior_result['confidence'],
                            'probabilities': behavior_result['probabilities']
                        }
                        if behavior_result['class'] in ['climb', 'fight']:
                            result['has_alarm'] = True
                            # 获取告警帧 (PIL Image -> numpy BGR)
                            videomae_frame_obj = None
                            if behavior_result.get('frame_obj') is not None:
                                frame_pil = behavior_result['frame_obj']
                                if hasattr(frame_pil, 'convert'):
                                    # PIL Image
                                    videomae_frame_obj = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                                else:
                                    videomae_frame_obj = frame_pil
                            # 更新告警状态
                            with state_lock:
                                alarm_state['videomae_has_alarm'] = True
                                alarm_state['videomae_behavior'] = result['behavior']
                                alarm_state['videomae_frame_num'] = 0
                                alarm_state['videomae_frame_obj'] = videomae_frame_obj
                            if callback:
                                callback({
                                    'type': 'videomae_alarm',
                                    'has_alarm': True,
                                    'behavior': result['behavior'],
                                    'frame_obj': videomae_frame_obj
                                })
                inner_logger.info(f"VideoMAE 线程完成 (文件): behavior={result['behavior']}")
        except Exception as e:
            inner_logger.error(f"VideoMAE 线程异常: {e}")
    
    # 启动告警监控线程
    monitor_thread = threading.Thread(target=alarm_monitor, name="AlarmMonitor-Thread", daemon=True)
    monitor_thread.start()
    
    # 启动并行检测线程
    threads = []
    
    if yolo_scenario_ids:
        t1 = threading.Thread(target=yolo_worker, name="YOLO-Thread", daemon=True)
        threads.append(t1)
        t1.start()
    
    if videomae_scenario_ids:
        t2 = threading.Thread(target=videomae_worker, name="VideoMAE-Thread", daemon=True)
        threads.append(t2)
        t2.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    inner_logger.info(f"并行流式检测完成: has_alarm={result['has_alarm']}")
    return result
