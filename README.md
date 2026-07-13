# 行为分析服务 (Behavior Analysis)

基于 YOLO + VideoMAE 的智能行为分析检测服务，支持实时视频流和视频文件的行为识别与告警。

## 功能特性

- **YOLO 场景检测** (ID 0-8): 人员入侵、安全帽、工装、打电话、抽烟、跌倒、安全带、睡岗、移动打电话
- **VideoMAE 行为识别** (ID 10-11): 攀爬、打架
- **并行检测**: YOLO 和 VideoMAE 并行运行，告警融合
- **实时告警**: 支持 RTSP 流实时检测和告警推送 (SSE)
- **目标跟踪**: OC-Sort 跟踪器，支持跨帧目标关联
- **区域检测**: 支持自定义检测区域（多边形）
- **告警图片**: 自动保存告警帧图片，包含检测框和统计面板

## 项目结构

```
behavior_analysis/
├── config/
│   └── config.json          # 配置文件
├── detect/
│   ├── app.py               # Flask API 服务
│   ├── detect_behavior.py   # 核心检测逻辑
│   ├── scenarios.py         # 场景检测函数
│   ├── drawing_utils.py     # 绘图工具（检测框、统计面板）
│   ├── logger.py            # 日志配置
│   └── rules/
│       └── phone_tracking_manager.py  # 移动打电话检测
├── models/
│   ├── yolo/                # YOLO 模型
│   └── videomae/            # VideoMAE 模型
├── resource/
│   ├── ai.ttf               # 字体文件
│   ├── logo.png             # Logo 图片
│   └── logo_en.png          # 英文 Logo
├── alert_images/            # 告警图片输出目录
├── logs/                    # 日志目录
├── requirements.txt         # 依赖列表
└── start_service.sh         # 启动脚本
```

## 环境要求

- Python 3.10+
- Conda 环境: `videomae_fune`

## 依赖

```
flask
ultralytics
opencv-python-headless
torch
torchvision
transformers
pillow
numpy
requests
boxmot
```

## 安装

```bash
# 激活 conda 环境
conda activate videomae_fune

# 安装依赖
pip install -r requirements.txt
```

## 启动服务

```bash
# 方式1: 直接启动
cd detect
python app.py

# 方式2: 使用启动脚本
bash start_service.sh
```

服务默认运行在 `http://0.0.0.0:10235`

## API 接口

### POST `/AI/behavior_analysis/v1/detect`

统一检测接口（仅支持视频输入）

**请求格式:**
```json
{
    "video": "path/rtsp://...",
    "scenario_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
    "detect_areas": [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
}
```

**参数说明:**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video | string | 是 | 视频路径或 RTSP 流地址 |
| scenario_ids | list | 否 | 场景 ID 列表，默认全部 |
| detect_areas | list | 否 | 检测区域多边形顶点坐标 |

**场景 ID 说明:**
| ID | 场景 | 检测方式 |
|----|------|----------|
| 0 | 人员入侵 | YOLO |
| 1 | 未戴安全帽 | YOLO |
| 2 | 未穿工装 | YOLO |
| 3 | 打电话 | YOLO |
| 4 | 抽烟 | YOLO |
| 5 | 跌倒 | YOLO |
| 6 | 未系安全带 | YOLO |
| 7 | 睡岗 | YOLO |
| 8 | 移动打电话 | YOLO |
| 10 | 攀爬 | VideoMAE |
| 11 | 打架 | VideoMAE |

**响应格式:**
```json
{
    "cost": 1.23,
    "statusCode": 0,
    "statusMsg": "success",
    "result": {
        "has_alarm": true,
        "scenarios": {"未戴安全帽": 1},
        "behavior": {"class": "climb", "confidence": 0.95},
        "detections_count": 5,
        "image_view_base64": "..."
    }
}
```

### POST `/AI/behavior_analysis/v1/detect_stream`

SSE 流式检测接口，实时推送告警

### GET `/AI/behavior_analysis/v1/scenarios`

获取支持的场景列表

## 配置说明

`config/config.json`:
```json
{
    "yolo_model": "./models/yolo/sevnce_cls13_1280_20260604.pt",
    "videomae_model": "./models/videomae",
    "videomae": {
        "num_frames": 16,
        "threshold": 0.7,
        "device": "auto"
    },
    "tracker": {
        "min_conf": 0.3,
        "delta_t": 3,
        "inertia": 0.1,
        "min_hits": 1
    }
}
```

## 告警图片

告警图片保存在 `alert_images/` 目录，包含:
- 检测框（绿色普通框，红色告警框）
- 左上角统计面板（显示各类告警计数）

文件名格式: `alarm_{时间}_f{帧号}_{告警描述}.jpg`

## Docker 部署

```bash
docker build -t behavior_analysis .
docker run -p 10235:10235 behavior_analysis
```
