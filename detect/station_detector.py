"""
岗位区域检测器
基于跟踪ID的时间稳定性检测：人员在原地停留一段时间后，自动生成岗位区域
"""
import time
import math
from collections import deque


class StationDetector:
    """岗位区域检测器"""
    
    def __init__(self, min_stay_duration=5000, position_threshold=30.0, 
                 min_history_size=50, region_expand_ratio=0.2):
        """
        初始化岗位检测器
        
        Args:
            min_stay_duration: 停留时长阈值（毫秒）
            position_threshold: 位置稳定性阈值（像素）
            min_history_size: 最小历史帧数
            region_expand_ratio: 区域扩展比例
        """
        self.min_stay_duration = min_stay_duration
        self.position_threshold = position_threshold
        self.min_history_size = min_history_size
        self.region_expand_ratio = region_expand_ratio
        
        # 人员轨迹: {track_id: PersonTrack}
        self.tracks = {}
        
        # 已识别的岗位: {track_id: StationRegion}
        self.stations = {}
    
    def get_stable_point(self, bbox):
        """
        计算稳定点（底部中心）
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            (x, y) 稳定点坐标
        """
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2
        bottom_y = y2 * 0.85  # 偏底部，模拟脚的位置
        return (center_x, bottom_y)
    
    def is_position_stable(self, current, last):
        """
        判断位置是否稳定
        
        Args:
            current: 当前稳定点 (x, y)
            last: 上一帧稳定点 (x, y)
        
        Returns:
            bool: 位置变化是否小于阈值
        """
        dx = current[0] - last[0]
        dy = current[1] - last[1]
        return (dx * dx + dy * dy) < (self.position_threshold * self.position_threshold)
    
    def calculate_region(self, history):
        """
        计算岗位区域（最小外接矩形 + 扩展）
        
        Args:
            history: 历史检测框列表 [[x1,y1,x2,y2], ...]
        
        Returns:
            [x1, y1, x2, y2] 岗位区域
        """
        if not history:
            return None
        
        min_x = min(h[0] for h in history)
        min_y = min(h[1] for h in history)
        max_x = max(h[2] for h in history)
        max_y = max(h[3] for h in history)
        
        # 向外扩展
        width = max_x - min_x
        height = max_y - min_y
        expand_x = width * self.region_expand_ratio
        expand_y = height * self.region_expand_ratio
        
        return [
            min_x - expand_x,
            min_y - expand_y,
            max_x + expand_x,
            max_y + expand_y
        ]
    
    def update(self, track_id, bbox, timestamp=None):
        """
        更新人员轨迹
        
        Args:
            track_id: 跟踪ID
            bbox: [x1, y1, x2, y2]
            timestamp: 时间戳（毫秒），None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time() * 1000
        
        stable_point = self.get_stable_point(bbox)
        
        # 初始化或获取轨迹
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'id': track_id,
                'history': deque(maxlen=1000),
                'stay_start_time': 0,
                'last_position': None,
                'has_station': track_id in self.stations,
                'station': self.stations.get(track_id)
            }
        
        track = self.tracks[track_id]
        
        # 如果已有岗位，只记录历史，不学习
        if track['has_station']:
            track['history'].append(bbox)
            return
        
        # 判断位置稳定性
        if track['last_position'] is not None:
            stable = self.is_position_stable(stable_point, track['last_position'])
        else:
            stable = False
            track['stay_start_time'] = timestamp
        
        if stable:
            # 稳定：开始或继续计时
            if track['stay_start_time'] == 0:
                track['stay_start_time'] = timestamp
            
            stay_duration = timestamp - track['stay_start_time']
            
            # 检查是否满足生成岗位的条件
            if (stay_duration >= self.min_stay_duration and 
                len(track['history']) >= self.min_history_size):
                # 生成岗位区域
                region = self.calculate_region(list(track['history']))
                if region:
                    station = {
                        'track_id': track_id,
                        'bbox': region,
                        'center': ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2),
                        'created_time': timestamp,
                        'is_valid': True
                    }
                    track['has_station'] = True
                    track['station'] = station
                    self.stations[track_id] = station
        else:
            # 不稳定：重置计时器
            track['stay_start_time'] = timestamp
        
        # 记录历史
        track['history'].append(bbox)
        track['last_position'] = stable_point
        
        # 清理过期轨迹
        self._cleanup_tracks(timestamp)
    
    def _cleanup_tracks(self, current_time, expire_time=10000):
        """清理过期轨迹"""
        expired_ids = []
        for track_id, track in self.tracks.items():
            if track['has_station']:
                continue
            # 检查最后一帧的时间（通过历史大小估算）
            if len(track['history']) > 0:
                # 简单清理：如果轨迹没有更新且没有岗位，标记过期
                pass
    
    def get_station_region(self, track_id):
        """获取指定人员的岗位区域"""
        return self.stations.get(track_id)
    
    def has_station_region(self, track_id):
        """检查是否已有岗位"""
        return track_id in self.stations
    
    def get_all_stations(self, person_detections, tracks=None, frame_id=0):
        """
        主入口：获取所有已识别的岗位区域
        
        Args:
            person_detections: 人员检测框列表 [[x1,y1,x2,y2,conf,cls], ...]
            tracks: 原始跟踪结果 [x1,y1,x2,y2,track_id,conf,cls,track_length]
            frame_id: 当前帧ID
        
        Returns:
            list: 岗位区域列表 [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'center': (x,y)}]
        """
        current_time = time.time() * 1000
        
        # 构建 track_id -> bbox 的映射
        track_map = {}
        if tracks is not None:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[:4]
                    tid = int(track[4])
                    cls = int(track[6])
                    if cls == 0:  # 只处理人员
                        track_map[tid] = [x1, y1, x2, y2]
        
        # 更新轨迹
        for det in person_detections:
            if len(det) >= 6:
                x1, y1, x2, y2 = det[:4]
                # 尝试从 tracks 中获取 track_id
                tid = self._find_track_id(det, track_map)
                if tid is not None:
                    self.update(tid, [x1, y1, x2, y2], current_time)
        
        # 返回所有已识别的岗位
        return list(self.stations.values())
    
    def _find_track_id(self, det, track_map):
        """通过检测框位置匹配 track_id"""
        x1, y1, x2, y2 = det[:4]
        det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        best_tid = None
        best_iou = 0
        
        for tid, bbox in track_map.items():
            # 计算 IoU
            iou = self._calculate_iou(det[:4], bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        
        return best_tid if best_iou > 0.3 else None
    
    def _calculate_iou(self, box1, box2):
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def reset_station(self, track_id):
        """重置指定岗位"""
        if track_id in self.stations:
            del self.stations[track_id]
        if track_id in self.tracks:
            self.tracks[track_id]['has_station'] = False
            self.tracks[track_id]['station'] = None
    
    def clear_all_stations(self):
        """清除所有岗位"""
        self.stations.clear()
        self.tracks.clear()


# 全局实例
_station_detector = None


def get_station_detector(config=None):
    """获取全局岗位检测器实例"""
    global _station_detector
    if _station_detector is None:
        if config:
            station_config = config.get('station_detector', {})
            _station_detector = StationDetector(
                min_stay_duration=station_config.get('min_stay_duration', 5000),
                position_threshold=station_config.get('position_threshold', 30.0),
                min_history_size=station_config.get('min_history_size', 50),
                region_expand_ratio=station_config.get('region_expand_ratio', 0.2)
            )
        else:
            _station_detector = StationDetector()
    return _station_detector
