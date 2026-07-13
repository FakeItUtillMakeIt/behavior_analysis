import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from logger import inner_logger as logger

class PhoneTrackingManager:
    """管理打电话跟踪状态，用于判断移动中打电话（基于滑动窗口平均速度）"""
    
    def __init__(self):
        self.track_moving_state = {}
        self.moving_speed_threshold = 5    # 移动速度阈值（像素/帧）
        self.static_speed_threshold = 3     # 静止速度阈值（像素/帧）
        self.window_size = 5                # 滑动窗口大小（计算平均速度的帧数）
        self.history_length = 30            # 保留最近N帧的历史
        self.required_moving_frames = 3     # 需要连续多少帧平均速度超过阈值才标记为移动
        
    def update_track(self, track_id, person_box, frame_id):
        """
        基于滑动窗口平均速度判断移动状态
        """
        if track_id is None:
            return False
        
        # 计算人体中心点
        center_x = (person_box[0] + person_box[2]) / 2
        center_y = (person_box[1] + person_box[3]) / 2
        
        # 初始化跟踪记录
        if track_id not in self.track_moving_state:
            self.track_moving_state[track_id] = {
                'history': [],           # 位置历史
                'speeds': [],            # 速度历史
                'is_moving': False,
                'moving_frames': 0,      # 连续移动帧计数
                'static_frames': 0       # 连续静止帧计数
            }
        
        state = self.track_moving_state[track_id]
        
        # 添加当前位置到历史
        state['history'].append((center_x, center_y, frame_id))
        if len(state['history']) > self.history_length:
            state['history'].pop(0)
        
        # 计算当前帧速度（与前一帧比较）
        current_speed = 0
        if len(state['history']) >= 2:
            prev = state['history'][-2]
            curr = state['history'][-1]
            frame_diff = curr[2] - prev[2]
            if frame_diff > 0:
                displacement = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
                current_speed = displacement / frame_diff
        
        # 添加到速度历史
        state['speeds'].append(current_speed)
        if len(state['speeds']) > self.window_size:
            state['speeds'].pop(0)
        
        # 计算窗口内的平均速度
        if len(state['speeds']) >= self.window_size // 2:  # 至少有一半的数据
            avg_speed = sum(state['speeds']) / len(state['speeds'])
        else:
            avg_speed = current_speed
        
        # 判断当前是否移动
        is_currently_moving = avg_speed > self.moving_speed_threshold
        
        # 更新连续计数
        if is_currently_moving:
            state['moving_frames'] += 1
            state['static_frames'] = 0
        else:
            state['moving_frames'] = 0
            state['static_frames'] += 1
        
        # 根据连续帧数判断状态
        was_moving = state['is_moving']
        
        if state['moving_frames'] >= self.required_moving_frames:
            if not state['is_moving']:
                logger.info(f"Track {track_id}: 进入移动状态 (连续移动 {state['moving_frames']} 帧, "
                          f"平均速度={avg_speed:.2f})")
            state['is_moving'] = True
        elif state['static_frames'] >= self.required_moving_frames:
            if state['is_moving']:
                logger.info(f"Track {track_id}: 退出移动状态 (连续静止 {state['static_frames']} 帧, "
                          f"平均速度={avg_speed:.2f})")
            state['is_moving'] = False
        
        logger.info(f"Track {track_id}: speed={current_speed:.2f}, avg_speed={avg_speed:.2f}, "
                    f"is_moving={state['is_moving']}, moving_frames={state['moving_frames']}")
        
        return state['is_moving']
    
    def get_moving_state(self, track_id):
        """获取指定track_id的移动状态"""
        if track_id in self.track_moving_state:
            state = self.track_moving_state[track_id]
            # 如果已经标记为移动，直接返回True
            if state['is_moving']:
                return True
            # 如果有速度数据，检查平均速度是否超过阈值
            if state['speeds'] and len(state['speeds']) >= 2:
                avg_speed = sum(state['speeds']) / len(state['speeds'])
                if avg_speed > self.moving_speed_threshold:
                    return True
            return state['is_moving']
        return False
    
    def find_track_by_position(self, person_box, threshold=100):
        """
        通过位置匹配找到最接近的历史track
        当track_id发生变化时，使用位置匹配来找到之前的track
        
        Args:
            person_box: 当前人体框 [x1, y1, x2, y2]
            threshold: 位置匹配阈值（像素），默认100
        
        Returns:
            int: 匹配的track_id，如果没有匹配则返回None
        """
        if not self.track_moving_state:
            return None
        
        center_x = (person_box[0] + person_box[2]) / 2
        center_y = (person_box[1] + person_box[3]) / 2
        
        best_track_id = None
        best_distance = threshold
        
        for track_id, state in self.track_moving_state.items():
            if not state['history']:
                continue
            # 使用最近的位置进行匹配
            last_pos = state['history'][-1]
            distance = ((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)**0.5
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        
        return best_track_id
    
    def cleanup_old_tracks(self, active_track_ids, max_age=50):
        """清理不活跃的跟踪记录"""
        inactive_ids = []
        for track_id in self.track_moving_state:
            if track_id not in active_track_ids:
                inactive_ids.append(track_id)
        
        for track_id in inactive_ids:
            logger.debug(f"清理不活跃的跟踪记录: track_id={track_id}")
            del self.track_moving_state[track_id]
