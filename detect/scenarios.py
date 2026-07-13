"""
场景检测函数集合
包含各种安全场景的检测逻辑：
- 人员入侵检测
- 安全帽佩戴检测
- 工装穿着检测
- 打电话检测
- 抽烟检测
- 跌倒检测
- 安全带检测
- 攀爬检测
"""

import numpy as np
from logger import inner_logger as logger
from rules.phone_tracking_manager import PhoneTrackingManager

# 全局跟踪管理器实例
_phone_tracking_manager = PhoneTrackingManager()


# 类别ID常量
PERSON = 0
HEAD = 1
HELMET = 2
CLOTHES_RED = 3
CLOTHES_GRAY = 4
CLOTHES_YELLOW = 5
CLOTHES_BLUE = 6
CLOTHES_SIMILAR = 7
CLOTHES_REFLECTIVE = 8
PHONE = 9
SMOKING = 10
DOWN = 11
SAFETY_BELT = 12
SLEEPING = 13


# ==================== 工具函数 ====================

def point_in_poly(point, poly):
    """
    判断点是否在多边形内（射线法）
    
    Args:
        point: (x, y) 或 [x, y]
        poly: 多边形，支持格式：
              - 扁平列表: [x1,y1,x2,y2,x3,y3,...]
              - 嵌套列表: [[x1,y1,x2,y2,...]] 或 [[x1,y1],[x2,y2],...]
    
    Returns:
        bool: True=在多边形内，False=在外
    """
    x, y = point
    
    # 处理嵌套列表格式 [[0,0,1920,0,1920,1080,0,1080]]
    if poly and len(poly) == 1 and isinstance(poly[0], list):
        poly = poly[0]
    
    # 转换为点对列表
    vertices = []
    if poly and isinstance(poly[0], (int, float)):
        # 扁平列表格式
        for i in range(0, len(poly), 2):
            if i + 1 < len(poly):
                vertices.append((float(poly[i]), float(poly[i+1])))
    elif poly and isinstance(poly[0], (list, tuple)):
        # 点列表格式
        for p in poly:
            if len(p) >= 2:
                vertices.append((float(p[0]), float(p[1])))
    else:
        return False
    
    if len(vertices) < 3:
        return False
    
    # 射线法判断
    inside = False
    n = len(vertices)
    
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        
        # 检查点是否在边上
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        
        # 射线与边相交的条件
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x_intersect > x:
                inside = not inside
    
    return inside


def is_point_in_box(point, box):
    """判断点是否在边界框内"""
    if point is None or box is None:
        return False
    if len(point) < 2 or len(box) < 4:
        return False
    
    x, y = point[0], point[1]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return x1 <= x <= x2 and y1 <= y <= y2


def is_box_intersect(box1, box2):
    """
    判断两个边界框是否相交
    box格式: [x1, y1, x2, y2]
    """
    try:
        if len(box1) < 4 or len(box2) < 4:
            return False
        
        x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
        x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]
        
        # 处理 numpy 数组
        if hasattr(x1_1, 'item'):
            x1_1 = x1_1.item()
            y1_1 = y1_1.item()
            x2_1 = x2_1.item()
            y2_1 = y2_1.item()
            x1_2 = x1_2.item()
            y1_2 = y1_2.item()
            x2_2 = x2_2.item()
            y2_2 = y2_2.item()
        
        # 检查是否相交
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    except (TypeError, ValueError, IndexError, AttributeError):
        return False


def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    try:
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 如果没有交集
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算各自的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 并集面积
        union = area1 + area2 - intersection
        
        # 返回IOU
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.error(f"calculate_iou error: {e}")
        return 0.0


def check_det_in_area(dets, area):
    """检查检测框是否在指定区域内"""
    if len(area) == 0:
        return dets, "全域检测"
    
    # 如果区域中个数不对
    for each_area in area:
        if len(each_area) != 8:
            logger.error(f"area format error: {each_area}")
            return dets, "检测区域格式错误"
    
    # 获取检测框的中心点坐标，判断中心点是否在检测区域内
    dets_in_area = []
    for det in dets:
        det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
        for each_area in area:
            if point_in_poly(det_center, each_area):
                dets_in_area.append(det)
    
    return dets_in_area, ""


def judge_det(det1, det2):
    """判断检测框det1是否在det2中"""
    det1_center = [(det1[0] + det1[2]) / 2, (det1[1] + det1[3]) / 2]
    if det1_center[0] > det2[0] and det1_center[0] < det2[2] and det1_center[1] > det2[1] and det1_center[1] < det2[3]:
        return True
    else:
        return False


def is_box_same(box1, box2, threshold=5):
    """判断两个边界框是否相同"""
    if len(box1) < 4 or len(box2) < 4:
        return False
    
    # 比较四个坐标，允许一定的误差
    return (abs(box1[0] - box2[0]) < threshold and
            abs(box1[1] - box2[1]) < threshold and
            abs(box1[2] - box2[2]) < threshold and
            abs(box1[3] - box2[3]) < threshold)


# ==================== 场景检测函数 ====================

def personel_intrusion(det):
    """
    人员入侵检测
    
    Args:
        det: 按类别分类的检测结果列表，det[PERSON] 为人体检测框
    
    Returns:
        tuple: (入侵人数, 入侵人员框列表)
    """
    logger.debug("check personel_intrusion")
    return len(det[PERSON])


def person_wear_helmet_check(det):
    """
    安全帽佩戴检测
    
    逻辑：
    1. 有人体、有安全帽/人头才检查
    2. 检查安全帽是否正确戴在人头上
    3. 未正确佩戴则计数
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 未戴安全帽人数
    """
    logger.debug("check person_wear_helmet_check")
    
    person_boxs = det[PERSON]
    head_boxs = det[HEAD]
    helmet_boxs = det[HELMET]
    
    if not person_boxs:
        return 0
    
    no_helmet_count = 0
    
    for person in person_boxs:
        person_box = person[:4]
        
        # 获取人体内的安全帽和人头
        has_helmet_in_person = any(
            is_point_in_box(((helmet[0] + helmet[2]) / 2, (helmet[1] + helmet[3]) / 2), person_box)
            for helmet in helmet_boxs
        )
        
        has_head_in_person = any(
            is_point_in_box(((head[0] + head[2]) / 2, (head[1] + head[3]) / 2), person_box)
            for head in head_boxs
        )
        
        # 情况1：有安全帽 + 有人头 → 检查安全帽是否在人头上
        if has_helmet_in_person and has_head_in_person:
            properly_worn = False
            for head in head_boxs:
                head_center = ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2)
                if not is_point_in_box(head_center, person_box):
                    continue
                
                for helmet in helmet_boxs:
                    helmet_center = ((helmet[0] + helmet[2]) / 2, (helmet[1] + helmet[3]) / 2)
                    if not is_point_in_box(helmet_center, person_box):
                        continue
                    
                    # 安全帽中心在人头内 = 正确佩戴
                    if is_point_in_box(helmet_center, head[:4]):
                        properly_worn = True
                        break
                if properly_worn:
                    break
            
            if not properly_worn:
                no_helmet_count += 1
        
        # 情况2：无安全帽 + 有人头 → 未戴安全帽
        elif not has_helmet_in_person and has_head_in_person:
            no_helmet_count += 1
    
    return no_helmet_count


def person_wear_clothes_check(det):
    """
    工装检测：如果人体范围内没有检测到任何工装，则告警
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 未穿工装人数
    """
    logger.debug("check person_wear_clothes_check")
    
    person_boxs = det[PERSON]
    clothes_red_boxs = det[CLOTHES_RED]
    clothes_gray_boxs = det[CLOTHES_GRAY]
    clothes_yellow_boxs = det[CLOTHES_YELLOW]
    clothes_blue_boxs = det[CLOTHES_BLUE]
    clothes_similar_boxs = det[CLOTHES_SIMILAR]
    clothes_reflective_boxs = det[CLOTHES_REFLECTIVE]
    
    if not person_boxs:
        return 0
    
    # 合并所有工装类型
    all_clothes = []
    all_clothes.extend(clothes_red_boxs)
    all_clothes.extend(clothes_gray_boxs)
    all_clothes.extend(clothes_yellow_boxs)
    all_clothes.extend(clothes_blue_boxs)
    all_clothes.extend(clothes_similar_boxs)
    all_clothes.extend(clothes_reflective_boxs)
    
    no_clothes_count = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        clothes_detected = False
        
        for clothes in all_clothes:
            clothes_center = ((clothes[0] + clothes[2]) / 2, (clothes[1] + clothes[3]) / 2)
            if is_point_in_box(clothes_center, person_box):
                clothes_detected = True
                break
        
        if not clothes_detected:
            no_clothes_count += 1
    
    return no_clothes_count


def person_using_phone_check(det):
    """
    打电话检测：检查电话框是否与人头框或安全帽框相交
    
    逻辑：
    1. 有人体、有电话才检查
    2. 筛选人体范围内的人头和安全帽
    3. 检查电话框是否与人头框或安全帽框相交
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 打电话人数
    """
    logger.debug("check person_using_phone_check")
    
    person_boxs = det[PERSON]
    head_boxs = det[HEAD]
    helmet_boxs = det[HELMET]
    phone_boxs = det[PHONE]
    
    if not person_boxs:
        return 0
    
    if not phone_boxs:
        return 0
    
    phone_count = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        phone_detected = False
        
        # 筛选当前人体范围内的人头
        heads_in_person = []
        for head in head_boxs:
            head_center = ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2)
            if is_point_in_box(head_center, person_box):
                heads_in_person.append(head)
        
        # 筛选当前人体范围内的安全帽
        helmets_in_person = []
        for helmet in helmet_boxs:
            helmet_center = ((helmet[0] + helmet[2]) / 2, (helmet[1] + helmet[3]) / 2)
            if is_point_in_box(helmet_center, person_box):
                helmets_in_person.append(helmet)
        
        # 合并人头和安全帽
        heads_in_person.extend(helmets_in_person)
        
        if not heads_in_person:
            continue
        
        # 检查电话框是否与人头框/安全帽框相交
        for head in heads_in_person:
            head_box = head[:4] if len(head) >= 4 else head
            for phone in phone_boxs:
                phone_box = phone[:4] if len(phone) >= 4 else phone
                if is_box_intersect(phone_box, head_box):
                    phone_detected = True
                    break
            if phone_detected:
                break
        
        if phone_detected:
            phone_count += 1
    
    return phone_count


def person_using_moving_phone_check(det, tracks=None, frame_id=0):
    """
    移动打电话检测：检测打电话同时正在移动的人员
    
    逻辑：
    1. 每帧跟踪所有人的移动状态
    2. 检测打电话（电话框与人头/安全帽相交）
    3. 只有移动中打电话才告警
    
    Args:
        det: 按类别分类的检测结果列表
        tracks: 跟踪结果数组 (N, 8) [x1,y1,x2,y2,track_id,conf,cls,track_length]
        frame_id: 当前帧ID，用于计算移动
    
    Returns:
        int: 移动打电话人数
    """
    logger.debug("check person_using_moving_phone_check")
    
    person_boxs = det[PERSON]
    head_boxs = det[HEAD]
    helmet_boxs = det[HELMET]
    phone_boxs = det[PHONE]
    
    # 没有人体，直接返回0
    if not person_boxs:
        return 0
    
    # 没有电话，直接返回0
    if not phone_boxs:
        return 0
    
    # 构建跟踪信息映射 {track_id: track_box}
    track_map = {}
    track_length_map = {}
    if tracks is not None and len(tracks) > 0:
        for track in tracks:
            if len(track) >= 8:
                track_id = int(track[4])
                track_box = track[:4]
                track_length = int(track[7])
                track_map[track_id] = track_box
                track_length_map[track_id] = track_length
    
    # 活跃的track_id集合，用于清理
    active_track_ids = set()
    
    # 每帧跟踪所有人的移动状态（无论是否检测到电话）
    # 优先使用track_id匹配，如果track_id不在track_map中则使用位置匹配
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        # 通过IOU匹配找到当前人体对应的track_id
        person_track_id = None
        if track_map:
            best_iou = 0
            for track_id, track_box in track_map.items():
                iou = calculate_iou(person_box, track_box)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    person_track_id = track_id
        
        if person_track_id is not None:
            active_track_ids.add(person_track_id)
            _phone_tracking_manager.update_track(person_track_id, person_box, frame_id)
        else:
            # track_id不在track_map中，使用位置匹配找到最接近的历史track
            matched_track_id = _phone_tracking_manager.find_track_by_position(person_box)
            if matched_track_id is not None:
                active_track_ids.add(matched_track_id)
                _phone_tracking_manager.update_track(matched_track_id, person_box, frame_id)
    
    moving_phone_count = 0
    
    # 遍历每个人体，检查打电话+移动
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        phone_detected = False
        person_track_id = None
        
        # 1. 找到当前人体对应的跟踪ID（通过IOU匹配）
        if track_map:
            best_iou = 0
            for track_id, track_box in track_map.items():
                iou = calculate_iou(person_box, track_box)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    person_track_id = track_id
        
        # 如果track_id不在track_map中，使用位置匹配
        if person_track_id is None:
            person_track_id = _phone_tracking_manager.find_track_by_position(person_box)
        
        # 2. 筛选当前人体范围内的人头（中心点在人体内）
        heads_in_person = []
        for head in head_boxs:
            head_box = head[:4] if len(head) >= 4 else head
            head_center = ((head_box[0] + head_box[2]) / 2, (head_box[1] + head_box[3]) / 2)
            if is_point_in_box(head_center, person_box):
                heads_in_person.append(head_box)
        
        # 3. 筛选当前人体范围内的安全帽（中心点在人体内）
        helmets_in_person = []
        for helmet in helmet_boxs:
            helmet_box = helmet[:4] if len(helmet) >= 4 else helmet
            helmet_center = ((helmet_box[0] + helmet_box[2]) / 2, (helmet_box[1] + helmet_box[3]) / 2)
            if is_point_in_box(helmet_center, person_box):
                helmets_in_person.append(helmet_box)
        
        # 合并人头和安全帽
        heads_in_person.extend(helmets_in_person)
        
        # 没有人头/安全帽，跳过
        if not heads_in_person:
            continue
        
        # 4. 筛选当前人体范围内的电话（中心点在人体内）
        phones_in_person = []
        for phone in phone_boxs:
            phone_box = phone[:4] if len(phone) >= 4 else phone
            phone_center = ((phone_box[0] + phone_box[2]) / 2, (phone_box[1] + phone_box[3]) / 2)
            if is_point_in_box(phone_center, person_box):
                phones_in_person.append(phone_box)
        
        if not phones_in_person:
            continue
        
        # 5. 检查电话框是否与人头框/安全帽框相交
        for head_box in heads_in_person:
            for phone_box in phones_in_person:
                if is_box_intersect(phone_box, head_box):
                    phone_detected = True
                    break
            if phone_detected:
                break
        
        # 6. 检测到打电话，判断是否在移动
        if phone_detected:
            if person_track_id is not None:
                is_moving = _phone_tracking_manager.get_moving_state(person_track_id)
                track_length = track_length_map.get(person_track_id, 0)
                logger.info(f"移动打电话检测: track_id={person_track_id}, track_length={track_length}, is_moving={is_moving}")
                
                # 辅助判断：跟踪长度超过10帧且正在移动
                if track_length > 10 and is_moving:
                    is_moving = True
                
                if is_moving:
                    moving_phone_count += 1
    
    # 清理不活跃的跟踪记录
    _phone_tracking_manager.cleanup_old_tracks(active_track_ids)
    
    return moving_phone_count


def person_smoking_check(det):
    """
    抽烟检测：检查抽烟框是否与人头框或安全帽框相交
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 抽烟人数
    """
    logger.debug("check person_smoking_check")
    
    person_boxs = det[PERSON]
    head_boxs = det[HEAD]
    helmet_boxs = det[HELMET]
    smoking_boxs = det[SMOKING]
    
    if not person_boxs:
        return 0
    
    if not smoking_boxs:
        return 0
    
    smoking_count = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        smoking_detected = False
        
        heads_in_person = []
        for head in head_boxs:
            head_center = ((head[0] + head[2]) / 2, (head[1] + head[3]) / 2)
            if is_point_in_box(head_center, person_box):
                heads_in_person.append(head)
        
        helmets_in_person = []
        for helmet in helmet_boxs:
            helmet_center = ((helmet[0] + helmet[2]) / 2, (helmet[1] + helmet[3]) / 2)
            if is_point_in_box(helmet_center, person_box):
                helmets_in_person.append(helmet)
        
        heads_in_person.extend(helmets_in_person)
        
        if not heads_in_person:
            continue
        
        for head in heads_in_person:
            head_box = head[:4] if len(head) >= 4 else head
            for smoking in smoking_boxs:
                smoking_box = smoking[:4] if len(smoking) >= 4 else smoking
                if is_box_intersect(smoking_box, head_box):
                    smoking_detected = True
                    break
            if smoking_detected:
                break
        
        if smoking_detected:
            smoking_count += 1
    
    return smoking_count


def person_falldown_check(det):
    """
    跌倒检测：检查跌倒框中心点是否在人体框内
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 跌倒人数
    """
    logger.debug("check person_falldown_check")
    
    person_boxs = det[PERSON]
    down_boxs = det[DOWN]
    
    if not person_boxs:
        return 0
    
    if not down_boxs:
        return 0
    
    falldown_num = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        
        for down in down_boxs:
            down_center = ((down[0] + down[2]) / 2, (down[1] + down[3]) / 2)
            if is_point_in_box(down_center, person_box):
                falldown_num += 1
                break
    
    return falldown_num


def person_wear_safety_belt_check(det):
    """
    安全带检测：检查人体是否佩戴安全带
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 未系安全带人数
    """
    logger.debug("check person_wear_safety_belt_check")
    
    person_boxs = det[PERSON]
    safety_belt_boxs = det[SAFETY_BELT]
    
    if not person_boxs:
        return 0
    
    no_safety_belt_num = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        safety_belt_detected = False
        
        if not safety_belt_boxs:
            safety_belt_detected = True
        else:
            for safety_belt in safety_belt_boxs:
                safety_belt_center = ((safety_belt[0] + safety_belt[2]) / 2, (safety_belt[1] + safety_belt[3]) / 2)
                if not is_point_in_box(safety_belt_center, person_box):
                    safety_belt_detected = True
                    break
        
        if not safety_belt_detected:
            no_safety_belt_num += 1
    
    return no_safety_belt_num

def person_gathering_check(det,thesh = 3):
    """聚集检测

    Args:
        det (_type_): _description_
        thesh (int, optional): 默认聚集人数.
    """
    logger.debug("check person_gathering_check")
    person_boxs = det[PERSON]
    if len(person_boxs) < thesh:
        return 0
    min_left = min([box[0] for box in person_boxs])
    min_top = min([box[1] for box in person_boxs])
    max_right = max([box[2] for box in person_boxs])
    max_bottom = max([box[3] for box in person_boxs])

    return len(person_boxs)

def person_sleeping_check(det):
    """
    睡岗检测
    
    Args:
        det: 按类别分类的检测结果列表
    
    Returns:
        int: 睡岗人数
    """
    logger.debug("check person_sleeping_check")
    
    person_boxs = det[PERSON]
    sleeping_boxs = det[SLEEPING] if len(det) > SLEEPING else []
    
    if not person_boxs:
        return 0
    
    if not sleeping_boxs:
        return 0
    
    sleeping_count = 0
    
    for person in person_boxs:
        person_box = person[:4] if len(person) >= 4 else person
        
        for sleeping in sleeping_boxs:
            sleeping_center = ((sleeping[0] + sleeping[2]) / 2, (sleeping[1] + sleeping[3]) / 2)
            if is_point_in_box(sleeping_center, person_box):
                sleeping_count += 1
                break
    
    return sleeping_count


# ==================== 告警检测调度 ====================

# 告警类型映射
ALARM_MAP = {
    0: personel_intrusion,
    1: person_wear_helmet_check,
    2: person_wear_clothes_check,
    3: person_using_phone_check,
    #3: person_using_moving_phone_check,
    4: person_smoking_check,
    5: person_falldown_check,
    6: person_wear_safety_belt_check,
    7: person_gathering_check,
    8: person_sleeping_check,
}

# 告警类型描述
ALARM_NAMES = {
    0: "人员入侵",
    1: "未戴安全帽",
    2: "未穿工装",
    3: "打电话",
    4: "抽烟",
    5: "跌倒",
    6: "未系安全带",
    7: "聚集",
    8: "睡岗",
    9: "离岗",
    10: "攀爬",
    11: "打架",
}

# YOLO 场景检测 ID（使用传统检测）
YOLO_SCENARIO_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# VideoMAE 行为识别 ID
VIDEOMAE_SCENARIO_IDS = [10, 11]


def classify_detections(all_dets, num_classes=14, track_info=None):
    """
    将检测结果按类别分类
    
    Args:
        all_dets: 所有检测结果，每个检测格式为 [x1, y1, x2, y2, conf, cls]
        num_classes: 类别数量
        track_info: 跟踪信息字典 {track_id: {'track_length': ..., 'bbox': ..., 'class': ...}}
    
    Returns:
        list: 按类别索引分类的检测框列表
    """
    boxes_by_class = [[] for _ in range(num_classes)]
    
    for det in all_dets:
        cls = int(det[-1])
        if cls < num_classes:
            # 如果有跟踪信息，添加到检测结果中
            det_with_track = list(det[:6]) if len(det) >= 6 else list(det)
            
            # 尝试匹配跟踪ID
            if track_info:
                det_bbox = det[:4]
                for tid, tinfo in track_info.items():
                    track_bbox = tinfo.get('bbox', [])
                    if len(track_bbox) == 4:
                        # 计算IoU匹配
                        iou = calculate_iou(det_bbox, track_bbox)
                        if iou > 0.5:  # IoU阈值
                            det_with_track.append(tid)  # track_id
                            det_with_track.append(tinfo.get('track_length', 0))  # track_length
                            break
            
            boxes_by_class[cls].append(det_with_track)
    
    return boxes_by_class


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def run_scenario_checks(all_dets, scenario_ids, num_classes=14, track_info=None, frame_id=0, tracks=None):
    """
    执行场景检测
    
    Args:
        all_dets: 所有检测结果
        scenario_ids: 要执行的场景ID列表
        num_classes: 类别数量
        track_info: 跟踪信息字典
        frame_id: 当前帧ID
        tracks: 原始跟踪结果数组 (N, 8) [x1,y1,x2,y2,track_id,conf,cls,track_length]
    
    Returns:
        dict: 检测结果 {scenario_id: count}
    """
    # 按类别分类检测结果（包含跟踪信息）
    boxes_by_class = classify_detections(all_dets, num_classes, track_info)
    
    results = {}
    
    for scenario_id in scenario_ids:
        if scenario_id in ALARM_MAP:
            check_func = ALARM_MAP[scenario_id]
            try:
                # 对于移动打电话场景，需要传递 tracks 和 frame_id
                if scenario_id == 8:  # person_using_moving_phone_check
                    count = check_func(boxes_by_class, tracks=tracks, frame_id=frame_id)
                else:
                    count = check_func(boxes_by_class)
                results[scenario_id] = count
                if count > 0:
                    logger.warning(f"场景检测告警: {ALARM_NAMES.get(scenario_id, 'unknown')} (id={scenario_id}), count={count}")
                else:
                    logger.debug(f"Scenario {scenario_id} ({ALARM_NAMES.get(scenario_id, 'unknown')}): {count}")
            except Exception as e:
                logger.error(f"Scenario {scenario_id} check error: {e}")
                results[scenario_id] = 0
        else:
            logger.warning(f"Unknown scenario ID: {scenario_id}")
            results[scenario_id] = 0
    
    return results


def get_alarm_summary(results):
    """
    获取告警摘要
    
    Args:
        results: 检测结果字典
    
    Returns:
        dict: 告警摘要 {scenario_name: count}
    """
    summary = {}
    for scenario_id, count in results.items():
        if count > 0:
            name = ALARM_NAMES.get(scenario_id, f"scenario_{scenario_id}")
            summary[name] = count
    return summary
