"""
绘图工具模块 - 绘制检测框和统计面板
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def put_text(image, xy, text, color=(0, 255, 0), text_size=50):
    """在PIL Image上绘制文字"""
    font_path = os.path.join(os.path.dirname(__file__), '..', 'resource', 'ai.ttf')
    try:
        ttfont = ImageFont.truetype(font=font_path, size=text_size)
    except:
        ttfont = ImageFont.load_default()
    
    draw = ImageDraw.Draw(image)
    draw.text(xy, text, fill=color, font=ttfont)
    return image


def draw_result(image, detections, alarm_person_boxes=None):
    """
    绘制检测结果和告警信息
    
    Args:
        image: numpy array (BGR) 或 PIL Image
        detections: 检测结果列表, 每个格式 [x1, y1, x2, y2, conf, cls]
        alarm_person_boxes: 告警人体框列表, 每个格式 [x1, y1, x2, y2, ...]
    
    Returns:
        PIL Image with drawn boxes
    """
    # 确保 image 是 PIL Image
    if isinstance(image, np.ndarray):
        result_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        result_img = image.copy()
    else:
        result_img = image
    
    if alarm_person_boxes is None:
        alarm_person_boxes = []
    
    # 创建告警框集合
    alarm_boxes_set = set()
    for box in alarm_person_boxes:
        if len(box) >= 4:
            box_key = tuple(round(coord) for coord in box[:4])
            alarm_boxes_set.add(box_key)
    
    for det in detections:
        if len(det) >= 6:
            box = det[:6]
            
            # 判断是否是告警的人体框
            box_key = tuple(round(coord) for coord in box[:4])
            is_alarm = box_key in alarm_boxes_set
            
            draw = ImageDraw.Draw(result_img)
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            
            if is_alarm:
                # 告警框：红色粗边框
                thickness = 3
                for i in range(thickness):
                    draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=(255, 0, 0))
            else:
                # 普通框：绿色细边框
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            del draw
    
    return result_img


def add_stats_panel(image, alarm_message, logo_img=None, scenario_ids=None, language="ch"):
    """
    在图像左上角添加统计面板
    
    Args:
        image: PIL Image
        alarm_message: 告警计数字典 {scenario_id: count} 或 {scenario_name: count}
        logo_img: logo图片路径或PIL Image
        scenario_ids: 要显示的场景ID列表
        language: 语言 "ch" 或 "en"
    
    Returns:
        PIL Image with stats panel
    """
    from scenarios import ALARM_NAMES
    
    # 场景分类
    all_person_items = {
        0: "入侵", 3: "打电话", 4: "抽烟", 5: "跌倒", 7: "睡岗", 8: "移动打电话"
    }
    all_safety_items = {
        1: "未戴安全帽", 2: "未穿工装", 6: "未戴安全带"
    }
    all_behavior_items = {
        10: "攀爬", 11: "打架"
    }
    
    # 根据 scenario_ids 过滤
    if scenario_ids is not None:
        person_items = {k: v for k, v in all_person_items.items() if k in scenario_ids}
        safety_items = {k: v for k, v in all_safety_items.items() if k in scenario_ids}
        behavior_items = {k: v for k, v in all_behavior_items.items() if k in scenario_ids}
    else:
        person_items = all_person_items
        safety_items = all_safety_items
        behavior_items = all_behavior_items
    
    # 获取图像尺寸
    img_width, img_height = image.size
    
    # 基准分辨率（适配1920*1080）
    BASE_WIDTH = 1920
    BASE_HEIGHT = 1080
    
    # 倍数
    whole_ratio = max(img_width / BASE_WIDTH, img_height / BASE_HEIGHT)
    scale_ratio = whole_ratio
    
    # 动态计算参数
    offset_x = int(20 * scale_ratio)
    offset_y = int(20 * scale_ratio)
    font_size_title = int(32 * scale_ratio)
    font_size_item = int(28 * scale_ratio)
    line_height = int(40 * scale_ratio)
    
    # 构建告警数据
    stats = {}
    
    # 人员行为
    person_stats = {}
    for code, name in person_items.items():
        if isinstance(alarm_message, dict):
            # 支持按ID或按名称查找
            count = alarm_message.get(code, alarm_message.get(name, 0))
        else:
            count = 0
        person_stats[name] = count
    if person_stats:
        stats["人员行为" if language == "ch" else "Person Behavior"] = person_stats
    
    # 劳保用品
    safety_stats = {}
    for code, name in safety_items.items():
        if isinstance(alarm_message, dict):
            count = alarm_message.get(code, alarm_message.get(name, 0))
        else:
            count = 0
        safety_stats[name] = count
    if safety_stats:
        stats["劳保用品" if language == "ch" else "Safety Items"] = safety_stats
    
    # 行为识别
    behavior_stats = {}
    for code, name in behavior_items.items():
        if isinstance(alarm_message, dict):
            count = alarm_message.get(code, alarm_message.get(name, 0))
        else:
            count = 0
        behavior_stats[name] = count
    if behavior_stats:
        stats["行为识别" if language == "ch" else "Behavior Recognition"] = behavior_stats
    
    # 如果没有要显示的项，返回原图
    if not stats:
        return image
    
    # 计算总行数
    total_lines = 0
    
    # logo 占用的行数
    logo_height = 0
    logo_width = 0
    if logo_img is not None:
        if isinstance(logo_img, str):
            logo = Image.open(logo_img)
        else:
            logo = logo_img
        
        # 获取logo原始尺寸
        original_logo_width, original_logo_height = logo.size
        
        # 计算缩放后的logo尺寸
        max_logo_width = int(464 * scale_ratio * 0.4)
        if original_logo_width > max_logo_width:
            scale = max_logo_width / original_logo_width
            logo_width = max_logo_width
            logo_height = int(original_logo_height * scale)
        else:
            logo_width = original_logo_width
            logo_height = original_logo_height
        
        logo_lines = int(logo_height / line_height) if line_height > 0 else 0
        total_lines += logo_lines
    
    # 计算各分类行数
    for title, items in stats.items():
        items_count = len(items)
        if items_count > 0:
            items_per_row = 3 if "行为" in title else 2
            rows = (items_count + items_per_row - 1) // items_per_row
            total_lines += rows + 1  # +1 是标题行
    
    # 计算面板尺寸
    panel_width = int(464 * scale_ratio)
    if panel_width < 200:
        panel_width = 200
    panel_height = int(total_lines * line_height + offset_y * 2 + logo_height)
    
    # 确保面板不超出图像边界
    panel_width = min(panel_width, img_width - offset_x * 2)
    panel_height = min(panel_height, img_height - offset_y * 2)
    
    # 创建半透明背景面板
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 创建覆盖层
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # 绘制背景矩形（灰色，半透明）
    bg_color = (200, 200, 200, 200)
    draw_overlay.rectangle([0, 0, panel_width, panel_height], fill=bg_color)
    
    # 合并背景
    image = Image.alpha_composite(image, overlay)
    
    # 绘制文字和logo的起始位置
    x = offset_x + int(10 * scale_ratio)
    y = offset_y + int(20 * scale_ratio)
    
    # ========== 绘制 Logo ==========
    if logo_img is not None:
        if isinstance(logo_img, str):
            logo = Image.open(logo_img)
        else:
            logo = logo_img
        
        if logo.size[0] > 0:
            target_width = int(panel_width * 0.4)
            target_height = int(logo.size[1] * target_width / logo.size[0])
            
            if target_height > panel_height * 0.3:
                target_height = int(panel_height * 0.3)
                target_width = int(logo.size[0] * target_height / logo.size[1])
            
            try:
                logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)
            except AttributeError:
                logo = logo.resize((target_width, target_height), Image.LANCZOS)
            
            logo_x = (panel_width - target_width) // 2
            logo_y = 0
            
            if logo.mode == 'RGBA':
                image.paste(logo, (logo_x, logo_y), logo)
            else:
                image.paste(logo, (logo_x, logo_y))
            
            y = logo_y + target_height + int(5 * scale_ratio)
        else:
            y = offset_y + int(30 * scale_ratio)
    else:
        y = offset_y + int(30 * scale_ratio)
    
    # ========== 绘制各分类 ==========
    for title, items in stats.items():
        # 标题
        image = put_text(image, (x, y), title, color=(0, 0, 200), text_size=font_size_title)
        y += line_height
        
        # 每行最多3项
        items_per_row = 3
        item_spacing = int(panel_width * 0.32)
        
        items_list = list(items.items())
        for i in range(0, len(items_list), items_per_row):
            row_items = items_list[i:i + items_per_row]
            current_x = x
            
            for name, value in row_items:
                if value > 0:
                    color = (255, 0, 0)  # 红色 - 有告警
                else:
                    color = (0, 150, 0)  # 绿色 - 无告警
                text = f"{name}({int(value)})"
                image = put_text(image, (current_x, y), text, color=color, text_size=font_size_item)
                current_x += item_spacing
            y += line_height
        
        y += int(5 * scale_ratio)
    
    return image


def draw_alarm_frame(frame, detections, alarm_boxes, alarm_scenarios, alarm_counts, logo_path=None, scenario_ids=None):
    """
    绘制告警帧：检测框 + 统计面板
    
    Args:
        frame: numpy array (BGR格式)
        detections: 检测结果列表 [x1, y1, x2, y2, conf, cls]
        alarm_boxes: 告警人体框列表
        alarm_scenarios: 告警场景字典 {scenario_name: count}
        alarm_counts: 告警计数 {scenario_id: count}
        logo_path: logo图片路径
        scenario_ids: 要显示的场景ID列表
    
    Returns:
        numpy array (BGR格式) 绘制后的图像
    """
    # 绘制检测框
    result_img = draw_result(frame, detections, alarm_boxes)
    
    # 添加统计面板
    if alarm_counts:
        result_img = add_stats_panel(result_img, alarm_counts, logo_img=logo_path, scenario_ids=scenario_ids)
    
    # 转换回numpy array
    if isinstance(result_img, Image.Image):
        if result_img.mode == 'RGBA':
            result_img = result_img.convert('RGB')
        result_np = np.array(result_img)
        # PIL是RGB，OpenCV需要BGR
        result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        return result_np
    
    return frame
