import os
import logging
from logging.handlers import TimedRotatingFileHandler

def create_logger(log_file, logger_name):
    """
    创建日志记录器
    
    Args:
        log_file: 日志文件路径
        logger_name: logger 名称
    
    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    # 1. 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 2. 创建 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 3. 避免重复添加 handler（防止多次调用时重复）
    if logger.handlers:
        return logger
    
    # 4. 创建定时轮转文件处理器
    try:
        logHandler = TimedRotatingFileHandler(
            log_file,
            when='midnight',      # 每天午夜轮转
            interval=1,
            backupCount=30,       # 保留30天的日志
            encoding='utf-8'
        )
    except Exception as e:
        # 如果文件处理器创建失败，降级使用 StreamHandler
        print(f"警告: 无法创建日志文件 {log_file}: {e}")
        logHandler = logging.StreamHandler()
    
    # 5. 设置格式
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    logHandler.setFormatter(logFormatter)
    
    # 6. 添加处理器
    logger.addHandler(logHandler)
    
    return logger


# 创建全局 logger 实例
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inner_logger = create_logger(os.path.join(project_root, "logs/detect.log"), "inner_logger")
outter_logger = create_logger(os.path.join(project_root, "logs/global.log"), "outter_logger")