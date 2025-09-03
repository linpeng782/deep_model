"""
统一配置管理模块
负责加载和管理所有项目配置信息
"""
import os
import yaml
from datetime import datetime


# 配置缓存变量
_config_cache = None


def load_factor_config(config_path=None):
    """
    加载因子配置文件
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "factor_config.yaml"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        raise RuntimeError(f"无法加载配置文件: {config_path}")


def get_config():
    """
    获取配置信息（懒加载模式）
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_factor_config()
    return _config_cache


def get_processing_config():
    """
    快捷获取处理配置
    """
    return get_config()["processing_config"]


def get_default_end_date():
    """
    快捷获取默认结束日期
    """
    return get_processing_config()["default_end_date"]


def get_test_mode():
    """
    快捷获取测试模式
    """
    return get_processing_config()["test_mode"]


def get_max_workers():
    """
    快捷获取最大工作线程数
    """
    return get_processing_config()["max_workers"]


def get_timestamped_output_path(base_path):
    """
    生成带时间戳的输出路径
    
    Args:
        base_path (str): 基础路径
    
    Returns:
        str: 带时间戳的完整路径
    """
    # 使用年月日格式的时间戳
    timestamp = datetime.now().strftime("%Y%m%d")
    timestamped_path = f"{base_path}_{timestamp}"
    
    # 确保目录存在
    os.makedirs(timestamped_path, exist_ok=True)
    
    return timestamped_path
