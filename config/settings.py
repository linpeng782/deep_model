"""
深度模型项目的配置设置文件
包含数据处理、回测等相关参数配置
"""
import os
from datetime import datetime

# 数据配置
DATA_CONFIG = {
    "max_workers": 4,        # 并行处理的最大工作线程数
    "use_parallel": True,    # 是否使用并行处理
}

# 回测配置
BACKTEST_CONFIG = {
    "initial_capital": 1000000,  # 初始资金
    "commission_rate": 0.0003,   # 手续费率
    "holding_months": 12,        # 持仓月数
}

# 通用配置获取函数
def get_config(config_type):
    """
    获取指定类型的配置
    
    Args:
        config_type (str): 配置类型，如 'data', 'backtest'
    
    Returns:
        dict: 对应的配置字典
    """
    config_map = {
        "data": DATA_CONFIG,
        "backtest": BACKTEST_CONFIG,
    }
    
    return config_map.get(config_type, {})

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

if __name__ == "__main__":
    # 测试配置获取
    data_config = get_config("data")
    print("数据配置:", data_config)
    
    backtest_config = get_config("backtest")
    print("回测配置:", backtest_config)
