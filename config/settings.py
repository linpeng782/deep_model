"""
深度模型项目的配置文件
包含当前需要的基本配置
"""

# 数据处理配置
DATA_CONFIG = {
    # 数据更新配置
    "end_date": "20250818",  # 数据结束日期
    
    # 并行处理配置
    "max_workers": 4,  # 最大并行工作进程数
    "use_parallel": True,  # 是否使用并行处理
    
    # 股票代码转换映射
    "code_mapping": {
        ".SZ": ".XSHE",  # 深交所
        ".SH": ".XSHG",  # 上交所
        ".BJ": ".BJSE",  # 北交所
    },
}

# 回测参数配置
BACKTEST_CONFIG = {
    "holding_months": 12,  # 持仓月数
    "initial_capital": 10000 * 10000,  # 初始资金（1亿）
    "benchmark_index": "000985.XSHG",  # 中证全指
}

def get_config(config_name):
    """
    获取指定配置
    
    参数:
        config_name: 配置名称，如 'data', 'backtest'
    
    返回:
        配置字典
    """
    config_map = {
        'data': DATA_CONFIG,
        'backtest': BACKTEST_CONFIG,
    }
    
    return config_map.get(config_name.lower(), {})

def get_timestamped_output_path(base_path, prefix="enhanced_factors_csv"):
    """
    生成带时间戳的输出路径
    
    参数:
        base_path: 基础路径
        prefix: 文件夹前缀
    
    返回:
        带时间戳的完整路径
    """
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d')
    base_output_dir = os.path.dirname(base_path)
    output_folder_name = f"{prefix}_{timestamp}"
    return os.path.join(base_output_dir, output_folder_name)

if __name__ == "__main__":
    # 测试配置获取
    print("数据配置:")
    print(get_config('data'))
    
    print("\n回测配置:")
    print(get_config('backtest'))
