"""
深度模型项目的路径配置文件
统一管理所有数据和文件路径
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "日线后复权及常用指标csv")
ENHANCED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "enhanced", "enhanced_factors_csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# 报告输出路径
REPORTS_DIR = os.path.join(PROJECT_ROOT, "docs", "reports")

# 确保目录存在
def ensure_dirs():
    """确保所有必要的目录都存在"""
    dirs_to_create = [
        RAW_DATA_DIR,
        ENHANCED_DATA_DIR,
        CACHE_DIR,
        REPORTS_DIR
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 目录已确保存在: {dir_path}")

if __name__ == "__main__":
    ensure_dirs()
