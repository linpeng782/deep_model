"""
数据工具模块
负责数据解析、转换和文件管理等通用功能
"""

import os
import re
from pathlib import Path
from datetime import datetime


def parse_stock_info_from_filename(filename):
    """
    从文件名解析股票信息
    输入: "000001.SZ-股票名称-日线后复权及常用指标-20250718.csv"
    输出: ("000001.SZ", "股票名称", "20250718")
    """
    pattern = r"([0-9]{6}\.[A-Z]{2})-(.+?)-日线后复权及常用指标-(\d{8})\.csv"
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def convert_stock_code(original_code):
    """
    转换股票代码格式
    SZ -> XSHE, SH -> XSHG, BJ -> BJSE
    """
    if original_code.endswith(".SZ"):
        return original_code.replace(".SZ", ".XSHE")
    elif original_code.endswith(".SH"):
        return original_code.replace(".SH", ".XSHG")
    elif original_code.endswith(".BJ"):
        return original_code.replace(".BJ", ".BJSE")
    else:
        return None  # 其他格式暂不处理


def get_stock_list_from_csv_folder(csv_folder_path, limit=None):
    """
    从CSV文件夹获取股票列表
    """
    csv_folder = Path(csv_folder_path)
    stock_list = []

    for csv_file in csv_folder.glob("*.csv"):
        original_code, stock_name, date = parse_stock_info_from_filename(csv_file.name)

        if original_code and stock_name:
            # 转换股票代码
            converted_code = convert_stock_code(original_code)

            if converted_code:  # 处理SZ、SH和BJ股票
                stock_list.append(
                    {
                        "original_code": original_code,
                        "converted_code": converted_code,
                        "stock_name": stock_name,
                        "date": date,
                    }
                )

    # 限制处理数量（用于测试）
    if limit:
        stock_list = stock_list[:limit]

    return stock_list


def save_failed_stocks(failed_stocks, process_type):
    """
    保存失败股票信息到文件

    Args:
        failed_stocks (list): 失败股票列表
        process_type (str): 处理类型 (batch_process_parallel, batch_process_single, retry_failed)
    """
    # 创建数据目录
    data_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    data_dir.mkdir(exist_ok=True)

    # 生成文件名，包含时间戳和处理类型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_{process_type}_{timestamp}.txt"
    file_path = data_dir / filename

    # 写入失败股票信息
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"失败股票记录 - {process_type}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"失败总数: {len(failed_stocks)}\n")
        f.write("=" * 60 + "\n\n")

        for i, stock in enumerate(failed_stocks, 1):
            f.write(f"{i:3d}. 股票代码: {stock['stock_code']}\n")
            f.write(f"     股票名称: {stock['stock_name']}\n")
            f.write(f"     失败原因: {stock['error']}\n")
            f.write("-" * 40 + "\n")

    print(f"失败股票记录已保存到: {file_path}")


def get_failed_stocks(csv_folder_path, output_folder_path, end_date):
    """
    获取尚未创建输出CSV文件的股票列表
    """
    # 获取所有股票列表
    all_stocks = get_stock_list_from_csv_folder(csv_folder_path)

    # 获取已存在的输出CSV文件
    output_folder = Path(output_folder_path)
    existing_files = set()

    if output_folder.exists():
        for csv_file in output_folder.glob("*.csv"):
            existing_files.add(csv_file.name)

    # 找出尚未创建的股票
    failed_stocks = []
    for stock_info in all_stocks:
        expected_filename = f"{stock_info['converted_code']}-{stock_info['stock_name']}-日线后复权及常用指标-{end_date}.csv"
        if expected_filename not in existing_files:
            failed_stocks.append(stock_info)

    return failed_stocks
