import pandas as pd
import numpy as np
import os
import re  # 用于正则表达式匹配，提取股票代码
import sys

sys.path.insert(0, "/Users/didi/KDCJ")
from factor_utils import *


def get_stock_data(file_path):
    """从指定目录加载所有股票数据的CSV文件。

    参数:
        file_path: 包含股票CSV文件的目录

    返回:
        一个以日期时间和股票代码为索引的MultiIndex DataFrame
    """
    # 初始化存储所有DataFrame的列表
    all_dfs = []

    # 获取目录中的所有文件
    all_files = os.listdir(file_path)

    # 筛选出CSV文件并构建完整路径
    csv_files = [
        os.path.join(file_path, f) for f in all_files if f.lower().endswith(".csv")
    ]

    # 为了测试，只取前3个文件
    # csv_files = csv_files[:10]

    # 记录CSV文件总数，用于显示处理进度
    total_files = len(csv_files)
    print(f"开始处理前 {total_files} 个CSV文件（测试模式）")

    # 遍历所有CSV文件进行处理
    for i, file_path in enumerate(csv_files):
        # 使用正则表达式从文件名中提取股票代码
        # 匹配格式：6位数字.交易所代码（如：000001.XSHE, 000001.XSHG, 430047.BJSE）
        match = re.search(r"(\d{6}\.(XSHE|XSHG|BJSE))", os.path.basename(file_path))

        if not match:
            print(f"跳过文件 {file_path} - 无法提取股票代码")
            continue

        # 获取匹配到的股票代码
        stock_code = match.group(1)
        print(f"正在处理文件：{os.path.basename(file_path)} (股票代码: {stock_code})")
        # 尝试读取CSV文件
        try:
            # 读取CSV文件到DataFrame
            df = pd.read_csv(file_path)

            # 将交易日期列从字符串格式（YYYYMMDD）转换为datetime类型
            df["交易日期"] = pd.to_datetime(df["交易日期"], format="%Y%m%d")
            start_date = df["交易日期"].min()
            end_date = df["交易日期"].max()

            # 添加未复权的开盘价
            open_price = get_price(
                stock_code,
                start_date,
                end_date,
                fields=["open"],
                adjust_type="none",
                skip_suspended=False,
            )

            # 将开盘价数据合并到原始DataFrame中
            # 首先重置open_price的索引，将日期索引转为列
            open_price_reset = open_price.reset_index()
            open_price_reset.columns = ["股票代码", "交易日期", "未复权开盘价"]
            df = df.merge(open_price_reset, on=["股票代码", "交易日期"], how="left")

            # 将处理好的DataFrame添加到列表中
            all_dfs.append(df)

            # 每处理10个文件或处理完所有文件时打印进度
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"已处理 {i+1}/{total_files} 个文件")

        except Exception as e:
            # 捕获并报告文件处理过程中的任何错误
            print(f"处理文件 {file_path} 时出错: {e}")

    # 检查是否成功读取了任何有效的数据文件
    if not all_dfs:
        raise ValueError("未找到有效的数据文件")

    # 将所有DataFrame垂直合并成一个大的DataFrame
    # ignore_index=True 重新生成连续的行索引
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 首先按股票代码和交易日期排序
    sorted_df = combined_df.sort_values(["股票代码", "交易日期"])

    # 设置股票代码和交易日期为多级索引
    multi_index_df = sorted_df.set_index(["股票代码", "交易日期"])

    # 给索引添加名称
    multi_index_df.index.names = ["order_book_id", "datetime"]

    return multi_index_df


def get_data_source_path(file_name):
    """
    获取数据源目录路径

    Returns:
        str: 数据源目录的完整路径
    """
    # 统一的数据源目录
    data_source_dir = "/Users/didi/KDCJ/deep_model/data/enhanced/" + file_name

    return data_source_dir


def get_output_file_path(file_name, output_dir=None):
    """
    生成输出文件的完整路径

    Args:
        file_name (str): 数据源目录路径
        output_dir (str, optional): 输出目录，默认使用统一目录

    Returns:
        str: 输出文件的完整路径
    """
    import os

    # 默认输出目录
    if output_dir is None:
        output_dir = "/Users/didi/KDCJ/deep_model/data/cache/bars"

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 从数据源目录名提取日期信息来生成文件名
    # 使用正则表达式提取日期（假设格式为 enhanced_factors_csv_YYYYMMDD）
    import re

    date_pattern = r"(\d{8})"
    match = re.search(date_pattern, file_name)

    if match:
        date_str = match.group(1)
        output_filename = f"{date_str}_bars_df.pkl"
    else:
        # 如果无法提取日期，使用原始文件名
        print(f"警告：无法从文件名 '{file_name}' 中提取日期，使用原始格式")
        output_filename = f"{file_name}_bars_df.pkl"
    output_file = os.path.join(output_dir, output_filename)

    return output_file


# 主程序入口：当直接运行此脚本时执行以下代码
if __name__ == "__main__":
    import os

    # 获取数据源目录路径
    file_name = "enhanced_factors_csv_20250819"
    data_source_dir = get_data_source_path(file_name)

    # 加载并合并所有股票数据
    print("开始加载股票数据...")
    bars_df = get_stock_data(data_source_dir)

    # 获取输出文件路径
    output_file = get_output_file_path(file_name)
    print(f"输出文件名: {os.path.basename(output_file)}")
    print(f"输出文件: {output_file}")

    # 保存结果
    bars_df.to_pickle(output_file)
    print(f"股票数据已保存到: {output_file}")
    print(f"数据形状: {bars_df.shape}")
    print(f"股票数量: {len(bars_df.index.get_level_values('order_book_id').unique())}")
