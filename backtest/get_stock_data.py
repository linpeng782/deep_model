import pandas as pd
import numpy as np
import os
import re  # 用于正则表达式匹配，提取股票代码


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
        # 匹配格式：6位数字.2位大写字母（如：000001.SZ）
        match = re.search(r"(\d{6}\.[A-Z]{2})", os.path.basename(file_path))

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

            # 检查是否已存在股票代码列，如果不存在则添加
            # 这确保每行数据都有对应的股票代码标识
            if "股票代码" not in df.columns:
                df["股票代码"] = stock_code

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

    # 计算未复权开盘价：开盘价 / 复权因子
    # 这样可以获得未进行复权调整的原始开盘价
    if (
        "复权因子" in combined_df.columns
        and "开盘价" in combined_df.columns
        and "收盘价" in combined_df.columns
    ):
        combined_df["未复权开盘价"] = combined_df["开盘价"] / combined_df["复权因子"]
        combined_df["未复权收盘价"] = combined_df["收盘价"] / combined_df["复权因子"]
        print("已添加未复权开盘价和收盘价列")
    else:
        print("警告：未找到复权因子或开盘价列")

    # 创建多级索引DataFrame，股票代码为外层索引，日期为内层索引
    # 首先按股票代码和交易日期排序
    sorted_df = combined_df.sort_values(["股票代码", "交易日期"])

    # 设置股票代码和交易日期为多级索引
    multi_index_df = sorted_df.set_index(["股票代码", "交易日期"])

    # 给索引添加名称
    multi_index_df.index.names = ["order_book_id", "datetime"]

    return multi_index_df


# 主程序入口：当直接运行此脚本时执行以下代码
if __name__ == "__main__":

    # 定义包含股票数据CSV文件的目录路径
    # 这个目录应该包含多个股票的日线后复权数据文件
    price_post = "/Users/didi/Projects/BackTest/日线后复权及常用指标csv"
    save_path = "/Users/didi/Projects/BackTest/bars_df.pkl"

    # 调用函数加载并合并所有股票数据
    bars_df = get_stock_data(price_post)
    bars_df.to_pickle(save_path)
    print("bars_df.pkl已保存")
