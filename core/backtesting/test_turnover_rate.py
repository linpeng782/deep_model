#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据order_book_ids从日线后复权及常用指标csv文件夹中读取对应的CSV文件
"""

import os
import glob
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from tqdm import *
from joblib import Parallel, delayed
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta

from rqdatac import *
from rqfactor import *
from rqfactor import Factor
from rqfactor.extension import *


init("13522652015", "123456")
import rqdatac

import seaborn as sns
import matplotlib.pyplot as plt

# 关闭通知
import warnings

warnings.filterwarnings("ignore")
import logging


def read_stock_csv_files(
    order_book_ids: List[str], csv_folder_path: str = "日线后复权及常用指标csv"
) -> Dict[str, pd.DataFrame]:
    """
    根据order_book_ids读取对应的CSV文件

    参数:
        order_book_ids: 股票代码列表，如['000001.XSHE','600000.XSHG']
        csv_folder_path: CSV文件夹路径

    返回:
        字典，键为股票代码，值为对应的DataFrame
    """

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_folder_full_path = os.path.join(current_dir, csv_folder_path)

    if not os.path.exists(csv_folder_full_path):
        raise FileNotFoundError(f"CSV文件夹不存在: {csv_folder_full_path}")

    # 存储结果的字典
    stock_data = {}

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(csv_folder_full_path, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 处理每个order_book_id
    for order_book_id in order_book_ids:
        # 提取股票代码（去掉交易所后缀）
        stock_code = order_book_id.split(".")[0]
        print(f"\n正在查找股票代码: {stock_code} (来自 {order_book_id})")

        # 查找匹配的CSV文件
        matching_files = []
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            # 检查文件名是否以股票代码开头
            if filename.startswith(stock_code + "."):
                matching_files.append(csv_file)

        if matching_files:
            if len(matching_files) > 1:
                print(f"  警告: 找到多个匹配文件:")
                for file in matching_files:
                    print(f"    - {os.path.basename(file)}")
                print(f"  使用第一个文件: {os.path.basename(matching_files[0])}")
            else:
                print(f"  找到匹配文件: {os.path.basename(matching_files[0])}")

            # 读取CSV文件
            try:
                df = pd.read_csv(matching_files[0])
                stock_data[order_book_id] = df
                print(f"  成功读取，数据形状: {df.shape}")

            except Exception as e:
                print(f"  错误: 读取文件失败 - {str(e)}")
        else:
            print(f"  未找到匹配的CSV文件")

    print(f"\n总共成功读取了 {len(stock_data)} 个股票的数据")
    return stock_data


def main():
    """主函数，演示如何使用"""
    # 测试用的order_book_ids
    order_book_ids = ["000001.XSHE", "600000.XSHG"]

    print("开始读取股票CSV文件...")
    print(f"目标股票代码: {order_book_ids}")

    try:
        # 读取CSV文件
        stock_data = read_stock_csv_files(order_book_ids)

        # 显示读取结果
        print("\n=== 读取结果汇总 ===")
        for order_book_id, df in stock_data.items():
            print(f"\n股票代码: {order_book_id}")
            print(f"数据行数: {len(df)}")
            print(f"数据列数: {len(df.columns)}")

            # 显示前5行数据
            print("前5行数据:")
            print(df.head())

    except Exception as e:
        print(f"执行出错: {str(e)}")


if __name__ == "__main__":
    main()

    order_book_ids = [
        "000001.XSHE",
        "000002.XSHE",
        "000008.XSHE",
        "000009.XSHE",
        "000017.XSHE",
        "000066.XSHE",
        "000089.XSHE",
        "300024.XSHE",
        "300113.XSHE",
        "600000.XSHG",
        "600004.XSHG",
        "600007.XSHG",
        "600017.XSHG",
        "600028.XSHG",
        "600900.XSHG",
        "600905.XSHG",
        "600916.XSHG",
    ]
    start_date = "2025-01-01"
    end_date = "2025-07-01"

    stock_data = read_stock_csv_files(order_book_ids)

    turnover_comparison = pd.DataFrame()
    for stock_code in order_book_ids:
        if stock_code in stock_data:
            df = stock_data[stock_code].copy()
            df["交易日期"] = pd.to_datetime(df["交易日期"], format="%Y%m%d")
            df = df.set_index("交易日期")
            df_filtered = df.loc[start_date:end_date]
            turnover_comparison[stock_code] = df_filtered["换手率(%)"]

    # turnover_comparison.index.name = turnover_ratio_rq.index.name

    turnover_ratio_rq = get_turnover_rate(
        order_book_ids, start_date, end_date
    ).today.unstack("order_book_id")

    # 逐个股票对比换手率数据
    for stock_code in order_book_ids:
        print(f"\n=== 对比股票: {stock_code} ===")

        # 直接对比指定日期范围的数据
        api_data = turnover_ratio_rq[stock_code].loc[start_date:end_date]
        csv_data = turnover_comparison[stock_code].loc[start_date:end_date]

        # 检查数据是否相等
        is_equal = api_data.equals(csv_data)
        print(f"数据是否完全相等: {is_equal}")

        if not is_equal:
            print(f"API数据样本: {api_data.head(3).values}")
            print(f"CSV数据样本: {csv_data.head(3).values}")
