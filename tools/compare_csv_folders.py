#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比新老CSV文件夹，找出缺失的股票
"""

import os
import re
from pathlib import Path

def parse_stock_code_from_filename(filename):
    """从文件名中提取股票代码"""
    # 匹配格式：000001.SZ-平安银行-日线后复权及常用指标-20250718.csv
    pattern = r"^(\d{6}\.[A-Z]{2,4})-(.+?)-日线后复权及常用指标-(\d{8})\.csv$"
    match = re.match(pattern, filename)
    if match:
        return match.group(1)  # 返回股票代码
    return None

def convert_stock_code(original_code):
    """将原始股票代码转换为米筐格式"""
    if original_code.endswith('.SZ'):
        return original_code.replace('.SZ', '.XSHE')
    elif original_code.endswith('.SH'):
        return original_code.replace('.SH', '.XSHG')
    elif original_code.endswith('.BJ'):
        return original_code.replace('.BJ', '.XBSE')
    else:
        return original_code

def get_stock_codes_from_folder(folder_path):
    """从文件夹中获取所有股票代码"""
    folder = Path(folder_path)
    stock_codes = set()
    
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return stock_codes
    
    for csv_file in folder.glob("*.csv"):
        stock_code = parse_stock_code_from_filename(csv_file.name)
        if stock_code:
            stock_codes.add(stock_code)
    
    return stock_codes

def compare_folders():
    """对比新老文件夹"""
    # 文件夹路径
    old_folder = "/Users/didi/KDCJ/deep_model/backtest/日线后复权及常用指标csv"
    new_folder = "/Users/didi/KDCJ/deep_model/enhanced_factors_csv"
    
    print("=== CSV文件夹对比分析 ===")
    print(f"原始文件夹: {old_folder}")
    print(f"新文件夹: {new_folder}")
    
    # 获取股票代码集合
    old_codes = get_stock_codes_from_folder(old_folder)
    new_codes_raw = get_stock_codes_from_folder(new_folder)
    
    # 将新文件夹的米筐格式代码转换回原始格式进行对比
    new_codes = set()
    for code in new_codes_raw:
        if code.endswith('.XSHE'):
            new_codes.add(code.replace('.XSHE', '.SZ'))
        elif code.endswith('.XSHG'):
            new_codes.add(code.replace('.XSHG', '.SH'))
        elif code.endswith('.BJSE'):
            new_codes.add(code.replace('.BJSE', '.BJ'))
        else:
            new_codes.add(code)
    
    print(f"\n原始文件夹股票数量: {len(old_codes)}")
    print(f"新文件夹股票数量: {len(new_codes)}")
    
    # 找出缺失的股票
    missing_stocks = old_codes - new_codes
    extra_stocks = new_codes - old_codes
    
    print(f"\n缺失的股票数量: {len(missing_stocks)}")
    if missing_stocks:
        print("缺失的股票代码:")
        for i, code in enumerate(sorted(missing_stocks), 1):
            print(f"  {i:3d}. {code}")
            if i >= 50:  # 只显示前50个
                print(f"  ... 还有 {len(missing_stocks) - 50} 只股票")
                break
    
    print(f"\n多余的股票数量: {len(extra_stocks)}")
    if extra_stocks:
        print("多余的股票代码:")
        for i, code in enumerate(sorted(extra_stocks), 1):
            print(f"  {i:3d}. {code}")
            if i >= 20:  # 只显示前20个
                print(f"  ... 还有 {len(extra_stocks) - 20} 只股票")
                break
    
    # 计算成功率
    success_rate = len(new_codes) / len(old_codes) * 100 if old_codes else 0
    print(f"\n处理成功率: {success_rate:.2f}%")
    
    # 保存缺失股票列表到文件
    if missing_stocks:
        missing_file = "/Users/didi/KDCJ/deep_model/missing_stocks.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write("缺失的股票代码列表:\n")
            for code in sorted(missing_stocks):
                f.write(f"{code}\n")
        print(f"\n缺失股票列表已保存到: {missing_file}")
    
    return missing_stocks, extra_stocks

if __name__ == "__main__":
    missing, extra = compare_folders()
