#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出所有有差异的文件并详细分析
"""

import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
import sys

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import ENHANCED_DATA_DIR

def compare_csv_files(file1_path, file2_path):
    """
    比较两个CSV文件的内容
    """
    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        df2 = pd.read_csv(file2_path, encoding='utf-8')
        
        if df1.shape != df2.shape or list(df1.columns) != list(df2.columns):
            return False, "形状或列名不匹配"
        
        # 比较数值列的内容
        numeric_cols = df1.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            differences = 0
            total_values = 0
            
            for col in numeric_cols:
                valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
                col_diff = valid_mask & ~np.isclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True)
                differences += col_diff.sum()
                total_values += len(df1[col])
            
            if differences > 0:
                return False, f"数值差异: {differences}/{total_values} ({differences/total_values:.6f})"
        
        # 检查非数值列
        non_numeric_cols = df1.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            if not df1[col].equals(df2[col]):
                return False, f"非数值列 '{col}' 有差异"
        
        return True, "完全一致"
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def find_all_differences():
    """
    找出所有有差异的文件
    """
    original_folder = ENHANCED_DATA_DIR
    new_folder = ENHANCED_DATA_DIR + "_20250819"
    
    original_path = Path(original_folder)
    new_path = Path(new_folder)
    
    if not original_path.exists() or not new_path.exists():
        print("文件夹不存在")
        return
    
    # 获取所有CSV文件
    original_files = {f.name: f for f in original_path.glob("*.csv")}
    new_files = {f.name: f for f in new_path.glob("*.csv")}
    
    # 找到共同文件
    common_files = []
    for filename in original_files.keys():
        if filename in new_files:
            common_files.append(filename)
    
    print(f"总共找到 {len(common_files)} 个共同文件")
    
    # 检查所有文件的差异
    different_files = []
    identical_files = 0
    
    for i, filename in enumerate(common_files, 1):
        if i % 100 == 0:
            print(f"检查进度: {i}/{len(common_files)}")
        
        file1_path = original_files[filename]
        file2_path = new_files[filename]
        
        is_identical, message = compare_csv_files(file1_path, file2_path)
        
        if not is_identical:
            different_files.append((filename, message))
        else:
            identical_files += 1
    
    print(f"\n检查完成!")
    print(f"完全一致的文件: {identical_files}")
    print(f"有差异的文件: {len(different_files)}")
    
    if different_files:
        print(f"\n有差异的文件列表:")
        for filename, message in different_files:
            print(f"  {filename}: {message}")
    
    return different_files

if __name__ == "__main__":
    different_files = find_all_differences()
