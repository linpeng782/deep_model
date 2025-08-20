#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门检查特定文件的差异详情
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import ENHANCED_DATA_DIR

def detailed_compare_files(file1_path, file2_path):
    """
    详细比较两个CSV文件的差异
    """
    print(f"详细比较文件: {Path(file1_path).name}")
    print("=" * 80)
    
    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        df2 = pd.read_csv(file2_path, encoding='utf-8')
        
        print(f"文件1形状: {df1.shape}")
        print(f"文件2形状: {df2.shape}")
        print(f"列名一致: {list(df1.columns) == list(df2.columns)}")
        
        if df1.shape != df2.shape:
            print("❌ 文件形状不一致")
            return
        
        if list(df1.columns) != list(df2.columns):
            print("❌ 列名不一致")
            return
        
        # 检查数值列的差异
        numeric_cols = df1.select_dtypes(include=['number']).columns
        print(f"\n数值列数量: {len(numeric_cols)}")
        print(f"数值列: {list(numeric_cols)}")
        
        total_differences = 0
        total_values = 0
        
        for col in numeric_cols:
            # 找出有差异的行
            valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
            col_diff = valid_mask & ~np.isclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True)
            
            if col_diff.any():
                diff_count = col_diff.sum()
                total_differences += diff_count
                total_values += len(df1[col])
                
                print(f"\n列 '{col}' 有 {diff_count} 个差异:")
                diff_indices = df1[col_diff].index[:10]  # 显示前10个差异
                
                for idx in diff_indices:
                    val1 = df1.loc[idx, col]
                    val2 = df2.loc[idx, col]
                    diff_val = abs(val1 - val2)
                    print(f"  行 {idx}: {val1} vs {val2} (差值: {diff_val:.15f})")
            else:
                total_values += len(df1[col])
        
        if total_differences == 0:
            print("\n✅ 所有数值列完全一致")
        else:
            print(f"\n⚠️ 总差异数: {total_differences}")
            print(f"总数值数: {total_values}")
            print(f"差异比例: {total_differences/total_values:.10f}")
        
        # 检查非数值列
        non_numeric_cols = df1.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"\n非数值列: {list(non_numeric_cols)}")
            for col in non_numeric_cols:
                if not df1[col].equals(df2[col]):
                    print(f"  列 '{col}' 有差异")
                    diff_mask = df1[col] != df2[col]
                    if diff_mask.any():
                        diff_indices = df1[diff_mask].index[:5]
                        for idx in diff_indices:
                            print(f"    行 {idx}: '{df1.loc[idx, col]}' vs '{df2.loc[idx, col]}'")
                else:
                    print(f"  列 '{col}' 完全一致")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

def check_known_differences(folder1=None, folder2=None, files=None):
    """
    检查已知有差异的文件
    
    Args:
        folder1 (str): 第一个文件夹路径
        folder2 (str): 第二个文件夹路径
        files (list): 要检查的文件列表，默认使用已知有差异的文件
    """
    if files is None:
        # 根据全面扫描的结果，这些文件有差异
        diff_files = [
            "871694.BJSE-中裕科技-日线后复权及常用指标-20250818.csv",
            "300452.XSHE-山河药辅-日线后复权及常用指标-20250818.csv",
            "688665.XSHG-四方光电-日线后复权及常用指标-20250818.csv",
            "301387.XSHE-光大同创-日线后复权及常用指标-20250818.csv",
            "603893.XSHG-瑞芯微-日线后复权及常用指标-20250818.csv",
            "831087.BJSE-秋乐种业-日线后复权及常用指标-20250818.csv",
            "002299.XSHE-圣农发展-日线后复权及常用指标-20250818.csv",
            "835174.BJSE-五新遂装-日线后复权及常用指标-20250818.csv",
        ]
    else:
        diff_files = files
    
    if folder1 is None:
        original_folder = ENHANCED_DATA_DIR
    else:
        original_folder = folder1
        
    if folder2 is None:
        new_folder = ENHANCED_DATA_DIR + "_20250819"
    else:
        new_folder = folder2
    
    for filename in diff_files:
        file1_path = os.path.join(original_folder, filename)
        file2_path = os.path.join(new_folder, filename)
        
        if os.path.exists(file1_path) and os.path.exists(file2_path):
            detailed_compare_files(file1_path, file2_path)
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"文件不存在: {filename}")
            if not os.path.exists(file1_path):
                print(f"  原始文件不存在: {file1_path}")
            if not os.path.exists(file2_path):
                print(f"  新文件不存在: {file2_path}")

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description='检查特定文件的差异详情')
    parser.add_argument('--folder1', type=str, help='第一个文件夹路径（相对于enhanced目录）')
    parser.add_argument('--folder2', type=str, help='第二个文件夹路径（相对于enhanced目录）')
    parser.add_argument('--files', type=str, nargs='+', help='要检查的文件列表')
    
    args = parser.parse_args()
    
    # 如果提供了参数，构建完整路径
    folder1_path = None
    folder2_path = None
    
    if args.folder1:
        folder1_path = os.path.join(os.path.dirname(ENHANCED_DATA_DIR), args.folder1)
        print(f"第一个文件夹: {folder1_path}")
        
    if args.folder2:
        folder2_path = os.path.join(os.path.dirname(ENHANCED_DATA_DIR), args.folder2)
        print(f"第二个文件夹: {folder2_path}")
    
    check_known_differences(folder1_path, folder2_path, args.files)
    
if __name__ == "__main__":
    main()
