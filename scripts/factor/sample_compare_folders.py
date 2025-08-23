#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽样比较两个CSV文件夹的内容，验证重构后代码的正确性
"""

import os
import pandas as pd
import numpy as np
import random
import argparse
from pathlib import Path
import sys

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import ENHANCED_DATA_DIR

def get_csv_files_mapping(folder1, folder2):
    """
    获取两个文件夹中对应的CSV文件映射
    
    Args:
        folder1 (str): 第一个文件夹路径
        folder2 (str): 第二个文件夹路径
    
    Returns:
        list: 包含对应文件路径的元组列表
    """
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    
    if not folder1_path.exists():
        print(f"文件夹不存在: {folder1}")
        return []
    
    if not folder2_path.exists():
        print(f"文件夹不存在: {folder2}")
        return []
    
    # 获取文件夹1中的所有CSV文件
    csv_files1 = {f.name: f for f in folder1_path.glob("*.csv")}
    csv_files2 = {f.name: f for f in folder2_path.glob("*.csv")}
    
    # 找到两个文件夹中都存在的文件
    common_files = []
    for filename in csv_files1.keys():
        if filename in csv_files2:
            common_files.append((csv_files1[filename], csv_files2[filename]))
    
    print(f"文件夹1中的文件数: {len(csv_files1)}")
    print(f"文件夹2中的文件数: {len(csv_files2)}")
    print(f"共同文件数: {len(common_files)}")
    
    return common_files

def compare_csv_files(file1_path, file2_path):
    """
    比较两个CSV文件的内容
    
    Args:
        file1_path (Path): 第一个CSV文件路径
        file2_path (Path): 第二个CSV文件路径
    
    Returns:
        dict: 比较结果
    """
    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        df2 = pd.read_csv(file2_path, encoding='utf-8')
        
        result = {
            'filename': file1_path.name,
            'file1_shape': df1.shape,
            'file2_shape': df2.shape,
            'shapes_match': df1.shape == df2.shape,
            'columns_match': list(df1.columns) == list(df2.columns),
            'data_identical': False,
            'sample_differences': []
        }
        
        # 如果形状和列名都匹配，检查数据内容
        if result['shapes_match'] and result['columns_match']:
            # 比较数值列的内容（排除字符串列）
            numeric_cols = df1.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                # 对数值列进行近似比较（考虑浮点数精度）
                differences = 0
                total_values = 0
                
                for col in numeric_cols:
                    # 使用 numpy.isclose 进行近似比较
                    valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
                    col_diff = valid_mask & ~np.isclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True)
                    differences += col_diff.sum()
                    total_values += len(df1[col])
                
                result['data_identical'] = differences == 0
                result['difference_ratio'] = differences / total_values if total_values > 0 else 0
                
                # 记录前几个差异示例
                if differences > 0:
                    diff_examples = []
                    for col in numeric_cols[:3]:  # 只检查前3列
                        valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
                        col_diff = valid_mask & ~np.isclose(df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True)
                        if col_diff.any():
                            diff_indices = df1[col_diff].index[:3]  # 只取前3个差异
                            for idx in diff_indices:
                                diff_examples.append({
                                    'column': col,
                                    'row': idx,
                                    'value1': df1.loc[idx, col],
                                    'value2': df2.loc[idx, col]
                                })
                    result['sample_differences'] = diff_examples
            else:
                # 如果没有数值列，直接比较所有内容
                result['data_identical'] = df1.equals(df2)
        
        return result
        
    except Exception as e:
        return {
            'filename': file1_path.name,
            'error': str(e),
            'file1_shape': None,
            'file2_shape': None,
            'shapes_match': False,
            'columns_match': False,
            'data_identical': False
        }

def sample_compare_folders(folder1, folder2, sample_size=500):
    """
    抽样比较两个文件夹的CSV文件
    
    Args:
        folder1 (str): 原始文件夹路径
        folder2 (str): 新生成文件夹路径
        sample_size (int): 抽样数量
    """
    print(f"开始抽样比较两个文件夹:")
    print(f"文件夹1: {folder1}")
    print(f"文件夹2: {folder2}")
    print(f"抽样数量: {sample_size}")
    print("=" * 60)
    
    # 获取文件映射
    common_files = get_csv_files_mapping(folder1, folder2)
    
    if not common_files:
        print("没有找到共同的文件进行比较")
        return
    
    # 抽样
    sample_files = random.sample(common_files, min(sample_size, len(common_files)))
    print(f"实际抽样数量: {len(sample_files)}")
    print("=" * 60)
    
    # 比较结果统计
    total_files = len(sample_files)
    identical_files = 0
    shape_mismatch = 0
    column_mismatch = 0
    data_mismatch = 0
    error_files = 0
    
    # 逐个比较文件
    for i, (file1, file2) in enumerate(sample_files, 1):
        print(f"比较进度: {i}/{total_files} - {file1.name}")
        
        result = compare_csv_files(file1, file2)
        
        if 'error' in result:
            error_files += 1
            print(f"  ❌ 错误: {result['error']}")
        elif result['data_identical']:
            identical_files += 1
            print(f"  ✅ 完全一致")
        else:
            if not result['shapes_match']:
                shape_mismatch += 1
                print(f"  ⚠️  形状不匹配: {result['file1_shape']} vs {result['file2_shape']}")
            elif not result['columns_match']:
                column_mismatch += 1
                print(f"  ⚠️  列名不匹配")
            else:
                data_mismatch += 1
                print(f"  ⚠️  数据内容不同 (差异比例: {result.get('difference_ratio', 0):.6f})")
                if result['sample_differences']:
                    print(f"    详细差异信息:")
                    for diff in result['sample_differences'][:5]:  # 显示前5个差异
                        print(f"      列: {diff['column']}, 行: {diff['row']}, 值1: {diff['value1']}, 值2: {diff['value2']}, 差值: {abs(diff['value1'] - diff['value2']):.10f}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("比较结果总结:")
    print(f"总文件数: {total_files}")
    print(f"完全一致: {identical_files} ({identical_files/total_files*100:.1f}%)")
    print(f"形状不匹配: {shape_mismatch} ({shape_mismatch/total_files*100:.1f}%)")
    print(f"列名不匹配: {column_mismatch} ({column_mismatch/total_files*100:.1f}%)")
    print(f"数据不匹配: {data_mismatch} ({data_mismatch/total_files*100:.1f}%)")
    print(f"错误文件: {error_files} ({error_files/total_files*100:.1f}%)")
    
    # 判断重构是否成功
    success_rate = identical_files / total_files * 100
    if success_rate >= 95:
        print(f"\n🎉 重构验证成功！一致性达到 {success_rate:.1f}%")
    elif success_rate >= 90:
        print(f"\n⚠️  重构基本成功，但存在少量差异。一致性: {success_rate:.1f}%")
    else:
        print(f"\n❌ 重构可能存在问题，一致性仅: {success_rate:.1f}%")

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(description='抽样比较两个文件夹中CSV文件的内容')
    parser.add_argument('--folder1', type=str, help='第一个文件夹路径（相对于enhanced目录）')
    parser.add_argument('--folder2', type=str, help='第二个文件夹路径（相对于enhanced目录）')
    parser.add_argument('--sample-size', type=int, default=500, help='抽样数量（默认500）')
    
    args = parser.parse_args()
    
    # 如果提供了参数，构建完整路径
    if args.folder1 and args.folder2:
        folder1_path = os.path.join(os.path.dirname(ENHANCED_DATA_DIR), args.folder1)
        folder2_path = os.path.join(os.path.dirname(ENHANCED_DATA_DIR), args.folder2)
        print(f"第一个文件夹: {folder1_path}")
        print(f"第二个文件夹: {folder2_path}")
    else:
        # 使用默认路径
        folder1_path = ENHANCED_DATA_DIR  # 原始的enhanced_factors_csv文件夹
        folder2_path = ENHANCED_DATA_DIR + "_20250819"  # 新生成的文件夹
        print("使用默认路径:")
        print(f"原始文件夹: {folder1_path}")
        print(f"新生成文件夹: {folder2_path}")
    
    # 执行抽样比较
    sample_compare_folders(folder1_path, folder2_path, sample_size=args.sample_size)
    
if __name__ == "__main__":
    main()
