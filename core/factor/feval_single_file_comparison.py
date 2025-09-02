#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门检查特定文件的差异详情
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from deep_model.config.paths import ENHANCED_DATA_DIR

# ==================== 快速配置区域 ====================
# 直接在这里设置要对比的两个文件路径，然后运行此文件即可
QUICK_COMPARE_FILE1 = "/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250902_test/000002.XSHE-万科A-日线后复权及常用指标-20250901.csv"
QUICK_COMPARE_FILE2 = "/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250902/000002.XSHE-万科A-日线后复权及常用指标-20250901.csv"

# 要对比的文件路径（修改这两个路径，然后运行此文件即可）
# ====================================================


def detailed_compare_files(file1_path, file2_path):
    """
    详细比较两个CSV文件的差异
    """
    print(f"详细比较文件: {Path(file1_path).name}")
    print("=" * 80)

    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path, encoding="utf-8")
        df2 = pd.read_csv(file2_path, encoding="utf-8")

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
        numeric_cols = df1.select_dtypes(include=["number"]).columns
        print(f"\n数值列数量: {len(numeric_cols)}")
        print(f"数值列: {list(numeric_cols)}")

        total_differences = 0
        total_values = 0

        for col in numeric_cols:
            # 找出有差异的行
            valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
            col_diff = valid_mask & ~np.isclose(
                df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True
            )

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
        non_numeric_cols = df1.select_dtypes(exclude=["number"]).columns
        if len(non_numeric_cols) > 0:
            print(f"\n非数值列: {list(non_numeric_cols)}")
            for col in non_numeric_cols:
                if not df1[col].equals(df2[col]):
                    print(f"  列 '{col}' 有差异")
                    diff_mask = df1[col] != df2[col]
                    if diff_mask.any():
                        diff_indices = df1[diff_mask].index[:5]
                        for idx in diff_indices:
                            print(
                                f"    行 {idx}: '{df1.loc[idx, col]}' vs '{df2.loc[idx, col]}'"
                            )
                else:
                    print(f"  列 '{col}' 完全一致")

    except Exception as e:
        print(f"❌ 错误: {str(e)}")


def main():
    """
    主函数，直接进行文件对比
    """
    print("=== 文件对比工具 ===")
    print(f"文件1: {QUICK_COMPARE_FILE1}")
    print(f"文件2: {QUICK_COMPARE_FILE2}")
    print()

    if os.path.exists(QUICK_COMPARE_FILE1) and os.path.exists(QUICK_COMPARE_FILE2):
        detailed_compare_files(QUICK_COMPARE_FILE1, QUICK_COMPARE_FILE2)
    else:
        print("❌ 文件不存在:")
        if not os.path.exists(QUICK_COMPARE_FILE1):
            print(f"  文件1不存在: {QUICK_COMPARE_FILE1}")
        if not os.path.exists(QUICK_COMPARE_FILE2):
            print(f"  文件2不存在: {QUICK_COMPARE_FILE2}")


if __name__ == "__main__":
    main()
