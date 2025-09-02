#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出所有有差异的文件并详细分析
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime


def compare_csv_files(file1_path, file2_path):
    """
    比较两个CSV文件的内容，自动截取相同的时间范围
    """
    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path, encoding="utf-8")
        df2 = pd.read_csv(file2_path, encoding="utf-8")

        # 检查列名是否一致
        if list(df1.columns) != list(df2.columns):
            return False, {
                "type": "column_mismatch",
                "overall_ratio": 1.8,  # 列名不匹配视为高优先级
            }

        # 如果有交易日期列，截取相同的时间范围
        if "交易日期" in df1.columns:
            # 找到两个数据集的共同日期范围
            dates1 = set(df1["交易日期"].astype(str))
            dates2 = set(df2["交易日期"].astype(str))
            common_dates = dates1.intersection(dates2)

            if len(common_dates) == 0:
                return False, {
                    "type": "no_common_dates",
                    "overall_ratio": 1.2,  # 没有共同日期视为高优先级
                }

            # 截取共同日期的数据
            df1 = df1[df1["交易日期"].astype(str).isin(common_dates)].reset_index(
                drop=True
            )
            df2 = df2[df2["交易日期"].astype(str).isin(common_dates)].reset_index(
                drop=True
            )

            # 按日期排序保证一致性
            df1 = df1.sort_values("交易日期").reset_index(drop=True)
            df2 = df2.sort_values("交易日期").reset_index(drop=True)

        # 现在检查形状是否一致
        if df1.shape != df2.shape:
            return False, {
                "type": "shape_mismatch",
                "shape1": df1.shape,
                "shape2": df2.shape,
                "overall_ratio": 1.5,  # 形状不匹配视为高优先级
            }

        # 比较数值列的内容
        numeric_cols = df1.select_dtypes(include=["number"]).columns

        if len(numeric_cols) > 0:
            differences = 0
            total_values = 0
            diff_details = []

            for col in numeric_cols:
                valid_mask = ~pd.isna(df1[col]) & ~pd.isna(df2[col])
                col_diff = valid_mask & ~np.isclose(
                    df1[col], df2[col], rtol=1e-10, atol=1e-10, equal_nan=True
                )
                col_diff_count = col_diff.sum()

                if col_diff_count > 0:
                    col_total = len(df1[col])
                    diff_ratio = col_diff_count / col_total
                    diff_details.append(
                        {
                            "column": col,
                            "diff_count": col_diff_count,
                            "total_count": col_total,
                            "diff_ratio": diff_ratio,
                        }
                    )

                differences += col_diff_count
                total_values += len(df1[col])

            if differences > 0:
                overall_ratio = differences / total_values
                return False, {
                    "type": "numeric_diff",
                    "overall_diff": differences,
                    "overall_total": total_values,
                    "overall_ratio": overall_ratio,
                    "column_details": diff_details,
                }

        # 检查非数值列
        non_numeric_cols = df1.select_dtypes(exclude=["number"]).columns
        for col in non_numeric_cols:
            if not df1[col].equals(df2[col]):
                return False, {
                    "type": "non_numeric_diff",
                    "column": col,
                    "overall_ratio": 1.0,  # 非数值列差异视为高优先级
                }

        return True, "完全一致"

    except Exception as e:
        return False, {
            "type": "error",
            "message": str(e),
            "overall_ratio": 2.0,  # 错误视为最高优先级
        }


def find_all_differences(folder1, folder2):
    """
    找出所有有差异的文件

    Args:
        folder1 (str): 第一个文件夹路径
        folder2 (str): 第二个文件夹路径
    """
    original_folder = folder1
    new_folder = folder2

    original_path = Path(original_folder)
    new_path = Path(new_folder)

    if not original_path.exists() or not new_path.exists():
        print("文件夹不存在")
        return

    # 获取所有CSV文件
    original_files = {f.name: f for f in original_path.glob("*.csv")}
    new_files = {f.name: f for f in new_path.glob("*.csv")}

    # 根据股票代码匹配文件（支持不同日期）
    def extract_stock_code(filename):
        """从文件名中提取股票代码"""
        # 文件名格式: 000001.XSHE-平安银行-日线后复权及常用指标-20250818.csv
        return filename.split("-")[0]  # 获取股票代码部分

    # 按股票代码分组
    original_files_list = list(original_path.glob("*.csv"))
    new_files_list = list(new_path.glob("*.csv"))

    print(f"第一个文件夹文件数: {len(original_files_list)}")
    print(f"第二个文件夹文件数: {len(new_files_list)}")

    original_by_code = {extract_stock_code(f.name): f for f in original_files_list}
    new_by_code = {extract_stock_code(f.name): f for f in new_files_list}

    print(f"第一个文件夹股票代码数: {len(original_by_code)}")
    print(f"第二个文件夹股票代码数: {len(new_by_code)}")

    # 找到共同股票代码的文件
    common_files = []
    for stock_code in original_by_code.keys():
        if stock_code in new_by_code:
            common_files.append(
                (original_by_code[stock_code].name, new_by_code[stock_code].name)
            )

    print(f"总共找到 {len(common_files)} 个共同文件")

    # 检查所有文件的差异
    different_files = []
    identical_files = 0

    for i, (filename1, filename2) in enumerate(common_files, 1):
        if i % 100 == 0:
            print(f"检查进度: {i}/{len(common_files)}")

        file1_path = original_files[filename1]
        file2_path = new_files[filename2]

        is_identical, result = compare_csv_files(file1_path, file2_path)

        if not is_identical:
            # 提取股票代码作为标识
            stock_code = extract_stock_code(filename1)
            different_files.append(
                {
                    "stock_code": stock_code,
                    "filename1": filename1,
                    "filename2": filename2,
                    "diff_info": result,
                }
            )
        else:
            identical_files += 1

    # 按差异程度排序（从大到小）
    if different_files:
        different_files.sort(
            key=lambda x: x["diff_info"].get("overall_ratio", 0), reverse=True
        )

    # 生成报告内容
    report_lines = []
    report_lines.append(f"米筐隔日数据下载对比报告")
    report_lines.append(f"=" * 60)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"第一个文件夹: {os.path.basename(original_folder)}")
    report_lines.append(f"第二个文件夹: {os.path.basename(new_folder)}")
    report_lines.append(f"第一个文件夹文件数: {len(original_files_list)}")
    report_lines.append(f"第二个文件夹文件数: {len(new_files_list)}")
    report_lines.append(f"共同股票代码数: {len(common_files)}")
    report_lines.append("")

    report_lines.append(f"检查结果汇总:")
    report_lines.append(
        f"完全一致的文件: {identical_files} ({identical_files/len(common_files)*100:.2f}%)"
    )
    report_lines.append(
        f"有差异的文件: {len(different_files)} ({len(different_files)/len(common_files)*100:.2f}%)"
    )
    report_lines.append("")

    # 添加差异指标说明文档
    report_lines.append("差异指标说明文档")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("总体差异 (Overall Difference)")
    report_lines.append("格式: 差异数据点/总数据点 (差异比例)")
    report_lines.append("示例: 2,186/33,500 (0.065254)")
    report_lines.append("含义:")
    report_lines.append("  - 2,186 = 整个CSV文件中有差异的数据点数量")
    report_lines.append("  - 33,500 = 整个CSV文件的总数据点数量 (行数 × 列数)")
    report_lines.append("  - 0.065254 = 差异比例 = 2,186 ÷ 33,500 = 6.5%")
    report_lines.append("")
    report_lines.append("具体差异列 (Column-wise Differences)")
    report_lines.append("格式: 列名: 差异数据点/该列总数据点 (该列差异比例)")
    report_lines.append("示例: 自由流通股本: 1,093/1,340 (0.815672)")
    report_lines.append("含义:")
    report_lines.append("  - 1,093 = 该列中有差异的数据点数量")
    report_lines.append("  - 1,340 = 该列的总数据点数量 (交易日数量)")
    report_lines.append("  - 0.815672 = 该列差异比例 = 1,093 ÷ 1,340 = 81.6%")
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("")

    # 输出到控制台
    print(f"\n检查完成!")
    print(f"完全一致的文件: {identical_files}")
    print(f"有差异的文件: {len(different_files)}")

    if different_files:
        # 先显示前5名差异最大的股票
        print("差异最大的前5名股票 (重点关注):")
        report_lines.append("差异最大的前5名股票 (重点关注):")
        report_lines.append("=" * 60)
        report_lines.append("")

        # 显示前5名（排除退市股票）
        non_delisted_files = [
            f for f in different_files if f["diff_info"]["type"] != "no_common_dates"
        ]
        top5_files = (
            non_delisted_files[:5]
            if len(non_delisted_files) >= 5
            else non_delisted_files
        )
        for i, diff_file in enumerate(top5_files, 1):
            diff_info = diff_file["diff_info"]

            # 控制台输出
            print(f"\n【TOP {i}】{diff_file['stock_code']}")

            # 报告输出
            report_lines.append(f"【TOP {i}】{diff_file['stock_code']}")

            if diff_info["type"] == "numeric_diff":
                console_output = f"  • 差异类型: 数值差异"
                print(console_output)
                report_lines.append(console_output)

                console_output = f"  • 总体差异: {diff_info['overall_diff']:,}/{diff_info['overall_total']:,} ({diff_info['overall_ratio']:.4f} = {diff_info['overall_ratio']*100:.2f}%)"
                print(console_output)
                report_lines.append(console_output)

                console_output = f"  • 主要差异列:"
                print(console_output)
                report_lines.append(console_output)

                # 只显示前3个差异最大的列
                sorted_cols = sorted(
                    diff_info["column_details"],
                    key=lambda x: x["diff_ratio"],
                    reverse=True,
                )[:3]
                for col_detail in sorted_cols:
                    console_output = f"    - {col_detail['column']}: {col_detail['diff_ratio']*100:.2f}% ({col_detail['diff_count']:,}/{col_detail['total_count']:,})"
                    print(console_output)
                    report_lines.append(console_output)

            elif diff_info["type"] == "no_common_dates":
                console_output = f"  • 差异类型: 无共同交易日期 (可能已退市)"
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "column_mismatch":
                console_output = f"  • 差异类型: 列名不匹配"
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "shape_mismatch":
                console_output = f"  • 差异类型: 形状不匹配"
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "error":
                console_output = f"  • 差异类型: 处理错误"
                print(console_output)
                report_lines.append(console_output)

            console_output = f"  • 文件: {diff_file['filename1']}"
            print(console_output)
            report_lines.append(console_output)

            console_output = f"         vs {diff_file['filename2']}"
            print(console_output)
            report_lines.append(console_output)
            report_lines.append("")

        print("\n" + "=" * 60)
        report_lines.append("=" * 60)
        report_lines.append("")

        # 再显示完整的排序列表
        print("所有差异文件完整列表 (按差异程度排序):")
        report_lines.append("所有差异文件完整列表 (按差异程度排序):")
        report_lines.append("=" * 60)

        for i, diff_file in enumerate(different_files, 1):
            diff_info = diff_file["diff_info"]

            # 控制台输出
            print(f"\n{i}. 股票代码: {diff_file['stock_code']}")

            # 报告输出
            report_lines.append(f"\n{i}. 股票代码: {diff_file['stock_code']}")

            if diff_info["type"] == "numeric_diff":
                console_output = f"    差异类型: 数值差异"
                print(console_output)
                report_lines.append(console_output)

                console_output = f"    总体差异: {diff_info['overall_diff']}/{diff_info['overall_total']} ({diff_info['overall_ratio']:.6f})"
                print(console_output)
                report_lines.append(console_output)

                console_output = f"    具体差异列:"
                print(console_output)
                report_lines.append(console_output)

                for col_detail in diff_info["column_details"]:
                    console_output = f"      - {col_detail['column']}: {col_detail['diff_count']}/{col_detail['total_count']} ({col_detail['diff_ratio']:.6f})"
                    print(console_output)
                    report_lines.append(console_output)

            elif diff_info["type"] == "column_mismatch":
                console_output = f"    差异类型: 列名不匹配"
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "shape_mismatch":
                console_output = f"    差异类型: 形状不匹配"
                print(console_output)
                report_lines.append(console_output)

                console_output = (
                    f"    形状: {diff_info['shape1']} vs {diff_info['shape2']}"
                )
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "no_common_dates":
                console_output = f"    差异类型: 没有共同的交易日期"
                print(console_output)
                report_lines.append(console_output)

            elif diff_info["type"] == "error":
                console_output = f"    差异类型: 错误"
                print(console_output)
                report_lines.append(console_output)

                console_output = f"    错误信息: {diff_info['message']}"
                print(console_output)
                report_lines.append(console_output)

            console_output = f"    文件: {diff_file['filename1']}"
            print(console_output)
            report_lines.append(console_output)

            console_output = f"          vs {diff_file['filename2']}"
            print(console_output)
            report_lines.append(console_output)

    # 保存报告到文件
    # 使用完整的文件夹名来命名
    folder1_name = os.path.basename(original_folder)
    folder2_name = os.path.basename(new_folder)

    # 从文件夹名中提取日期部分（找到8位数字）
    import re

    folder1_date_match = re.search(r"(\d{8})", folder1_name)
    folder2_date_match = re.search(r"(\d{8})", folder2_name)

    folder1_date = folder1_date_match.group(1) if folder1_date_match else "unknown"
    folder2_date = folder2_date_match.group(1) if folder2_date_match else "unknown"

    # 构建文件名：第二个文件夹_vs_第一个文件夹_report.txt
    report_filename = f"{folder2_date}_vs_{folder1_date}_report.txt"
    
    # 获取当前日期作为文件夹名
    current_date = datetime.now().strftime("%Y%m%d")
    
    # 创建带日期的报告目录
    report_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "comparison_reports", current_date
    )
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_filename)

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")
        print(f"\n对比报告已保存到: {report_path}")
    except Exception as e:
        print(f"\n保存报告时发生错误: {e}")

    return different_files


def main():
    """
    主函数，用户只需要修改这里的两个文件夹路径
    """
    # 用户输入：修改这两个路径
    folder1_path = (
        "/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250902"
    )
    folder2_path = (
        "/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250902_test"
    )

    print(f"第一个文件夹: {folder1_path}")
    print(f"第二个文件夹: {folder2_path}")

    different_files = find_all_differences(folder1_path, folder2_path)


if __name__ == "__main__":
    main()
