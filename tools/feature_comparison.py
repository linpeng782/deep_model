#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征对比工具
用于对比原始CSV文件和enhanced文件夹中的特征数据一致性
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings("ignore")


class FeatureComparator:
    """特征对比器"""

    def __init__(self, original_folder, enhanced_folder):
        """
        初始化特征对比器

        Args:
            original_folder: 原始CSV文件夹路径
            enhanced_folder: enhanced文件夹路径
        """
        self.original_folder = Path(original_folder)
        self.enhanced_folder = Path(enhanced_folder)

        # 特征映射字典：{enhanced列名: 原始列名}
        self.feature_mapping = {
            "换手率": "换手率(%)",
            "成交量": "成交量(手)",
            "成交额": "成交额(千元)",
            "市盈率_TTM": "市盈率(TTM)",
            "市净率_TTM": "市净率",
            "市销率_TTM": "市销率(TTM)",
            "股息率_TTM": "股息率(TTM)(%)",
            "总市值": "总市值(万元)",
            "流通市值": "流通市值(万元)",
            "换手率(自由流动股)": "换手率(自由流通股)",
        }

    def load_stock_data(self, stock_code):
        """
        加载指定股票的原始和enhanced数据

        Args:
            stock_code: 股票代码，如'000001.XSHE'

        Returns:
            tuple: (original_df, enhanced_df)
        """
        # 转换股票代码格式
        if stock_code.endswith(".XSHE"):
            original_code = stock_code.replace(".XSHE", ".SZ")
        elif stock_code.endswith(".XSHG"):
            original_code = stock_code.replace(".XSHG", ".SH")
        elif stock_code.endswith(".BJSE"):
            original_code = stock_code.replace(".BJSE", ".BJ")
        else:
            original_code = stock_code

        # 查找原始文件
        original_files = list(self.original_folder.glob(f"{original_code}-*.csv"))
        if not original_files:
            raise FileNotFoundError(f"未找到原始文件: {original_code}")

        # 查找enhanced文件
        enhanced_files = list(self.enhanced_folder.glob(f"{stock_code}-*.csv"))
        if not enhanced_files:
            raise FileNotFoundError(f"未找到enhanced文件: {stock_code}")

        # 加载数据
        df_original = pd.read_csv(original_files[0], encoding="utf-8")
        df_enhanced = pd.read_csv(enhanced_files[0], encoding="utf-8")

        # 转换日期格式
        df_original["交易日期"] = pd.to_datetime(df_original["交易日期"].astype(str))
        df_enhanced["交易日期"] = pd.to_datetime(df_enhanced["交易日期"].astype(str))

        return df_original, df_enhanced

    def align_data_by_date(
        self, df_original, df_enhanced, start_date=None, end_date=None
    ):
        """
        按日期对齐数据

        Args:
            df_original: 原始数据
            df_enhanced: enhanced数据
            start_date: 起始日期，默认为enhanced数据的起始日期
            end_date: 结束日期，默认为原始数据的结束日期

        Returns:
            tuple: (aligned_original, aligned_enhanced)
        """
        if start_date is None:
            start_date = df_enhanced["交易日期"].min()
        if end_date is None:
            end_date = df_original["交易日期"].max()

        # 过滤日期范围
        df_orig_filtered = df_original[
            (df_original["交易日期"] >= start_date)
            & (df_original["交易日期"] <= end_date)
        ].copy()

        df_enh_filtered = df_enhanced[
            (df_enhanced["交易日期"] >= start_date)
            & (df_enhanced["交易日期"] <= end_date)
        ].copy()

        # 按日期合并，保留交集
        merged = pd.merge(
            df_orig_filtered,
            df_enh_filtered,
            on="交易日期",
            suffixes=("_原始", "_enhanced"),
        )

        return merged

    def compare_feature(self, stock_code, enhanced_feature, original_feature=None):
        """
        对比指定特征

        Args:
            stock_code: 股票代码
            enhanced_feature: enhanced文件中的特征名
            original_feature: 原始文件中的特征名，如果为None则从映射字典中查找

        Returns:
            dict: 对比结果
        """
        if original_feature is None:
            original_feature = self.feature_mapping.get(enhanced_feature)
            if original_feature is None:
                raise ValueError(f"未找到特征映射: {enhanced_feature}")

        # 加载数据
        df_original, df_enhanced = self.load_stock_data(stock_code)

        # 对齐数据
        merged_data = self.align_data_by_date(df_original, df_enhanced)

        if merged_data.empty:
            return {"status": "error", "message": "没有重叠的交易日期"}

        # 提取对比特征
        # 先尝试带后缀的列名，如果不存在则使用原始列名
        original_col = f"{original_feature}_原始"
        if original_col not in merged_data.columns:
            original_col = original_feature

        enhanced_col = f"{enhanced_feature}_enhanced"
        if enhanced_col not in merged_data.columns:
            enhanced_col = enhanced_feature

        if original_col not in merged_data.columns:
            return {
                "status": "error",
                "message": f"原始数据中未找到列: {original_feature} (尝试了 {original_feature}_原始 和 {original_feature})",
            }

        if enhanced_col not in merged_data.columns:
            return {
                "status": "error",
                "message": f"Enhanced数据中未找到列: {enhanced_feature} (尝试了 {enhanced_feature}_enhanced 和 {enhanced_feature})",
            }

        # 获取数值数据，处理缺失值
        original_values = pd.to_numeric(merged_data[original_col], errors="coerce")
        enhanced_values = pd.to_numeric(merged_data[enhanced_col], errors="coerce")

        # 计算统计指标，过滤掉NaN和无穷大值
        valid_mask = ~(
            original_values.isna()
            | enhanced_values.isna()
            | np.isinf(original_values)
            | np.isinf(enhanced_values)
        )
        valid_original = original_values[valid_mask]
        valid_enhanced = enhanced_values[valid_mask]

        if len(valid_original) == 0:
            return {"status": "error", "message": "没有有效的数值数据进行对比"}

        # 计算差异
        diff = valid_enhanced - valid_original
        abs_diff = np.abs(diff)

        # 统计结果
        result = {
            "status": "success",
            "stock_code": stock_code,
            "enhanced_feature": enhanced_feature,
            "original_feature": original_feature,
            "data_points": len(valid_original),
            "date_range": {
                "start": merged_data["交易日期"].min().strftime("%Y-%m-%d"),
                "end": merged_data["交易日期"].max().strftime("%Y-%m-%d"),
            },
            "statistics": {
                "correlation": round(float(valid_original.corr(valid_enhanced)), 6),
                "mean_abs_diff": round(float(abs_diff.mean()), 6),
            },
            "data": merged_data[["交易日期", original_col, enhanced_col]].copy(),
        }

        return result

    def batch_compare_features(self, stock_codes, features, output_dir=None):
        """
        批量对比多个股票的多个特征

        Args:
            stock_codes: 股票代码列表
            features: 特征列表（enhanced文件中的列名）
            output_dir: 输出目录，如果为None则不保存

        Returns:
            dict: 批量对比结果
        """
        results = {}

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        for stock_code in stock_codes:
            results[stock_code] = {}
            print(f"\n处理股票: {stock_code}")

            for feature in features:
                print(f"  对比特征: {feature}")
                try:
                    result = self.compare_feature(stock_code, feature)
                    results[stock_code][feature] = result

                    if result["status"] == "success":
                        print(
                            f"    ✓ 成功 - 相关系数: {result['statistics']['correlation']:.6f}"
                        )
                    else:
                        print(f"    ✗ 失败 - {result['message']}")

                except Exception as e:
                    print(f"    ✗ 错误 - {str(e)}")
                    results[stock_code][feature] = {
                        "status": "error",
                        "message": str(e),
                    }

        return results

    def _compare_single_stock_feature(self, stock_code, feature):
        """
        单个股票特征对比的辅助函数（用于并行处理）

        Args:
            stock_code: 股票代码
            feature: 特征名称

        Returns:
            tuple: (stock_code, feature, result)
        """
        try:
            result = self.compare_feature(stock_code, feature)
            return stock_code, feature, result
        except Exception as e:
            return stock_code, feature, {"status": "error", "message": str(e)}

    def batch_compare_features_parallel(
        self, stock_codes, features, max_workers=8, output_dir=None
    ):
        """
        并行批量对比多个股票的多个特征

        Args:
            stock_codes: 股票代码列表
            features: 特征列表（enhanced文件中的列名）
            max_workers: 最大并行工作线程数
            output_dir: 输出目录，如果为None则不保存

        Returns:
            dict: 批量对比结果
        """
        results = {}
        total_tasks = len(stock_codes) * len(features)
        completed_tasks = 0

        print(f"开始并行处理 {len(stock_codes)} 只股票的 {len(features)} 个特征...")
        print(f"总任务数: {total_tasks}, 并行线程数: {max_workers}")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        # 初始化结果字典
        for stock_code in stock_codes:
            results[stock_code] = {}

        # 创建任务列表
        tasks = []
        for stock_code in stock_codes:
            for feature in features:
                tasks.append((stock_code, feature))

        start_time = time.time()

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(
                    self._compare_single_stock_feature, stock_code, feature
                ): (stock_code, feature)
                for stock_code, feature in tasks
            }

            # 处理完成的任务
            for future in as_completed(future_to_task):
                stock_code, feature = future_to_task[future]
                completed_tasks += 1

                try:
                    _, _, result = future.result()
                    results[stock_code][feature] = result

                    if result["status"] == "success":
                        correlation = result["statistics"]["correlation"]
                        print(
                            f"[{completed_tasks:4d}/{total_tasks}] ✓ {stock_code} - {feature} - 相关系数: {correlation:.4f}"
                        )
                    else:
                        print(
                            f"[{completed_tasks:4d}/{total_tasks}] ✗ {stock_code} - {feature} - {result['message']}"
                        )

                except Exception as e:
                    print(
                        f"[{completed_tasks:4d}/{total_tasks}] ✗ {stock_code} - {feature} - 错误: {str(e)}"
                    )
                    results[stock_code][feature] = {
                        "status": "error",
                        "message": str(e),
                    }

                # 每处理100个任务显示进度
                if completed_tasks % 100 == 0 or completed_tasks == total_tasks:
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_time = remaining_tasks * avg_time_per_task

                    print(
                        f"\n进度: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.1f}%)"
                    )
                    print(
                        f"已用时间: {elapsed_time:.1f}秒, 预计剩余时间: {estimated_remaining_time:.1f}秒"
                    )
                    print(f"平均每任务耗时: {avg_time_per_task:.2f}秒\n")

        total_time = time.time() - start_time
        print(f"\n并行处理完成！总耗时: {total_time:.1f}秒")
        print(f"平均每任务耗时: {total_time/total_tasks:.2f}秒")

        return results

    def generate_summary_report(self, batch_results, output_path=None):
        """
        生成综合统计报告

        Args:
            batch_results: batch_compare_features的返回结果
            output_path: 报告保存路径

        Returns:
            dict: 综合统计结果
        """
        summary_data = []
        feature_stats = {}

        # 收集所有成功的对比结果
        for stock_code, stock_results in batch_results.items():
            for feature, result in stock_results.items():
                if result["status"] == "success":
                    stats = result["statistics"]
                    summary_data.append(
                        {
                            "股票代码": stock_code,
                            "特征名称": feature,
                            "数据点数": result["data_points"],
                            "日期范围": f"{result['date_range']['start']} 至 {result['date_range']['end']}",
                            "相关系数": stats["correlation"],
                            "平均绝对差异": stats["mean_abs_diff"],
                        }
                    )

                    # 按特征统计
                    if feature not in feature_stats:
                        feature_stats[feature] = {
                            "correlations": [],
                            "mean_abs_diffs": [],
                            "success_count": 0,
                            "total_count": 0,
                        }

                    feature_stats[feature]["correlations"].append(stats["correlation"])
                    feature_stats[feature]["mean_abs_diffs"].append(
                        stats["mean_abs_diff"]
                    )
                    feature_stats[feature]["success_count"] += 1

                # 统计总数
                if feature not in feature_stats:
                    feature_stats[feature] = {
                        "correlations": [],
                        "mean_abs_diffs": [],
                        "success_count": 0,
                        "total_count": 0,
                    }
                feature_stats[feature]["total_count"] += 1

        # 创建详细报告 DataFrame
        detailed_df = pd.DataFrame(summary_data)

        # 创建特征统计报告
        feature_summary = []
        for feature, stats in feature_stats.items():
            if stats["correlations"]:  # 只处理有成功数据的特征
                feature_summary.append(
                    {
                        "特征名称": feature,
                        "成功股票数": stats["success_count"],
                        "总股票数": stats["total_count"],
                        "成功率(%)": (stats["success_count"] / stats["total_count"])
                        * 100,
                        "平均相关系数": np.mean(stats["correlations"]),
                        "相关系数标准差": np.std(stats["correlations"]),
                        "最小相关系数": np.min(stats["correlations"]),
                        "最大相关系数": np.max(stats["correlations"]),
                        "平均绝对差异均值": np.mean(stats["mean_abs_diffs"]),
                    }
                )

        feature_summary_df = pd.DataFrame(feature_summary)

        # 生成总体统计
        total_comparisons = len(summary_data)
        total_stocks = len(set([row["股票代码"] for row in summary_data]))
        total_features = len(set([row["特征名称"] for row in summary_data]))

        overall_stats = {
            "总对比数": total_comparisons,
            "涉及股票数": total_stocks,
            "涉及特征数": total_features,
            "平均相关系数": (
                detailed_df["相关系数"].mean() if not detailed_df.empty else 0
            ),
            "相关系数中位数": (
                detailed_df["相关系数"].median() if not detailed_df.empty else 0
            ),
            "高质量对比数(相关系数>0.99)": (
                len(detailed_df[detailed_df["相关系数"] > 0.99])
                if not detailed_df.empty
                else 0
            ),
            "优秀对比数(相关系数>0.95)": (
                len(detailed_df[detailed_df["相关系数"] > 0.95])
                if not detailed_df.empty
                else 0
            ),
        }

        # 打印报告
        print("\n" + "=" * 80)
        print("特征对比综合统计报告")
        print("=" * 80)

        print(f"\n总体概况:")
        for key, value in overall_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\n各特征统计概况:")
        if not feature_summary_df.empty:
            print(feature_summary_df.round(4).to_string(index=False))

        # 保存报告
        if output_path:
            output_path = Path(output_path)

            # 保存详细报告
            detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
            detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")

            # 保存特征统计报告
            feature_path = (
                output_path.parent / f"{output_path.stem}_feature_summary.csv"
            )
            feature_summary_df.to_csv(feature_path, index=False, encoding="utf-8")

            # 保存总体统计
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("特征对比综合统计报告\n")
                f.write("=" * 80 + "\n\n")

                f.write("总体概况:\n")
                for key, value in overall_stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

                f.write("\n各特征统计概况:\n")
                if not feature_summary_df.empty:
                    f.write(feature_summary_df.round(4).to_string(index=False))

            print(f"\n报告已保存:")
            print(f"  总体报告: {output_path}")
            print(f"  详细数据: {detailed_path}")
            print(f"  特征统计: {feature_path}")

        return {
            "overall_stats": overall_stats,
            "feature_summary": feature_summary_df,
            "detailed_data": detailed_df,
            "raw_results": batch_results,
        }


def test_single_stock_feature(stock_code="000001.XSHE", feature="换手率(自由流动股)"):
    """测试单只股票的单个特征对比

    Args:
        stock_code (str): 股票代码，默认为"000001.XSHE"
        feature (str): 特征名称，默认为"换手率(自由流动股)"
    """
    # 初始化对比器
    comparator = FeatureComparator(
        original_folder="/Users/didi/KDCJ/deep_model/backtest/日线后复权及常用指标csv",
        enhanced_folder="/Users/didi/KDCJ/deep_model/enhanced_factors_csv",
    )

    print(f"开始对比 {stock_code} 的 {feature} 特征...")

    try:
        result = comparator.compare_feature(stock_code, feature)

        if result["status"] == "success":
            print("\n=== 对比结果 ===")
            print(f"股票代码: {result['stock_code']}")
            print(f"Enhanced特征: {result['enhanced_feature']}")
            print(f"原始特征: {result['original_feature']}")
            print(f"数据点数: {result['data_points']:,}")
            print(
                f"日期范围: {result['date_range']['start']} 至 {result['date_range']['end']}"
            )

            stats = result["statistics"]
            print(f"\n=== 统计信息 ===")
            print(f"相关系数: {stats['correlation']:.6f}")
            print(f"平均绝对差异: {stats['mean_abs_diff']:.6f}")

        else:
            print(f"对比失败: {result['message']}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


def test_batch_stocks_features():
    """测试批量股票的多个特征对比"""
    # 初始化对比器
    comparator = FeatureComparator(
        original_folder="/Users/didi/KDCJ/deep_model/backtest/日线后复权及常用指标csv",
        enhanced_folder="/Users/didi/KDCJ/deep_model/enhanced_factors_csv",
    )

    # 选择20只股票进行测试（包括深市、上市、北交所）
    test_stocks = [
        "000001.XSHE",  # 平安银行
        "000002.XSHE",  # 万科A
        "000858.XSHE",  # 五粮液
        "002415.XSHE",  # 海康威视
        "300059.XSHE",  # 东方财富
        "600000.XSHG",  # 浦发银行
        "600036.XSHG",  # 招商银行
        "600519.XSHG",  # 贵州茅台
        "600887.XSHG",  # 伊利股份
        "601318.XSHG",  # 中国平安
        "430017.BJSE",  # 星轰股份
        "430047.BJSE",  # 诺思科技
        "430139.BJSE",  # 嘉诚股份
        "430198.BJSE",  # 微创光电
        "430300.BJSE",  # 永顺生物
        "831087.BJSE",  # 秋乐种业
        "831305.BJSE",  # 科达自控
        "836419.BJSE",  # 万德股份
        "838971.BJSE",  # 天马新材
        "920682.BJSE",  # 球冠电缆
    ]

    # 选择要对比的特征
    test_features = ["换手率"]

    print(f"开始批量对比 {len(test_stocks)} 只股票的 {len(test_features)} 个特征...")

    # 执行并行批量对比（使用并行处理）
    batch_results = comparator.batch_compare_features_parallel(
        test_stocks, test_features, max_workers=8
    )

    # 生成综合统计报告
    output_path = "/Users/didi/KDCJ/deep_model/feature_comparison_report.txt"
    summary = comparator.generate_summary_report(batch_results, output_path)

    return summary


def test_large_scale_comparison():
    """大规模特征对比测试（适用于6000只股票）"""
    # 初始化对比器
    comparator = FeatureComparator(
        original_folder="/Users/didi/KDCJ/deep_model/backtest/日线后复权及常用指标csv",
        enhanced_folder="/Users/didi/KDCJ/deep_model/enhanced_factors_csv",
    )

    # 获取所有可用的股票代码
    enhanced_files = list(
        Path("/Users/didi/KDCJ/deep_model/enhanced_factors_csv").glob("*.csv")
    )
    all_stocks = []

    for file in enhanced_files:
        # 从文件名提取股票代码
        filename = file.name
        if "-" in filename:
            stock_code = filename.split("-")[0]
            all_stocks.append(stock_code)

    print(f"找到 {len(all_stocks)} 只股票可用于对比")

    # 先只测试前100只股票
    test_stocks = all_stocks
    print(f"先测试前 {len(test_stocks)} 只股票")

    # 选择要对比的特征
    test_features = ["换手率(自由流动股)"]

    print(f"开始大规模对比 {len(test_stocks)} 只股票的 {len(test_features)} 个特征...")
    print(f"预计总任务数: {len(test_stocks) * len(test_features)}")

    # 执行并行批量对比（使用更多线程）
    batch_results = comparator.batch_compare_features_parallel(
        test_stocks, test_features, max_workers=8
    )

    # 生成综合统计报告
    output_path = "/Users/didi/KDCJ/deep_model/large_scale_comparison_report.txt"
    summary = comparator.generate_summary_report(batch_results, output_path)

    return summary


if __name__ == "__main__":
    # 选择测试模式
    test_mode = "large_scale"  # "single", "batch", 或 "large_scale"

    if test_mode == "single":
        test_single_stock_feature()
    elif test_mode == "batch":
        test_batch_stocks_features()
    elif test_mode == "large_scale":
        test_large_scale_comparison()
