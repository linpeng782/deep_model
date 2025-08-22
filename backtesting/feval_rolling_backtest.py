"""
调试回测代码
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from rolling_backtest_post import (
    rolling_backtest,
    get_performance_analysis,
    plot_backtest_performance,
)


def run_multi_scenario_backtest(
    portfolio_weights, bars_df, benchmark_index, scenarios="open_open"
):
    """
    执行多场景回测对比

    参数:
        portfolio_weights: 投资组合权重数据
        bars_df: 股票价格数据
        benchmark_index: 基准指数代码
        scenarios: 测试场景选择
            - "open_open": 只测试开盘价卖出-开盘价买入
            - "all": 测试所有三种场景
            - list: 自定义场景列表
    """
    # 定义所有可用的交易时点参数
    all_test_scenarios = [
        {
            "sell_timing": "close",
            "buy_timing": "close",
            "name": "收盘价卖出-收盘价买入",
        },
        {
            "sell_timing": "open",
            "buy_timing": "close",
            "name": "开盘价卖出-收盘价买入",
        },
        {
            "sell_timing": "open",
            "buy_timing": "open",
            "name": "开盘价卖出-开盘价买入",
        },
    ]

    # 根据scenarios参数选择要测试的场景
    if scenarios == "open_open":
        # 只测试开盘价卖出-开盘价买入
        test_scenarios = [all_test_scenarios[2]]  # 第三个场景
    elif scenarios == "all":
        # 测试所有场景
        test_scenarios = all_test_scenarios
    elif isinstance(scenarios, list):
        # 自定义场景列表
        test_scenarios = scenarios
    else:
        raise ValueError(f"不支持的scenarios参数: {scenarios}")

    # 存储所有结果
    all_results = []

    print(f"\n将执行 {len(test_scenarios)} 个测试场景")

    # 逐个执行回测
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n正在执行第{i}组测试: {scenario['name']}")
        print(
            f"参数: sell_timing='{scenario['sell_timing']}', buy_timing='{scenario['buy_timing']}'"
        )
        print("-" * 60)

        # 执行回测
        account_result = rolling_backtest(
            portfolio_weights,
            bars_df,
            holding_months=1,
            sell_timing=scenario["sell_timing"],
            buy_timing=scenario["buy_timing"],
        )

        # 分析结果
        performance_cumnet, result_df = get_performance_analysis(
            account_result, benchmark_index=benchmark_index
        )

        # 存储结果
        all_results.append(
            {
                "scenario": scenario,
                "performance_cumnet": performance_cumnet,
                "result_df": result_df,
            }
        )

        print(f"第{i}组测试完成\n")

    # 创建合并后的DataFrame
    combined_df = all_results[0]["result_df"]["指标"].to_frame()

    for result in all_results:
        scenario_name = result["scenario"]["name"]
        combined_df[scenario_name] = result["result_df"]["数值"]

    # 设置pandas显示选项以保证对齐
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)

    print(combined_df.to_string())

    # 找到年化收益最高的结果用于绘图
    annual_returns = []
    for result in all_results:
        annual_return = result["result_df"][
            result["result_df"]["指标"] == "策略年化收益"
        ]["数值"].iloc[0]
        annual_returns.append(annual_return)

    best_idx = annual_returns.index(max(annual_returns))
    best_result = all_results[best_idx]

    print(
        f"\n最优结果: {best_result['scenario']['name']} (年化收益: {max(annual_returns):.6f})"
    )

    # 使用最优结果绘图
    print(f"\n正在为最优结果绘制图表...")
    fig = plot_backtest_performance(best_result["performance_cumnet"], benchmark_index)

    return best_result, combined_df


def get_portfolio_weights_path(end_date, rank_n):
    """
    根据结束日期和排名参数获取portfolio_weights文件路径
    """
    import glob

    buy_list_dir = "/Users/didi/KDCJ/deep_model/data/cache/buy_list"
    # 查找包含end_date和rank_n的文件
    pattern = f"*_{end_date.replace('-', '')}_rank{rank_n}_weights.pkl"
    matching_files = glob.glob(os.path.join(buy_list_dir, pattern))

    if not matching_files:
        raise FileNotFoundError(f"未找到匹配的portfolio_weights文件: {pattern}")
    elif len(matching_files) > 1:
        print(f"警告：找到多个匹配文件，使用第一个: {matching_files[0]}")

    return matching_files[0]


def get_bars_df_path(end_date):
    """
    根据结束日期获取bars_df文件路径
    """
    import glob

    bars_dir = "/Users/didi/KDCJ/deep_model/data/cache/bars"
    # 查找包含end_date的文件
    pattern = f"{end_date.replace('-', '')}_bars_df.pkl"
    bars_df_path = os.path.join(bars_dir, pattern)

    if not os.path.exists(bars_df_path):
        raise FileNotFoundError(f"未找到bars_df文件: {bars_df_path}")

    return bars_df_path


if __name__ == "__main__":

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    end_date = "2025-08-19"
    rank_n = 10

    # 读取买入队列
    portfolio_weights_path = get_portfolio_weights_path(end_date, rank_n)
    print(f"读取portfolio_weights: {os.path.basename(portfolio_weights_path)}")
    portfolio_weights = pd.read_pickle(portfolio_weights_path)

    # 读取股票价格数据
    bars_df_path = get_bars_df_path(end_date)
    print(f"读取bars_df: {os.path.basename(bars_df_path)}")
    bars_df = pd.read_pickle(bars_df_path)

    # 读取基准
    index_item = "000852.XSHG"

    # 执行多场景回测对比
    # 可选参数:
    # scenarios="open_open"  - 只测试开盘价卖出-开盘价买入（默认）
    # scenarios="all"       - 测试所有三种场景
    best_result, comparison_df = run_multi_scenario_backtest(
        portfolio_weights, bars_df, index_item
    )
