"""
调试回测代码
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from rolling_backtest import rolling_backtest, get_performance_analysis


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

    signal_end_date = "2025-08-19"
    backtest_end_date = "2025-08-19"
    rank_n = 10
    portfolio_count = 12

    # 读取信号队列
    portfolio_weights_path = get_portfolio_weights_path(signal_end_date, rank_n)
    print(f"读取portfolio_weights: {os.path.basename(portfolio_weights_path)}")
    portfolio_weights = pd.read_pickle(portfolio_weights_path)
    # 根据backtest_end_date截取portfolio_weights
    portfolio_weights = portfolio_weights.loc[:backtest_end_date]

    # 读取股票价格数据
    bars_df_path = get_bars_df_path(backtest_end_date)
    print(f"读取bars_df: {os.path.basename(bars_df_path)}")
    bars_df = pd.read_pickle(bars_df_path)

    # 读取基准
    index_item = "000852.XSHG"

    account_result = rolling_backtest(
        portfolio_weights, bars_df, portfolio_count=portfolio_count
    )

    performance_cumnet, result = get_performance_analysis(
        account_result,
        benchmark_index=index_item,
        portfolio_weights=portfolio_weights,
        portfolio_count=portfolio_count,
        factor_name="rank",
        rank_n=rank_n,
    )
    print(result)
