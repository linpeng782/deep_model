"""
12个月滚动持仓回测框架
作者：基于原backtest函数重构
日期：2025-07-23
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from tqdm import *
import os
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

logging.getLogger().setLevel(logging.ERROR)


def calculate_target_holdings(
    target_weights, available_cash, stock_prices, min_trade_units, sell_cost_rate
):
    """
    计算目标持仓数量

    :param target_weights: 目标权重 Series
    :param available_cash: 可用资金
    :param stock_prices: 股票价格 Series
    :param min_trade_units: 最小交易单位 Series
    :param sell_cost_rate: 卖出成本费率（用于预留手续费）
    :return: 目标持仓数量 Series
    """
    # 按权重分配资金
    allocated_cash = target_weights * available_cash

    # 计算调整后价格（预留卖出手续费）
    adjusted_prices = stock_prices * (1 + sell_cost_rate)

    # 计算可购买的最小交易单位数量（向下取整）
    units_to_buy = allocated_cash / adjusted_prices // min_trade_units

    # 转换为实际股数
    target_holdings = units_to_buy * min_trade_units

    return target_holdings


# 获取指定期间内每个月的第一个交易日
def get_monthly_first_trading_days(start_date, end_date):
    """
    获取指定期间内每个月的第一个交易日

    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 每月第一个交易日列表
    """
    # 获取所有交易日
    all_trading_days = get_trading_dates(start_date, end_date)

    monthly_first_days = []
    current_month = None

    for date in all_trading_days:
        date_month = (date.year, date.month)
        if current_month != date_month:
            monthly_first_days.append(date)
            current_month = date_month

    return monthly_first_days


# 生成调仓日期序列
def get_rebalance_dates(start_date, end_date, frequency, portfolio_weights):
    """
    根据频率生成调仓日期序列

    :param start_date: 开始日期
    :param end_date: 结束日期
    :param frequency: 调仓频率 ('daily', 'weekly', 'monthly', 或整数天数)
    :param portfolio_weights: 投资组合权重矩阵（用于获取交易日）
    :return: 调仓日期序列 (pd.DatetimeIndex)
    """
    all_trading_days = portfolio_weights.index

    if frequency == "daily":
        return all_trading_days
    elif frequency == "weekly":
        # 每周第一个交易日
        weekly_first_days = []
        current_week = None
        for date in all_trading_days:
            week_period = date.to_period("W")
            if current_week != week_period:
                weekly_first_days.append(date)
                current_week = week_period
        return pd.DatetimeIndex(weekly_first_days)
    elif frequency == "monthly":
        # 每月第一个交易日（原逻辑）
        return pd.DatetimeIndex(get_monthly_first_trading_days(start_date, end_date))
    elif isinstance(frequency, int):
        # 自定义天数间隔
        rebalance_dates = []
        current_idx = 0
        while current_idx < len(all_trading_days):
            rebalance_dates.append(all_trading_days[current_idx])
            current_idx += frequency
        return pd.DatetimeIndex(rebalance_dates)
    else:
        raise ValueError(f"不支持的调仓频率: {frequency}")


# 获取到期日期（灵活版本）
def get_expire_date_flexible(start_idx, portfolio_count, rebalance_dates, frequency):
    """
    根据调仓日期序列计算到期日期

    :param start_idx: 建仓日期在rebalance_dates中的索引
    :param portfolio_count: 组合数量（持仓期数）
    :param rebalance_dates: 调仓日期序列
    :param frequency: 调仓频率
    :return: 到期日期（pd.Timestamp）
    """
    expire_idx = start_idx + portfolio_count
    if expire_idx < len(rebalance_dates):
        return pd.Timestamp(rebalance_dates[expire_idx])
    else:
        # 超出范围时，基于最后日期推算
        last_date = pd.Timestamp(rebalance_dates[-1])
        if isinstance(frequency, int):
            # 按天数推算
            days_to_add = (expire_idx - len(rebalance_dates) + 1) * frequency
            return last_date + pd.Timedelta(days=days_to_add)
        elif frequency == "weekly":
            weeks_to_add = expire_idx - len(rebalance_dates) + 1
            return last_date + pd.Timedelta(weeks=weeks_to_add)
        elif frequency == "monthly":
            months_to_add = expire_idx - len(rebalance_dates) + 1
            return last_date + relativedelta(months=months_to_add)
        else:  # daily
            days_to_add = expire_idx - len(rebalance_dates) + 1
            return last_date + pd.Timedelta(days=days_to_add)


# 获取到期日期（原版本，保持向后兼容）
def get_expire_date(start_month_idx, portfolio_count, monthly_first_days):
    """
    根据建仓月份索引和组合数量，计算到期日期
    :param start_month_idx: 建仓月份在monthly_first_days中的索引
    :param portfolio_count: 组合数量（也是持仓月数）
    :param monthly_first_days: 每月第一个交易日列表
    :return: 到期日期（pd.Timestamp）
    """
    expire_month_idx = start_month_idx + portfolio_count
    if expire_month_idx < len(monthly_first_days):
        return pd.Timestamp(monthly_first_days[expire_month_idx])
    else:
        # 如果超出了monthly_first_days的范围，返回最后一个日期后的N个月
        last_date = pd.Timestamp(monthly_first_days[-1])
        return last_date + relativedelta(months=portfolio_count)


# 计算单笔交易的手续费
def calc_transaction_fee(
    transaction_value, min_transaction_fee, sell_cost_rate, buy_cost_rate
):
    """
    计算单笔交易的手续费
    :param transaction_value: 交易金额（正数为买入，负数为卖出）
    :param min_transaction_fee: 最低交易手续费
    :param sell_cost_rate: 卖出成本费率
    :param buy_cost_rate: 买入成本费率
    :return: 交易手续费
    """
    if pd.isna(transaction_value) or transaction_value == 0:
        return 0  # 无交易时手续费为0
    elif transaction_value < 0:  # 卖出交易（负数）
        fee = -transaction_value * sell_cost_rate  # 卖出手续费：印花税 + 过户费 + 佣金
    else:  # 买入交易（正数）
        fee = transaction_value * buy_cost_rate  # 买入手续费：过户费 + 佣金

    # 应用最低手续费限制
    return max(fee, min_transaction_fee)  # 返回实际手续费和最低手续费中的较大值


# 获取股票价格数据
def get_stock_bars(stock_price_data, portfolio_weights, adjust):
    """
    从股票价格数据中获取指定时间范围和股票列表的开盘价数据

    参数:
        stock_price_data: 股票价格数据（多级索引DataFrame）
        portfolio_weights: 投资组合权重矩阵
        adjust: 复权类型（'post'或'none'）

    返回:
        开盘价DataFrame（日期为行索引，股票代码为列索引）
    """

    # 计算时间范围：开始日期，结束日期
    start_date = portfolio_weights.index.min()
    end_date = portfolio_weights.index.max()

    print(f"筛选时间范围: {start_date} 到 {end_date}")

    # 获取股票列表
    stock_list = portfolio_weights.columns.tolist()
    print(f"股票数量: {len(stock_list)}")

    # 获取stock_price_data中所有的股票代码并去重
    stock_book = stock_price_data.index.get_level_values("order_book_id").unique()
    print(f"stock_price_data中的股票数量: {len(stock_book)}")

    # 按时间范围和股票列表筛选数据
    filtered_data = stock_price_data.loc[
        (stock_price_data.index.get_level_values("order_book_id").isin(stock_list))
        & (stock_price_data.index.get_level_values("datetime") >= start_date)
        & (stock_price_data.index.get_level_values("datetime") <= end_date)
    ]

    # 根据复权类型返回相应的开盘价数据
    if adjust == "post":
        open_price = filtered_data["开盘价"].unstack("order_book_id")
        return open_price
    elif adjust == "none":
        open_price = filtered_data["未复权开盘价"].unstack("order_book_id")
        return open_price


# 获取基准
def get_benchmark(df, benchmark, benchmark_type="mcw"):
    """
    :param df: 买入队列 -> dataframe/unstack
    :param benchmark: 基准指数 -> str
    :return ret: 基准的逐日收益 -> dataframe
    """
    start_date = get_previous_trading_date(df.index.min(), 1).strftime("%F")
    end_date = df.index.max().strftime("%F")
    if benchmark_type == "mcw":
        price_open = get_price(
            [benchmark], start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
    else:
        index_fix = INDEX_FIX(start_date, end_date, benchmark)
        stock_list = index_fix.columns.tolist()
        price_open = get_price(
            stock_list, start_date, end_date, fields=["open"]
        ).open.unstack("order_book_id")
        price_open = price_open.pct_change().mask(~index_fix).mean(axis=1)
        price_open = (1 + price_open).cumprod().to_frame(benchmark)

    return price_open


def rolling_backtest(
    portfolio_weights,
    bars_df,
    portfolio_count=12,
    rebalance_frequency="monthly",
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
):
    """
    N个组合滚动持仓回测框架（支持灵活调仓频率）

    :param portfolio_weights: 投资组合权重矩阵 -> DataFrame
    :param portfolio_count: 组合数量 -> int (资金分割份数，如12表示分为12个组合)
    :param rebalance_frequency: 调仓频率 -> str/int
        - "daily": 每日调仓
        - "weekly": 每周调仓
        - "monthly": 每月调仓（默认）
        - int: 自定义天数间隔（如12表示每12天调仓）
    :param bars_df: 股票价格数据 -> DataFrame
    :param initial_capital: 初始资金 -> float
    :param stamp_tax_rate: 印花税费率 -> float
    :param transfer_fee_rate: 过户费费率 -> float
    :param commission_rate: 佣金费率 -> float
    :param min_transaction_fee: 最低交易手续费 -> float
    :param cash_annual_yield: 现金账户年化收益率 -> float
    :return: 账户历史记录 -> DataFrame
    """

    # =========================== 基础参数初始化 ===========================
    # 每个组合的资金分配（总资金的 1/portfolio_count）
    portfolio_capital = initial_capital / portfolio_count
    # 买入成本费率：过户费 + 佣金
    buy_cost_rate = transfer_fee_rate + commission_rate
    # 卖出成本费率：印花税 + 过户费 + 佣金
    sell_cost_rate = stamp_tax_rate + transfer_fee_rate + commission_rate
    # 现金账户日利率（年化收益率转换为日收益率）
    daily_cash_yield = (1 + cash_annual_yield) ** (1 / 252) - 1

    # =========================== 数据结构初始化 ===========================
    # 创建账户历史记录表，索引为所有交易日
    account_history = pd.DataFrame(
        index=portfolio_weights.index,
        columns=["total_account_asset", "holding_market_cap", "cash_account"],
    )

    # 获取所有股票的开盘价格数据（未复权）
    print("获取所有股票的开盘价格数据（未复权）")
    open_prices = get_stock_bars(bars_df, portfolio_weights, "none")
    # 获取所有股票的后复权价格数据
    print("获取所有股票的后复权价格数据")
    adjusted_prices = get_stock_bars(bars_df, portfolio_weights, "post")
    # 获取每只股票的最小交易单位（通常为100股）
    min_trade_units = pd.Series(
        dict([(stock, 100) for stock in portfolio_weights.columns.tolist()])
    )

    # 生成调仓日期序列
    signal_start_date = portfolio_weights.index.min()
    signal_end_date = portfolio_weights.index.max()
    rebalance_dates = get_rebalance_dates(
        signal_start_date, signal_end_date, rebalance_frequency, portfolio_weights
    )

    print(f"调仓频率: {rebalance_frequency}")
    print(f"总调仓次数: {len(rebalance_dates)}")
    print(f"资金分割份数: {portfolio_count}")

    # =========================== N个组合管理 ===========================
    # 存储N个组合的信息和历史记录
    portfolios = {}
    portfolio_histories = {}

    # 初始化N个空组合及其历史记录
    for i in range(portfolio_count):
        portfolios[i] = {
            "holdings": pd.Series(dtype=float),  # 持仓股票及数量
            "cash": 0.0,  # 现金余额
            "start_date": None,  # 建仓日期
            "expire_date": None,  # 到期日期
            "is_active": False,  # 是否激活
        }
        portfolio_histories[i] = {
            "total_account_asset": [],
            "holding_market_cap": [],
            "cash_account": [],
        }

    # =========================== 开始滚动建仓和调仓 ===========================
    portfolio_index = 0  # 当前使用的组合索引（0到portfolio_count-1循环）

    for date_idx, rebalance_date_raw in enumerate(tqdm(rebalance_dates)):

        # 统一转换为pd.Timestamp类型
        rebalance_date = pd.Timestamp(rebalance_date_raw)

        # if rebalance_date == pd.Timestamp("2017-01-03"):
        #     breakpoint()

        # 获取当前调仓日的目标权重
        target_weights = portfolio_weights.loc[rebalance_date].dropna()
        # 获取当前调仓日的目标股票
        target_stocks = target_weights.index.tolist()
        # 计算目标股票的开盘价
        target_prices = open_prices.loc[rebalance_date, target_stocks]

        # =========================== 处理当前组合，更新持仓 ===========================
        current_portfolio = portfolios[portfolio_index]

        # 检查组合状态和到期情况
        if current_portfolio["is_active"]:
            # 组合到期，进行调仓
            print(f"组合{portfolio_index}号到期调仓，调仓日期：{rebalance_date}")

            # 取最后一个时间段的最后一天的值
            last_period_records = portfolio_histories[portfolio_index][
                "total_account_asset"
            ][-1]
            available_cash = last_period_records.loc[rebalance_date]  # 最后一天的总资产

            # 重置组合的到期日期（新的N个周期）
            current_portfolio["expire_date"] = get_expire_date_flexible(
                date_idx, portfolio_count, rebalance_dates, rebalance_frequency
            )
            current_portfolio["start_date"] = rebalance_date  # 更新开始日期

        else:
            # 新建仓，使用固定的月度资金
            print(f"组合{portfolio_index}号首次建仓，建仓日期：{rebalance_date}")
            available_cash = portfolio_capital
            current_portfolio["is_active"] = True
            current_portfolio["start_date"] = rebalance_date
            # 计算到期日期（N个周期后的调仓日）
            current_portfolio["expire_date"] = get_expire_date_flexible(
                date_idx, portfolio_count, rebalance_dates, rebalance_frequency
            )

        # =========================== 计算目标持仓 ===========================
        # 计算目标持仓数量（本次调仓需要买入的股票数量）
        target_holdings = calculate_target_holdings(
            target_weights,
            available_cash,
            target_prices,
            min_trade_units.loc[target_stocks],
            sell_cost_rate,
        )

        # =========================== 计算持仓变动 ===========================
        ## 步骤1：计算持仓变动量（目标持仓 - 历史持仓）
        # fill_value=0 确保新增股票（历史持仓为空）和清仓股票（目标持仓为空）都能正确计算
        holdings_change_raw = target_holdings.sub(
            current_portfolio["holdings"], fill_value=0
        )

        ## 步骤2：过滤掉无变动的股票（变动量为0的股票,用np.nan替换）
        holdings_change_filtered = holdings_change_raw.replace(0, np.nan)

        ## 步骤3：删除NaN，获取最终的交易执行列表
        trades_to_execute = holdings_change_filtered.dropna()

        # 获取当前调仓日的所有股票开盘价
        current_prices = open_prices.loc[rebalance_date]

        # =========================== 执行交易并计算成本 ===========================

        # 计算总交易成本
        total_transaction_cost = (
            (current_prices * trades_to_execute)
            .apply(
                lambda x: calc_transaction_fee(
                    x, min_transaction_fee, sell_cost_rate, buy_cost_rate
                )
            )
            .sum()
        )

        # 更新持仓
        current_portfolio["holdings"] = target_holdings

        # =========================== 价格复权调整 ===========================
        # 计算从建仓日到到期日的价格复权比率
        portfolio_start_date = rebalance_date
        portfolio_end_date = current_portfolio["expire_date"]

        period_adj_prices = adjusted_prices.loc[portfolio_start_date:portfolio_end_date]
        base_adj_prices = adjusted_prices.loc[portfolio_start_date]
        price_multipliers = period_adj_prices.div(base_adj_prices, axis=1)
        simulated_prices = price_multipliers.mul(current_prices, axis=1).dropna(
            axis=1, how="all"
        )

        # 处理价格缺失的情况：当价格为NaN时，使用前一日价格填充
        simulated_prices_filled = simulated_prices.ffill()

        # =========================== 计算投资组合市值 ===========================
        # 投资组合市值 = 每只股票的(调整后价格 * 持仓数量)的总和
        portfolio_market_value = (simulated_prices_filled * target_holdings).sum(axis=1)

        # =========================== 计算现金账户余额 ===========================

        # 更新现金余额
        current_portfolio["cash"] = (
            available_cash
            - total_transaction_cost
            - portfolio_market_value.loc[rebalance_date]
        )

        # 计算期间现金账户的复利增长（按日计息）
        cash_balance = pd.Series(
            [
                current_portfolio["cash"]
                * ((1 + daily_cash_yield) ** (day + 1))  # 复利计息公式
                for day in range(0, len(portfolio_market_value))
            ],
            index=portfolio_market_value.index,
        )

        # =========================== 计算账户总资产 ===========================
        # 总资产 = 持仓市值 + 现金余额
        total_portfolio_value = portfolio_market_value + cash_balance

        # =========================== 保存组合账户历史记录 ===========================
        # 将当前期间的记录追加到该组合的历史记录中
        portfolio_histories[portfolio_index]["total_account_asset"].append(
            round(total_portfolio_value, 2)
        )
        portfolio_histories[portfolio_index]["holding_market_cap"].append(
            round(portfolio_market_value, 2)
        )
        portfolio_histories[portfolio_index]["cash_account"].append(
            round(cash_balance, 2)
        )

        # 移动到下一个组合索引
        portfolio_index = (portfolio_index + 1) % portfolio_count

    # =========================== 连接每个组合的多个时间段记录 ===========================
    print("连接每个组合的多个时间段记录...")

    # 为每个组合连接其所有时间段的记录
    combined_portfolio_histories = {}

    for portfolio_index in range(portfolio_count):
        if len(portfolio_histories[portfolio_index]["total_account_asset"]) == 0:
            continue

        print(f"连接组合{portfolio_index}号的记录...")

        # 连接该组合的所有时间段记录
        combined_total_asset = pd.concat(
            portfolio_histories[portfolio_index]["total_account_asset"]
        )
        combined_market_cap = pd.concat(
            portfolio_histories[portfolio_index]["holding_market_cap"]
        )
        combined_cash = pd.concat(portfolio_histories[portfolio_index]["cash_account"])

        # 处理重复日期：在调仓日会有两个值，保留后面的值（调仓后的资产）
        combined_total_asset = combined_total_asset.groupby(
            combined_total_asset.index
        ).last()
        combined_market_cap = combined_market_cap.groupby(
            combined_market_cap.index
        ).last()
        combined_cash = combined_cash.groupby(combined_cash.index).last()

        combined_portfolio_histories[portfolio_index] = {
            "total_account_asset": combined_total_asset,
            "holding_market_cap": combined_market_cap,
            "cash_account": combined_cash,
        }

    # =========================== 汇总所有组合的账户历史 ===========================
    print("汇总所有组合的账户历史...")

    # 获取所有组合的日期范围
    all_dates = set()
    for history in combined_portfolio_histories.values():
        all_dates.update(history["total_account_asset"].index)
    all_dates = sorted(all_dates)

    # 为每个组合补齐日期，处理未开始和已结束的情况
    aligned_portfolios = {}
    for portfolio_index, history in combined_portfolio_histories.items():
        # 创建完整的日期索引，默认值为每个组合的预分配资金
        aligned_series = pd.Series(portfolio_capital, index=all_dates)

        # 填入实际的资产数据
        portfolio_dates = history["total_account_asset"].index
        aligned_series.loc[portfolio_dates] = history["total_account_asset"]

        # # 处理组合结束后的情况：保持最后一天的资产值不变
        # if len(portfolio_dates) > 0:
        #     last_date = portfolio_dates.max()
        #     last_value = history["total_account_asset"].loc[last_date]

        #     # 对于组合结束后的日期，保持最后资产值
        #     future_dates = [d for d in all_dates if d > last_date]
        #     if future_dates:
        #         aligned_series.loc[future_dates] = last_value

        aligned_portfolios[portfolio_index] = aligned_series

    # 汇总所有组合的资产
    total_assets = pd.Series(0.0, index=all_dates)
    for portfolio_series in aligned_portfolios.values():
        total_assets += portfolio_series

    # 只保存total_account_asset，其他列设为0或NaN
    for date in tqdm(all_dates):
        account_history.loc[date, "total_account_asset"] = round(
            total_assets.loc[date], 2
        )
        account_history.loc[date, "holding_market_cap"] = 0  # 不再分别计算
        account_history.loc[date, "cash_account"] = 0  # 不再分别计算

    # =========================== 添加初始记录 ===========================
    #   在第一个交易日之前添加初始资本记录
    initial_date = pd.to_datetime(
        get_previous_trading_date(account_history.index.min(), 1)
    )
    account_history.loc[initial_date] = [initial_capital, 0, initial_capital]
    account_history = account_history.sort_index()
    return account_history


# 回测绩效指标计算
def get_performance_analysis(
    account_result,
    rf=0.03,
    benchmark_index="000985.XSHG",
    portfolio_weights=None,
    portfolio_count=None,
    factor_name=None,
    rank_n=None,
    save_path=None,
    show_plot=False,
):

    # 加入基准
    performance = pd.concat(
        [
            account_result["total_account_asset"].to_frame("strategy"),
            get_benchmark(account_result, benchmark_index),
        ],
        axis=1,
    )
    performance_net = performance.pct_change().dropna(how="all")  # 清算至当日开盘
    performance_cumnet = (1 + performance_net).cumprod()
    performance_cumnet["alpha"] = (
        performance_cumnet["strategy"] / performance_cumnet[benchmark_index]
    )
    performance_cumnet = performance_cumnet.fillna(1)

    # 指标计算
    performance_pct = performance_cumnet.pct_change().dropna()

    # 策略收益
    strategy_name, benchmark_name, alpha_name = performance_cumnet.columns.tolist()
    Strategy_Final_Return = performance_cumnet[strategy_name].iloc[-1] - 1

    # 策略年化收益
    Strategy_Annualized_Return_EAR = (1 + Strategy_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 基准收益
    Benchmark_Final_Return = performance_cumnet[benchmark_name].iloc[-1] - 1

    # 基准年化收益
    Benchmark_Annualized_Return_EAR = (1 + Benchmark_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # alpha
    ols_result = sm.OLS(
        performance_pct[strategy_name] * 252 - rf,
        sm.add_constant(performance_pct[benchmark_name] * 252 - rf),
    ).fit()
    Alpha = ols_result.params[0]

    # beta
    Beta = ols_result.params[1]

    # beta_2 = np.cov(performance_pct[strategy_name],performance_pct[benchmark_name])[0,1]/performance_pct[benchmark_name].var()
    # 波动率
    Strategy_Volatility = performance_pct[strategy_name].std() * np.sqrt(252)

    # 夏普
    Strategy_Sharpe = (Strategy_Annualized_Return_EAR - rf) / Strategy_Volatility

    # 下行波动率
    strategy_ret = performance_pct[strategy_name]
    Strategy_Down_Volatility = strategy_ret[strategy_ret < 0].std() * np.sqrt(252)

    # sortino
    Sortino = (Strategy_Annualized_Return_EAR - rf) / Strategy_Down_Volatility

    # 跟踪误差
    Tracking_Error = (
        performance_pct[strategy_name] - performance_pct[benchmark_name]
    ).std() * np.sqrt(252)

    # 信息比率
    Information_Ratio = (
        Strategy_Annualized_Return_EAR - Benchmark_Annualized_Return_EAR
    ) / Tracking_Error

    # 最大回测
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[strategy_name])
            - performance_cumnet[strategy_name]
        )
        / np.maximum.accumulate(performance_cumnet[strategy_name])
    )
    j = np.argmax(performance_cumnet[strategy_name][:i])
    Max_Drawdown = (
        1 - performance_cumnet[strategy_name][i] / performance_cumnet[strategy_name][j]
    )

    # 卡玛比率
    Calmar = (Strategy_Annualized_Return_EAR) / Max_Drawdown

    # 超额收益
    Alpha_Final_Return = performance_cumnet[alpha_name].iloc[-1] - 1

    # 超额年化收益
    Alpha_Annualized_Return_EAR = (1 + Alpha_Final_Return) ** (
        252 / len(performance_cumnet)
    ) - 1

    # 超额波动率
    Alpha_Volatility = performance_pct[alpha_name].std() * np.sqrt(252)

    # 超额夏普
    Alpha_Sharpe = (Alpha_Annualized_Return_EAR - rf) / Alpha_Volatility

    # 超额最大回测
    i = np.argmax(
        (
            np.maximum.accumulate(performance_cumnet[alpha_name])
            - performance_cumnet[alpha_name]
        )
        / np.maximum.accumulate(performance_cumnet[alpha_name])
    )
    j = np.argmax(performance_cumnet[alpha_name][:i])
    Alpha_Max_Drawdown = (
        1 - performance_cumnet[alpha_name][i] / performance_cumnet[alpha_name][j]
    )

    # 胜率
    performance_pct["win"] = performance_pct[alpha_name] > 0
    Win_Ratio = performance_pct["win"].value_counts().loc[True] / len(performance_pct)

    # 盈亏比
    profit_lose = performance_pct.groupby("win")[alpha_name].mean()
    Profit_Lose_Ratio = abs(profit_lose[True] / profit_lose[False])

    result = {
        "策略累计收益": round(Strategy_Final_Return, 4),
        "策略年化收益": round(Strategy_Annualized_Return_EAR, 4),
        "基准累计收益": round(Benchmark_Final_Return, 4),
        "基准年化收益": round(Benchmark_Annualized_Return_EAR, 4),
        "阿尔法": round(Alpha, 4),
        "贝塔": round(Beta, 4),
        "波动率": round(Strategy_Volatility, 4),
        "夏普比率": round(Strategy_Sharpe, 4),
        "下行波动率": round(Strategy_Down_Volatility, 4),
        "索提诺比率": round(Sortino, 4),
        "跟踪误差": round(Tracking_Error, 4),
        "信息比率": round(Information_Ratio, 4),
        "最大回撤": round(Max_Drawdown, 4),
        "卡玛比率": round(Calmar, 4),
        "超额累计收益": round(Alpha_Final_Return, 4),
        "超额年化收益": round(Alpha_Annualized_Return_EAR, 4),
        "超额波动率": round(Alpha_Volatility, 4),
        "超额夏普": round(Alpha_Sharpe, 4),
        "超额最大回撤": round(Alpha_Max_Drawdown, 4),
        "日胜率": round(Win_Ratio, 4),
        "盈亏比": round(Profit_Lose_Ratio, 4),
    }

    result_df = pd.DataFrame(list(result.items()), columns=["指标", "数值"])

    # 创建分离式策略报告：收益曲线图 + 绩效指标表
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import os

    # 设置中文字体
    rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # 生成文件名和路径
    if factor_name and portfolio_weights is not None:
        from datetime import datetime

        start_date = portfolio_weights.index[0].strftime("%Y-%m-%d")
        end_date = portfolio_weights.index[-1].strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # 分别为收益曲线和指标表格创建独立目录
        charts_dir = "/Users/didi/KDCJ/deep_model/outputs/rolling/performance_charts"
        tables_dir = "/Users/didi/KDCJ/deep_model/outputs/rolling/metrics_tables"
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        chart_filename = f"rolling_{portfolio_count}_{factor_name}_{rank_n}_{benchmark_index}_{start_date}_{end_date}_{timestamp}_chart.png"
        table_filename = f"rolling_{portfolio_count}_{factor_name}_{rank_n}_{benchmark_index}_{start_date}_{end_date}_{timestamp}_table.png"
        chart_path = os.path.join(charts_dir, chart_filename)
        table_path = os.path.join(tables_dir, table_filename)

    # ==================== 图1：收益曲线图 ====================
    fig1, ax1 = plt.subplots(figsize=(16, 9))  # 16:9比例，更适合时间序列

    # 绘制策略和基准收益曲线
    ax1.plot(
        performance_cumnet.index,
        performance_cumnet[strategy_name],
        color="#1f77b4",
        linewidth=2.5,
        label="策略收益",
        alpha=0.9,
    )
    ax1.plot(
        performance_cumnet.index,
        performance_cumnet[benchmark_name],
        color="#ff7f0e",
        linewidth=2.5,
        label="基准收益",
        alpha=0.9,
    )

    # 创建第二个y轴显示超额收益
    ax2 = ax1.twinx()
    ax2.plot(
        performance_cumnet.index,
        performance_cumnet[alpha_name],
        color="#2ca02c",
        linewidth=2,
        alpha=0.7,
        label="超额收益",
    )
    ax2.set_ylabel("超额收益", color="#2ca02c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    # 设置主图样式
    ax1.set_title(
        f"rolling_{portfolio_count}_{factor_name}_{rank_n}_收益曲线分析",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xlabel("日期", fontsize=12)
    ax1.set_ylabel("累积收益", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=11)
    ax2.legend(loc="upper right", fontsize=11)

    # 美化图表
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()

    # 保存收益曲线图
    if factor_name and portfolio_weights is not None:
        plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"收益曲线图已保存到: {chart_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ==================== 图2：绩效指标表 ====================
    fig2, ax3 = plt.subplots(figsize=(12, 16))  # 竖向布局，适合表格
    ax3.axis("off")

    # 准备表格数据
    result_df = pd.DataFrame([result]).T
    result_df.columns = ["数值"]

    # 创建表格数据
    table_data = []
    for idx, row in result_df.iterrows():
        table_data.append([idx, f"{row['数值']:.4f}"])

    # 绘制表格
    table = ax3.table(
        cellText=table_data,
        colLabels=["绩效指标", "数值"],
        cellLoc="left",
        loc="center",
        colWidths=[0.7, 0.3],
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # 更大的字体
    table.scale(1, 2.2)  # 更大的行高

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white", size=14)

    # 设置交替行颜色和样式
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor("#F8F9FA")
        # 设置数值列的字体为粗体
        table[(i, 1)].set_text_props(weight="bold")

    # 添加标题
    ax3.text(
        0.5,
        0.95,
        f"{factor_name}_{rank_n}_绩效指标表",
        transform=ax3.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="top",
    )

    plt.tight_layout()

    # 保存绩效指标表
    if factor_name and portfolio_weights is not None:
        plt.savefig(table_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"绩效指标表已保存到: {table_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # 打印结果表格到控制台
    print(pd.DataFrame([result]).T)

    return performance_cumnet, result
