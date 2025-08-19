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


# 获取到期日期
def get_expire_date(start_month_idx, holding_months, monthly_first_days):
    """
    根据建仓月份索引和持仓月数，计算到期日期
    :param start_month_idx: 建仓月份在monthly_first_days中的索引
    :param holding_months: 持仓月数
    :param monthly_first_days: 每月第一个交易日列表
    :return: 到期日期（pd.Timestamp）
    """
    expire_month_idx = start_month_idx + holding_months
    if expire_month_idx < len(monthly_first_days):
        return pd.Timestamp(monthly_first_days[expire_month_idx])
    else:
        # 如果超出了monthly_first_days的范围，返回最后一个日期后的N个月
        last_date = pd.Timestamp(monthly_first_days[-1])
        return last_date + relativedelta(months=holding_months)


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
def get_stock_bars(stock_price_data, portfolio_weights, adjust, price_type):
    """
    从股票价格数据中获取指定时间范围和股票列表的价格数据

    参数:
        stock_price_data: 股票价格数据（多级索引DataFrame）
        portfolio_weights: 投资组合权重矩阵
        adjust: 复权类型（'post'或'none'）
        price_type: 价格类型（'open'开盘价, 'close'收盘价）

    返回:
        价格DataFrame（日期为行索引，股票代码为列索引）
    """

    # 计算时间范围：开始日期，结束日期
    start_date = portfolio_weights.index.min()
    end_date = portfolio_weights.index.max()
    # 获取股票列表
    stock_list = portfolio_weights.columns.tolist()
    # 按时间范围和股票列表筛选数据
    filtered_data = stock_price_data.loc[
        (stock_price_data.index.get_level_values("order_book_id").isin(stock_list))
        & (stock_price_data.index.get_level_values("datetime") >= start_date)
        & (stock_price_data.index.get_level_values("datetime") <= end_date)
    ]

    # 根据价格类型和复权类型返回相应的价格数据
    # 返回开盘价
    if price_type == "open":
        # 返回后复权开盘价
        if adjust == "post":
            return filtered_data["开盘价"].unstack("order_book_id")
        # 返回未复权开盘价
        else:
            return filtered_data["未复权开盘价"].unstack("order_book_id")
    # 返回收盘价
    elif price_type == "close":
        # 返回后复权收盘价
        if adjust == "post":
            return filtered_data["收盘价"].unstack("order_book_id")
        # 返回未复权收盘价
        else:
            return filtered_data["未复权收盘价"].unstack("order_book_id")
    else:
        raise ValueError(f"不支持的价格类型: {price_type}，请使用 'open' 或 'close'")


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
    holding_months=12,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    cash_annual_yield=0.02,
    sell_timing="open",
    buy_timing="open",
):
    """
    N个月滚动持仓回测框架

    :param portfolio_weights: 投资组合权重矩阵 -> DataFrame
    :param bars_df: 股票价格数据 -> DataFrame
    :param holding_months: 持仓月数 -> int (默认12个月，可设置为2,3,4等进行快速调试)
    :param initial_capital: 初始资本 -> float
    :param stamp_tax_rate: 印花税率 -> float
    :param transfer_fee_rate: 过户费率 -> float
    :param commission_rate: 佣金率 -> float
    :param cash_annual_yield: 现金年化收益率 -> float
    :param sell_timing: 卖出时点 -> str ('默认open开盘价')
    :param buy_timing: 买入时点 -> str ('默认open开盘价')
    :return: 账户历史记录 -> DataFrame
    """

    # =========================== 基础参数初始化 ===========================
    # 每月投入资金（总资金的 1/holding_months）
    monthly_capital = initial_capital / holding_months
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

    # 获取卖出价格数据-后复权
    sell_prices = get_stock_bars(bars_df, portfolio_weights, "post", sell_timing)
    # 获取买入价格数据-后复权
    buy_prices = get_stock_bars(bars_df, portfolio_weights, "post", buy_timing)
    # 获取每月第一个交易日
    start_date = portfolio_weights.index.min()
    end_date = portfolio_weights.index.max()
    monthly_first_days = get_monthly_first_trading_days(start_date, end_date)

    # =========================== N个月组合管理 ===========================
    # 初始化N个月组合的信息和历史记录
    monthly_portfolios = {}
    portfolio_histories = {}

    for i in range(holding_months):
        # 初始化组合信息
        monthly_portfolios[i] = {
            "holdings": pd.Series(dtype=float),  # 持仓股票及数量
            "cash": 0.0,  # 现金余额
            "start_date": None,  # 建仓日期
            "expire_date": None,  # 到期日期
            "is_active": False,  # 是否激活
        }
        # 初始化历史记录
        portfolio_histories[i] = {
            "total_account_asset": [],
            "holding_market_cap": [],
            "cash_account": [],
        }

    # =========================== 开始逐月建仓和调仓 ===========================
    portfolio_index = 0  # 当前使用的组合索引（0到holding_months-1循环）

    for month_idx, month_date_raw in enumerate(tqdm(monthly_first_days)):

        # 统一转换为pd.Timestamp类型
        rebalance_date = pd.Timestamp(month_date_raw)
        
        # 调试：组合0的第一次调仓（第13个月，portfolio_index=0）
        # if month_idx == 12 and portfolio_index == 0:
        #     print(f"调试点：组合0的第一次调仓，日期：{rebalance_date}")
        #     breakpoint()

        # 检查该调仓日期是否在portfolio_weights的索引中
        if rebalance_date not in portfolio_weights.index:
            continue

        # 获取当前调仓日的目标权重
        current_target_weights = portfolio_weights.loc[rebalance_date].dropna()
        if len(current_target_weights) == 0:
            continue
        # 获取当前调仓日的目标股票
        target_stocks = current_target_weights.index.tolist()
        # 获取当前调仓日的卖出和买入价格
        current_sell_prices = sell_prices.loc[rebalance_date]
        current_buy_prices = buy_prices.loc[rebalance_date]

        # =========================== 处理当前组合，更新持仓 ===========================
        current_portfolio = monthly_portfolios[portfolio_index]

        # 检查组合状态和到期情况
        if current_portfolio["is_active"]:
            # 检查组合是否到期
            if rebalance_date < current_portfolio["expire_date"]:
                # 组合未到期，跳过操作，移动到下一个组合
                portfolio_index = (portfolio_index + 1) % holding_months
                continue

            # 组合到期，进行调仓
            print(f"组合{portfolio_index}号到期调仓，日期：{rebalance_date}")

            # 从该组合的历史记录中获取上一期最后一天的总资产作为可用资金
            if len(portfolio_histories[portfolio_index]["total_account_asset"]) > 0:
                # 取最后一个时间段的最后一天的值
                last_period_records = portfolio_histories[portfolio_index][
                    "total_account_asset"
                ][-1]
                total_asset_before_rebalance = last_period_records.iloc[-1]  # 最后一天的总资产
            else:
                # 如果没有历史记录，使用月度资金（理论上不应该发生）
                total_asset_before_rebalance = monthly_capital
                print(f"警告：未找到历史记录，使用月度资金: {total_asset_before_rebalance:.2f}")

            # 重置组合的到期日期（新的N个月周期）
            current_portfolio["expire_date"] = get_expire_date(
                month_idx, holding_months, monthly_first_days
            )
            current_portfolio["start_date"] = rebalance_date  # 更新开始日期

        else:
            # 新建仓，使用固定的月度资金
            print(f"组合{portfolio_index}号首次建仓，日期：{rebalance_date}")
            available_cash = monthly_capital
            current_portfolio["is_active"] = True
            current_portfolio["start_date"] = rebalance_date
            # 计算到期日期（N个月后的第一个交易日）
            current_portfolio["expire_date"] = get_expire_date(
                month_idx, holding_months, monthly_first_days
            )

        # =========================== 全清仓全持仓逻辑 ===========================
        # 第一步：如果有持仓，先全部清仓
        sell_cost = 0.0
        if len(current_portfolio["holdings"]) > 0:
            # 计算当前持仓市值
            current_market_value = (current_portfolio["holdings"] * current_sell_prices).sum()
            # 计算卖出手续费
            sell_cost = current_market_value * sell_cost_rate
            # 清仓后的可用资金 = 总资产 - 卖出手续费
            available_cash = total_asset_before_rebalance - sell_cost

        # 第二步：计算买入手续费并分配现金
        buy_cost = available_cash * buy_cost_rate
        investable_cash = available_cash - buy_cost

        # 第三步：等权分配现金（矩阵运算）
        if len(target_stocks) > 0:
            # 使用矩阵运算计算目标持仓（等权分配）
            target_holdings = (
                current_target_weights * investable_cash / current_buy_prices.loc[target_stocks]
            ).round(4)  # 保疙4位小数

            # 计算实际投资金额
            actual_investment = (target_holdings * current_buy_prices.loc[target_stocks]).sum()

            # 计算剩余现金
            remaining_cash = investable_cash - actual_investment

        else:
            target_holdings = pd.Series(dtype=float)
            remaining_cash = investable_cash

        # 更新组合信息
        current_portfolio["holdings"] = target_holdings
        current_portfolio["cash"] = remaining_cash

        # 计算持仓市值
        start_date = rebalance_date
        end_date = current_portfolio["expire_date"]
        # 只获取持仓股票的价格数据
        held_stocks = target_holdings.index.tolist()
        period_post_prices = buy_prices.loc[start_date:end_date, held_stocks]
        # 计算每日持仓市值：DataFrame(价格) * Series(持仓数量)
        # Pandas会自动按列索引(股票代码)对齐进行广播运算
        # 结果：每日每只股票市值 = 当日价格 × 持仓数量
        # sum(axis=1)：按行求和，得到每日总持仓市值
        portfolio_market_value = (period_post_prices * target_holdings).sum(axis=1)

        # =========================== 计算现金账户余额 ===========================
        # 计算期间现金账户的复利增长（按日计息）
        cash_balance = pd.Series(
            [
                current_portfolio["cash"]
                * ((1 + daily_cash_yield) ** day)  # 复利计息公式
                for day in range(0, len(portfolio_market_value))
            ],  # 对每一天计算
            index=portfolio_market_value.index,
        )  # 使用相同的日期索引

        # =========================== 计算账户总资产 ===========================
        total_portfolio_value = (
            portfolio_market_value + cash_balance
        )  # 总资产 = 持仓市值 + 现金余额

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
        portfolio_index = (portfolio_index + 1) % holding_months

    # =========================== 连接每个组合的多个时间段记录 ===========================
    print("正在连接每个组合的多个时间段记录...")

    # 为每个组合连接其所有时间段的记录
    combined_portfolio_histories = {}

    for portfolio_index in range(holding_months):
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
    print("正在汇总所有组合的账户历史...")

    # 获取所有组合的日期范围
    all_dates = set()
    for history in combined_portfolio_histories.values():
        all_dates.update(history["total_account_asset"].index)
    all_dates = sorted(all_dates)

    # 为每个组合补齐日期，处理未开始和已结束的情况
    aligned_portfolios = {}
    for portfolio_index, history in combined_portfolio_histories.items():
        # 创建完整的日期索引，默认值为每个组合的预分配资金
        aligned_series = pd.Series(monthly_capital, index=all_dates)

        # 填入实际的资产数据
        portfolio_dates = history["total_account_asset"].index
        aligned_series.loc[portfolio_dates] = history["total_account_asset"]

        # 处理组合结束后的情况：保持最后一天的资产值不变
        if len(portfolio_dates) > 0:
            last_date = portfolio_dates.max()
            last_value = history["total_account_asset"].loc[last_date]

            # 对于组合结束后的日期，保持最后资产值
            future_dates = [d for d in all_dates if d > last_date]
            if future_dates:
                aligned_series.loc[future_dates] = last_value

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
    # 在第一个交易日之前添加初始资本记录
    initial_date = pd.to_datetime(
        get_previous_trading_date(account_history.index.min(), 1)
    )
    account_history.loc[initial_date] = [initial_capital, 0, initial_capital]
    account_history = account_history.sort_index()
    return account_history


# 回测绩效指标计算
def get_performance_analysis(account_result, rf=0.03, benchmark_index="000985.XSHG"):

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
        "策略累计收益": round(Strategy_Final_Return, 6),
        "策略年化收益": round(Strategy_Annualized_Return_EAR, 6),
        "基准累计收益": round(Benchmark_Final_Return, 6),
        "基准年化收益": round(Benchmark_Annualized_Return_EAR, 6),
        "阿尔法": round(Alpha, 6),
        "贝塔": round(Beta, 6),
        "波动率": round(Strategy_Volatility, 6),
        "夏普比率": round(Strategy_Sharpe, 6),
        "下行波动率": round(Strategy_Down_Volatility, 6),
        "索提诺比率": round(Sortino, 6),
        "跟踪误差": round(Tracking_Error, 6),
        "信息比率": round(Information_Ratio, 6),
        "最大回撤": round(Max_Drawdown, 6),
        "卡玛比率": round(Calmar, 6),
        "超额累计收益": round(Alpha_Final_Return, 6),
        "超额年化收益": round(Alpha_Annualized_Return_EAR, 6),
        "超额波动率": round(Alpha_Volatility, 6),
        "超额夏普": round(Alpha_Sharpe, 6),
        "超额最大回撤": round(Alpha_Max_Drawdown, 6),
        "日胜率": round(Win_Ratio, 6),
        "盈亏比": round(Profit_Lose_Ratio, 6),
    }

    result_df = pd.DataFrame(list(result.items()), columns=["指标", "数值"])
    return performance_cumnet, result_df


def plot_backtest_performance(
    performance_cumnet, benchmark_index, figsize=(12, 8), save_chart=True
):
    """
    绘制策略回测结果图表

    参数:
    performance_cumnet: DataFrame, 包含策略、基准和alpha的累计收益数据
    benchmark_index: str, 基准指数代码
    figsize: tuple, 图表大小
    save_chart: bool, 是否保存图表到文件

    返回:
    fig: matplotlib图表对象
    """
    import os
    from datetime import datetime

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建图表和双轴
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # 获取列名
    columns = performance_cumnet.columns.tolist()
    strategy_col = columns[0]  # 策略列
    benchmark_col = columns[1]  # 基准列
    alpha_col = columns[2]  # 超额收益列

    # 绘制左轴：策略和基准累计收益
    line1 = ax1.plot(
        performance_cumnet.index,
        performance_cumnet[strategy_col],
        color="red",
        linewidth=2,
        label="策略",
    )
    line2 = ax1.plot(
        performance_cumnet.index,
        performance_cumnet[benchmark_col],
        color="blue",
        linewidth=2,
        label=benchmark_index,
    )

    # 绘制右轴：超额累计收益
    line3 = ax2.plot(
        performance_cumnet.index,
        performance_cumnet[alpha_col],
        color="green",
        linewidth=2,
        label="超额收益",
    )

    # 设置左轴标签和格式
    ax1.set_xlabel("日期", fontsize=12)
    ax1.set_ylabel("策略累计收益", fontsize=12, color="red")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)

    # 设置右轴标签和格式
    ax2.set_ylabel("超额累计收益", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelcolor="black")

    # 设置图表标题
    plt.title(
        f"策略回测（对应基准：{benchmark_index}）",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表到文件
    if save_chart:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 创建保存图片的文件夹
        output_dir = os.path.join(current_dir, "backtest_charts")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建文件夹: {output_dir}")

        # 生成带时间戳的文件名（精确到分钟）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"策略回测结果_{benchmark_index}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        # 保存图片
        fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"图片已保存至: {filepath}")

    return fig
