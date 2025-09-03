"""
因子计算模块
负责股票数据获取和因子计算的核心逻辑
"""

import sys
import os

# 导入统一的factor_utils包
kdcj_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, kdcj_root)
try:
    from factor_utils import *

    print("成功导入factor_utils")
except ImportError as e:
    print(f"无法导入factor_utils: {e}")

from config_manager import get_config


def get_stock_data(stock_symbol, start_date, end_date):
    """
    获取股票的所有原始数据
    """
    # 技术因子数据
    tech_list = ["open", "high", "low", "close", "volume", "total_turnover"]
    daily_tech = get_price(
        stock_symbol,
        start_date,
        end_date,
        fields=tech_list,
        adjust_type="post_volume",
        skip_suspended=False,
    ).sort_index()

    # 换手率数据
    daily_tech["turnover_rate"] = get_turnover_rate(
        stock_symbol, start_date, end_date
    ).today

    # 未复权成交量
    daily_tech["unadjusted_volume"] = (
        get_price(
            stock_symbol,
            start_date,
            end_date,
            fields="volume",
            adjust_type="none",
            skip_suspended=False,
        )
        .sort_index()
        .volume
    )

    # 自由流通股本
    vol_start_date = (
        daily_tech["unadjusted_volume"].index.get_level_values("date").min()
    )
    daily_tech["stock_free_circulation"] = get_shares(
        stock_symbol, vol_start_date, end_date
    ).free_circulation

    return daily_tech


def generate_factors_for_stock(stock_symbol, end_date):
    """
    为单只股票生成所有因子数据
    """
    try:
        # 获取配置信息
        config = get_config()
        fundamental_factors = config["fundamental_factors"]
        factor_name_mapping = config["factor_name_mapping"]
        processing_config = config["processing_config"]
        column_order = config["column_order"]

        # 获取股票基本信息
        start_date = instruments(stock_symbol).listed_date
        stock_name = instruments(stock_symbol).symbol

        # 获取所有原始数据
        daily_tech = get_stock_data(stock_symbol, start_date, end_date)

        # 计算前一日收盘价
        daily_tech["prev_close"] = daily_tech["close"].shift(1)

        # 计算涨跌额
        daily_tech["change_amount"] = daily_tech["close"] - daily_tech["prev_close"]

        # 计算涨跌幅
        daily_tech["change_pct"] = (
            daily_tech["close"] - daily_tech["prev_close"]
        ) / daily_tech["prev_close"]

        # 计算振幅
        daily_tech["amplitude"] = (daily_tech["high"] - daily_tech["low"]) / daily_tech[
            "prev_close"
        ]

        # 计算自由流通股本换手率
        daily_tech["free_turnover"] = (
            daily_tech["unadjusted_volume"] / daily_tech["stock_free_circulation"]
        ) * 100

        # 计算vwap
        daily_tech["vwap"] = daily_tech["total_turnover"] / daily_tech["volume"]

        # 获取基本面因子（使用配置文件中的因子列表）
        daily_fund = get_factor(stock_symbol, fundamental_factors, start_date, end_date)

        # 合并技术因子和基本面因子
        daily_factors = daily_tech.join(daily_fund, how="outer")

        # 添加股票名称列
        daily_factors["stock_name"] = stock_name

        # 调整列顺序：交易日期、股票代码在前，然后是各种因子
        # 首先重置索引为列
        daily_factors = daily_factors.reset_index()

        # 使用配置文件中的列顺序
        daily_factors = daily_factors[column_order]

        # 将列名从英文转换为中文
        daily_factors = daily_factors.rename(columns=factor_name_mapping)

        # 保留指定位数的小数
        daily_factors = daily_factors.round(processing_config["decimal_places"])

        # 将交易日期格式改为YYYYMMDD格式（不带斜杠）
        daily_factors["交易日期"] = daily_factors["交易日期"].dt.strftime("%Y%m%d")

        # 删除第一行数据（因为prev_close计算导致第一行为NaN）
        daily_factors = daily_factors.iloc[1:]

        # 找出换手率为0的行
        zero_turnover_rows = daily_factors[daily_factors["换手率(%)"] == 0]

        if len(zero_turnover_rows) > 0:
            print(f"\n换手率为0的日期数量: {len(zero_turnover_rows)}")
            zero_turnover_dates = zero_turnover_rows["交易日期"].tolist()
            zero_turnover_dates_set = set(zero_turnover_dates)
            print(
                f"换手率为0的日期: {sorted(zero_turnover_dates)[:20]}"
            )  # 最多显示20个

        # 过滤掉换手率为0的行
        before_filter_count = len(daily_factors)
        daily_factors_filtered = daily_factors[daily_factors["换手率(%)"] > 0]
        after_filter_count = len(daily_factors_filtered)
        filtered_count = before_filter_count - after_filter_count

        if filtered_count > 0:
            print(f"过滤掉换手率为0的行数: {filtered_count}")

        # 返回过滤后的DataFrame（移除了换手率为0的行）
        return daily_factors_filtered

    except Exception as e:
        print(f"处理股票 {stock_symbol} 时出错: {str(e)}")
        return None
