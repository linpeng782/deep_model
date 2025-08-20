import pandas as pd
import numpy as np
import os
import re


def get_buy_signal(file_path, rank_n=9):
    """读取top1k排名文件。

    参数:
        file_path: top1k文件的完整路径

    返回:
        包含日期、股票代码和排名的DataFrame
    """
    print(f"读取文件: {file_path}")

    # 读取文件数据
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 3:
                        date = parts[0]
                        stock_code = parts[1]
                        rank = int(parts[2])
                        data.append(
                            {"日期": date, "股票代码": stock_code, "排名": rank}
                        )
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 转换为DataFrame
    df = pd.DataFrame(data)

    if not df.empty:
        # 转换日期格式
        df["日期"] = pd.to_datetime(df["日期"], format="%Y%m%d")

        # 将数据重构为透视表格式
        # 日期作为行索引，股票代码作为列标签，排名作为值
        print("\n重构数据为透视表格式...")
        pivot_df = df.pivot_table(
            index="日期",  # 日期作为行索引
            columns="股票代码",  # 股票代码作为列标签
            values="排名",  # 排名作为数值
            aggfunc="first",  # 如果有重复，取第一个值
        )

        # 添加交易所后缀到股票代码
        print("\n为股票代码添加交易所后缀...")

        def add_exchange_suffix(stock_code):
            """
            根据股票代码添加相应的交易所后缀

            规则（根据最新规则）：
            - 上交所：主板 (60)，科创板 (68) -> .SH
            - 深交所：主板 (00)，创业板 (30) -> .SZ
            - 北交所：普通股票 (8) -> .BJ
            """
            # 上交所：主板和科创板
            if stock_code.startswith(("60", "68")):
                return f"{stock_code}.SH"
            # 深交所：主板、创业板
            elif stock_code.startswith(("00", "30")):
                return f"{stock_code}.SZ"
            # 北交所：所有类型
            elif stock_code.startswith(("8")):
                return f"{stock_code}.BJ"
            else:
                # 如果不匹配任何规则，保持原样并打印警告
                print(f"警告：股票代码 {stock_code} 不匹配任何交易所规则")
                return stock_code

        # 重命名列（股票代码）
        new_columns = [add_exchange_suffix(col) for col in pivot_df.columns]
        pivot_df.columns = new_columns

        print(f"转换后的股票代码示例: {list(pivot_df.columns[:5])}")

        # 1. 生成买入信号
        pivot_df = pivot_df <= rank_n
        pivot_df = pivot_df.astype(int).replace(0, np.nan)

        # 2. 删除从未被选中的股票（列删除）
        pivot_df = pivot_df.dropna(how="all", axis=1)

        # 3. 向后推移一天（避免未来函数）
        buy_list = pivot_df.shift(1)

        # 4. 删除完全没有信号的日期（行删除）
        buy_list = buy_list.dropna(how="all")

        # 5. 计算权重
        portfolio_weights = buy_list.div(buy_list.sum(axis=1), axis=0)

    return portfolio_weights


if __name__ == "__main__":
    # 读取top1k排名文件
    signal_file = "/Users/didi/Projects/BackTest/top1k_l5_unclip_N6.0.pred.250.trim"
    portfolio_weights = get_buy_signal(signal_file, rank_n=9)
    portfolio_weights.to_pickle("portfolio_weights.pkl")
    print("portfolio_weights.pkl已保存")
