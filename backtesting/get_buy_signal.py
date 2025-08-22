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

            规则（根据米筐数据格式）：
            - 上交所：主板 (60)，科创板 (68) -> .XSHG
            - 深交所：主板 (00)，创业板 (30) -> .XSHE  
            - 北交所：(43, 83, 87, 92) -> .BJSE
            """
            # 上交所：主板和科创板
            if stock_code.startswith(("60", "68")):
                return f"{stock_code}.XSHG"
            # 深交所：主板、创业板
            elif stock_code.startswith(("00", "30")):
                return f"{stock_code}.XSHE"
            # 北交所：所有类型
            elif stock_code.startswith(("43", "83", "87", "92")):
                return f"{stock_code}.BJSE"
            else:
                # 如果不匹配任何规则，保持原样并打印警告
                print(f"警告：股票代码 {stock_code} 不匹配任何交易所规则")
                return stock_code

        # 重命名列（股票代码）
        new_columns = [add_exchange_suffix(col) for col in pivot_df.columns]
        pivot_df.columns = new_columns

        print(f"转换后的股票代码示例: {list(pivot_df.columns[:5])}")

        # 1. 生成买入信号,rank_n-1是为了解决原信号文件从0开始排名
        pivot_df = pivot_df <= rank_n - 1
        pivot_df = pivot_df.astype(int).replace(0, np.nan)

        # 2. 删除从未被选中的股票（列删除）
        pivot_df = pivot_df.dropna(axis=1, how="all")

        # 3. 向后推移一天（避免未来函数）
        buy_list = pivot_df.shift(1)

        # 4. 删除完全没有信号的日期（行删除）
        buy_list = buy_list.dropna(how="all")

        # 5. 计算权重
        portfolio_weights = buy_list.div(buy_list.sum(axis=1), axis=0)

    return portfolio_weights


def get_signal_file_path(signal_filename):
    """
    获取信号文件的完整路径
    
    Args:
        signal_filename (str): 信号文件名
        hold_days (int): 持有天数，默认20天
    
    Returns:
        str: 信号文件的完整路径
    """
    import os
    
    # 基础信号目录
    base_signal_dir = "/Users/didi/KDCJ/deep_model/data/signal"
    
    # 构建完整路径
    signal_file = os.path.join(base_signal_dir, signal_filename)
    
    return signal_file


def get_output_file_path(signal_filename, rank_n, output_dir=None):
    """
    生成输出文件的完整路径
    
    Args:
        signal_filename (str): 信号文件名，格式如 "20160104_20250819_signal"
        rank_n (int): 选股数量
        output_dir (str, optional): 输出目录，默认使用统一目录
    
    Returns:
        str: 输出文件的完整路径，格式为 "startdate_enddate_rankn_weights.pkl"
    """
    import os
    import re
    
    # 默认输出目录
    if output_dir is None:
        output_dir = "/Users/didi/KDCJ/deep_model/data/cache/buy_list"
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 从信号文件名中提取日期信息
    # 假设格式为 "startdate_enddate_signal" 或 "startdate_enddate_其他"
    base_name = os.path.splitext(signal_filename)[0]  # 去掉扩展名
    
    # 使用正则表达式提取日期
    date_pattern = r'(\d{8})_(\d{8})'
    match = re.search(date_pattern, base_name)
    
    if match:
        start_date = match.group(1)
        end_date = match.group(2)
        output_filename = f"{start_date}_{end_date}_rank{rank_n}_weights.pkl"
    else:
        # 如果无法提取日期，使用原始文件名
        print(f"警告：无法从文件名 '{signal_filename}' 中提取日期，使用原始格式")
        output_filename = f"{base_name}_rank{rank_n}_weights.pkl"
    
    output_file = os.path.join(output_dir, output_filename)
    
    return output_file


if __name__ == "__main__":
    import os
    
    # 配置参数
    signal_filename = "20160104_20250619_signal"  # 信号文件名
    rank_n = 30  # 选股数量
    
    # 获取信号文件路径
    signal_file = get_signal_file_path(signal_filename)
    
    # 检查文件是否存在
    if not os.path.exists(signal_file):
        print(f"错误：信号文件不存在: {signal_file}")
        signal_dir = os.path.dirname(signal_file)
        print(f"请确保文件存在于: {signal_dir}")
        
    print(f"读取信号文件: {signal_file}")
   
   
    # 生成投资组合权重
    portfolio_weights = get_buy_signal(signal_file, rank_n=rank_n)
    
    # 获取输出文件路径
    output_file = get_output_file_path(signal_filename, rank_n)
    print(f"输出文件名: {os.path.basename(output_file)}")
    
    # 保存结果
    portfolio_weights.to_pickle(output_file)
    print(f"投资组合权重已保存到: {output_file}")
    print(f"数据形状: {portfolio_weights.shape}")
