import sys
import os
import pandas as pd
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import RAW_DATA_DIR, ENHANCED_DATA_DIR
from config.settings import get_timestamped_output_path

# 导入统一的factor_utils包
kdcj_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, kdcj_root)
try:
    from factor_utils import *

    print("成功导入factor_utils")
except ImportError as e:
    print(f"无法导入factor_utils: {e}")

# 因子中英文映射字典
factor_name_mapping = {
    # 基本信息
    "date": "交易日期",
    "order_book_id": "股票代码",
    "stock_name": "股票简称",
    # 技术因子
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "prev_close": "昨收价",
    "change_amount": "涨跌额",
    "change_pct": "涨跌幅",
    "amplitude": "振幅",
    "unadjusted_volume": "未复权成交量",
    "volume": "成交量",
    "total_turnover": "成交额",
    "turnover_rate": "换手率(%)",
    "free_turnover": "换手率(自由流通股)",
    "stock_free_circulation": "自由流通股本",
    # 基本面因子
    "pe_ratio_lyr": "市盈率_最近年报",
    "pe_ratio_ttm": "市盈率_TTM",
    "pb_ratio_lyr": "市净率_最近年报",
    "pb_ratio_ttm": "市净率_TTM",
    "pb_ratio_lf": "市净率_最新财报",
    "ps_ratio_lyr": "市销率_最近年报",
    "ps_ratio_ttm": "市销率_TTM",
    "dividend_yield_ttm": "股息率_TTM",
    "market_cap_3": "总市值",
    "market_cap_2": "流通市值",
}


def parse_stock_info_from_filename(filename):
    """
    从文件名解析股票信息
    输入: "000001.SZ-平安银行-日线后复权及常用指标-20250718.csv"
    输出: ("000001.SZ", "平安银行", "20250718")
    """
    pattern = r"([0-9]{6}\.[A-Z]{2})-(.+?)-日线后复权及常用指标-(\d{8})\.csv"
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def convert_stock_code(original_code):
    """
    转换股票代码格式
    SZ -> XSHE, SH -> XSHG, BJ -> BJSE
    """
    if original_code.endswith(".SZ"):
        return original_code.replace(".SZ", ".XSHE")
    elif original_code.endswith(".SH"):
        return original_code.replace(".SH", ".XSHG")
    elif original_code.endswith(".BJ"):
        return original_code.replace(".BJ", ".BJSE")
    else:
        return None  # 其他格式暂不处理


def generate_factors_for_stock(stock_symbol, end_date):
    """
    为单只股票生成所有因子数据
    """
    try:

        # 获取股票基本信息
        start_date = instruments(stock_symbol).listed_date
        stock_name = instruments(stock_symbol).symbol

        # 技术因子
        tech_list = ["open", "high", "low", "close", "volume", "total_turnover"]

        daily_tech = get_price(
            stock_symbol,
            start_date,
            end_date,
            fields=tech_list,
            adjust_type="post",
            skip_suspended=False,
        ).sort_index()

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

        # 流通股本换手率
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

        # 自由流动股本换手率
        # 获取unadjusted_volume的起始日期
        vol_start_date = (
            daily_tech["unadjusted_volume"].index.get_level_values("date").min()
        )

        # 使用vol_start_date获取stock_free_circulation的截面
        daily_tech["stock_free_circulation"] = get_shares(
            stock_symbol, vol_start_date, end_date
        ).free_circulation

        # 计算自由流通股本换手率
        daily_tech["free_turnover"] = (
            daily_tech["unadjusted_volume"] / daily_tech["stock_free_circulation"]
        ) * 100

        # 基本面因子
        fund_list = [
            "pe_ratio_lyr",
            "pe_ratio_ttm",
            "pb_ratio_lyr",
            "pb_ratio_ttm",
            "pb_ratio_lf",
            "ps_ratio_lyr",
            "ps_ratio_ttm",
            "dividend_yield_ttm",
            "market_cap_3",
            "market_cap_2",
            "ep_ratio_ttm",
            "book_to_market_ratio_ttm",
            "ebit_ttm",
            "ebitda_ttm",
            "ebit_per_share_ttm",
            "return_on_equity_lyr",
            "return_on_equity_ttm",
        ]

        # 获取基本面因子
        daily_fund = get_factor(stock_symbol, fund_list, start_date, end_date)

        # 合并技术因子和基本面因子
        daily_factors = daily_tech.join(daily_fund, how="outer")

        # 添加股票名称列
        daily_factors["stock_name"] = stock_name

        # 调整列顺序：交易日期、股票代码在前，然后是各种因子
        # 首先重置索引为列
        daily_factors = daily_factors.reset_index()

        desired_order = [
            "date",
            "order_book_id",
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "change_amount",
            "change_pct",
            "amplitude",
            "unadjusted_volume",
            "volume",
            "total_turnover",
            "turnover_rate",
            "stock_free_circulation",
            "free_turnover",
            "pe_ratio_lyr",
            "pe_ratio_ttm",
            "pb_ratio_lyr",
            "pb_ratio_ttm",
            "pb_ratio_lf",
            "ps_ratio_lyr",
            "ps_ratio_ttm",
            "dividend_yield_ttm",
            "market_cap_3",
            "market_cap_2",
        ]

        # 重新排列列顺序
        daily_factors = daily_factors[desired_order]

        # 将列名从英文转换为中文
        daily_factors = daily_factors.rename(columns=factor_name_mapping)

        # 保留4位小数
        daily_factors = daily_factors.round(4)

        # 将交易日期格式改为YYYYMMDD格式（不带斜杠）
        daily_factors["交易日期"] = daily_factors["交易日期"].dt.strftime("%Y%m%d")

        # 删除第一行数据（因为prev_close计算导致第一行为NaN）
        daily_factors = daily_factors.iloc[1:]

        # 读取原始平安银行 CSV 文件进行对比
        if stock_symbol == "000001.XSHE":
            original_csv_path = os.path.join(
                RAW_DATA_DIR, "000001.SZ-平安银行-日线后复权及常用指标-20250718.csv"
            )
            try:
                original_df = pd.read_csv(original_csv_path, encoding="utf-8")

                # 使用daily_factors的第一天作为起始日期来截取原始CSV
                factors_start_date = daily_factors["交易日期"].min()
                factors_end_date = daily_factors["交易日期"].max()

                # 过滤原始CSV数据，只保留与daily_factors相同时间范围的数据
                original_df["交易日期"] = original_df["交易日期"].astype(str)
                original_filtered = original_df[
                    (original_df["交易日期"] >= factors_start_date)
                    & (original_df["交易日期"] <= factors_end_date)
                ]

                original_dates = set(original_filtered["交易日期"].tolist())
                new_dates = set(daily_factors["交易日期"].tolist())

                print(
                    f"\n=== 交易日期对比分析 ({factors_start_date} ~ {factors_end_date}) ==="
                )
                print(f"原始CSV日期数量: {len(original_dates)}")
                print(f"新生成数据日期数: {len(new_dates)}")

                missing_in_new = original_dates - new_dates
                extra_in_new = new_dates - original_dates

                if missing_in_new:
                    print(f"新数据中缺失的日期数: {len(missing_in_new)}")
                    print(f"缺失日期示例: {sorted(list(missing_in_new))[:10]}")

                if extra_in_new:
                    print(f"新数据中多余的日期数: {len(extra_in_new)}")
                    print(f"多余日期示例: {sorted(list(extra_in_new))[:10]}")

                if not missing_in_new and not extra_in_new:
                    print("交易日期完全一致")
                else:
                    print("交易日期存在差异")
                print("=" * 30)

            except Exception as e:
                print(f"读取原始CSV文件失败: {e}")

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

        # 如果是平安银行，对比过滤后的日期与原始CSV日期
        if stock_symbol == "000001.XSHE" and "original_dates" in locals():
            filtered_dates = set(daily_factors_filtered["交易日期"].tolist())

            print(f"\n=== 过滤后日期对比分析 ===")
            print(f"原始CSV日期数量: {len(original_dates)}")
            print(f"过滤后数据日期数: {len(filtered_dates)}")

            missing_in_filtered = original_dates - filtered_dates
            extra_in_filtered = filtered_dates - original_dates

            if missing_in_filtered:
                print(f"过滤后数据中缺失的日期数: {len(missing_in_filtered)}")
                print(f"缺失日期示例: {sorted(list(missing_in_filtered))[:10]}")

            if extra_in_filtered:
                print(f"过滤后数据中多余的日期数: {len(extra_in_filtered)}")
                print(f"多余日期示例: {sorted(list(extra_in_filtered))[:10]}")

            if not missing_in_filtered and not extra_in_filtered:
                print("过滤后交易日期与原始CSV完全一致")
            else:
                print("过滤后交易日期与原始CSV存在差异")
            print("=" * 30)

        # 返回过滤后的DataFrame（移除了换手率为0的行）
        return daily_factors_filtered

    except Exception as e:
        print(f"处理股票 {stock_symbol} 时出错: {str(e)}")
        return None


def get_stock_list_from_csv_folder(csv_folder_path, limit=None):
    """
    从CSV文件夹获取股票列表
    """
    csv_folder = Path(csv_folder_path)
    stock_list = []

    for csv_file in csv_folder.glob("*.csv"):
        original_code, stock_name, date = parse_stock_info_from_filename(csv_file.name)

        if original_code and stock_name:
            # 转换股票代码
            converted_code = convert_stock_code(original_code)

            if converted_code:  # 处理SZ、SH和BJ股票
                stock_list.append(
                    {
                        "original_code": original_code,
                        "converted_code": converted_code,
                        "stock_name": stock_name,
                        "date": date,
                    }
                )

    # 限制处理数量（用于测试）
    if limit:
        stock_list = stock_list[:limit]

    return stock_list


def batch_process_stocks_parallel(
    csv_folder_path, output_folder_path, end_date, limit=None, max_workers=4
):
    """
    并行批量处理股票因子数据
    """
    # 获取股票列表
    stock_list = get_stock_list_from_csv_folder(csv_folder_path, limit)

    if not stock_list:
        print("未找到任何股票CSV文件")
        return

    print(f"准备并行处理 {len(stock_list)} 只股票，使用 {max_workers} 个线程")

    # 创建输出文件夹
    output_folder = Path(output_folder_path)
    output_folder.mkdir(exist_ok=True)

    # 删除旧文件
    for old_file in output_folder.glob("*.csv"):
        old_file.unlink()

    success_count = 0
    failed_count = 0

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 直接提交generate_factors_for_stock任务
        future_to_stock = {
            executor.submit(
                generate_factors_for_stock, stock_info["converted_code"], end_date
            ): stock_info
            for stock_info in stock_list
        }

        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_stock), 1):
            stock_info = future_to_stock[future]
            stock_symbol = stock_info["converted_code"]
            stock_name = stock_info["stock_name"]

            try:
                factors_df = future.result()

                if factors_df is not None:
                    # 在主线程中保存文件
                    output_filename = f"{stock_symbol}-{stock_name}-日线后复权及常用指标-{end_date}.csv"
                    output_path = output_folder / output_filename
                    factors_df.to_csv(output_path, encoding="utf-8", index=False)

                    success_count += 1
                    print(
                        f"进度: {i}/{len(stock_list)} - 成功: {output_filename} - 形状: {factors_df.shape}"
                    )
                else:
                    failed_count += 1
                    print(
                        f"进度: {i}/{len(stock_list)} - 失败: {stock_symbol} - 错误: 生成因子失败"
                    )

            except Exception as e:
                failed_count += 1
                print(
                    f"进度: {i}/{len(stock_list)} - 失败: {stock_symbol} - 错误: {str(e)}"
                )

    print(f"\n处理完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {failed_count} 只")
    print(f"总计: {len(stock_list)} 只")


def batch_process_stocks(csv_folder_path, output_folder_path, end_date, limit=None):
    """
    批量处理股票因子数据
    """
    # 创建输出文件夹
    output_folder = Path(output_folder_path)
    output_folder.mkdir(exist_ok=True)

    # 获取股票列表
    stock_list = get_stock_list_from_csv_folder(csv_folder_path, limit)

    print(f"准备处理 {len(stock_list)} 只股票")

    success_count = 0
    error_count = 0

    for i, stock_info in enumerate(stock_list, 1):
        print(f"\n进度: {i}/{len(stock_list)}")

        # 生成因子数据
        factors_df = generate_factors_for_stock(stock_info["converted_code"], end_date)

        if factors_df is not None:
            # 生成输出文件名（使用米筐格式代码和统一的结束日期）
            output_filename = f"{stock_info['converted_code']}-{stock_info['stock_name']}-日线后复权及常用指标-{end_date}.csv"
            output_path = output_folder / output_filename

            # 保存CSV文件
            factors_df.to_csv(output_path, encoding="utf-8")
            print(f"成功保存: {output_filename}")
            success_count += 1
        else:
            print(f"处理失败: {stock_info['converted_code']}")
            error_count += 1

        # 添加延时避免API限制


def get_failed_stocks(csv_folder_path, output_folder_path, end_date):
    """
    获取尚未创建输出CSV文件的股票列表
    """
    # 获取所有股票列表
    all_stocks = get_stock_list_from_csv_folder(csv_folder_path)

    # 获取已存在的输出CSV文件
    output_folder = Path(output_folder_path)
    existing_files = set()

    if output_folder.exists():
        for csv_file in output_folder.glob("*.csv"):
            existing_files.add(csv_file.name)

    # 找出尚未创建的股票
    failed_stocks = []
    for stock_info in all_stocks:
        expected_filename = f"{stock_info['converted_code']}-{stock_info['stock_name']}-日线后复权及常用指标-{end_date}.csv"
        if expected_filename not in existing_files:
            failed_stocks.append(stock_info)

    return failed_stocks


def retry_failed_stocks(
    csv_folder_path, output_folder_path, end_date, use_parallel=False, max_workers=2
):
    """
    重试处理失败的股票（尚未创建输出CSV文件的股票）
    """
    failed_stocks = get_failed_stocks(csv_folder_path, output_folder_path, end_date)

    if not failed_stocks:
        print("✅ 所有股票都已成功处理，无需重试")
        return

    print(f"找到 {len(failed_stocks)} 只尚未处理的股票")
    print(f"示例: {[stock['converted_code'] for stock in failed_stocks[:5]]}")

    # 创建输出文件夹
    output_folder = Path(output_folder_path)
    output_folder.mkdir(exist_ok=True)

    success_count = 0
    failed_count = 0

    if use_parallel:
        print(f"使用并行模式重试，{max_workers} 个线程")

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(
                    generate_factors_for_stock,
                    stock_info["converted_code"],
                    end_date,
                ): stock_info
                for stock_info in failed_stocks
            }

            for i, future in enumerate(as_completed(future_to_stock), 1):
                stock_info = future_to_stock[future]
                stock_symbol = stock_info["converted_code"]
                stock_name = stock_info["stock_name"]

                try:
                    factors_df = future.result()

                    if factors_df is not None:
                        output_filename = f"{stock_symbol}-{stock_name}-日线后复权及常用指标-{end_date}.csv"
                        output_path = output_folder / output_filename
                        factors_df.to_csv(output_path, encoding="utf-8", index=False)

                        success_count += 1
                        print(
                            f"重试进度: {i}/{len(failed_stocks)} - 成功: {output_filename}"
                        )
                    else:
                        failed_count += 1
                        print(
                            f"重试进度: {i}/{len(failed_stocks)} - 失败: {stock_symbol} - 错误: 生成因子失败"
                        )

                except Exception as e:
                    failed_count += 1
                    print(
                        f"重试进度: {i}/{len(failed_stocks)} - 失败: {stock_symbol} - 错误: {str(e)}"
                    )
    else:
        print(f"使用串行模式重试")

        for i, stock_info in enumerate(failed_stocks, 1):
            print(
                f"\n重试进度: {i}/{len(failed_stocks)} - {stock_info['converted_code']}"
            )

            try:
                factors_df = generate_factors_for_stock(
                    stock_info["converted_code"], end_date
                )

                if factors_df is not None:
                    output_filename = f"{stock_info['converted_code']}-{stock_info['stock_name']}-日线后复权及常用指标-{end_date}.csv"
                    output_path = output_folder / output_filename
                    factors_df.to_csv(output_path, encoding="utf-8", index=False)
                    print(f"成功保存: {output_filename}")
                    success_count += 1
                else:
                    print(f"失败: {stock_info['converted_code']}")
                    failed_count += 1

                # 串行模式下添加延迟避免API限制
                time.sleep(1)

            except Exception as e:
                print(f"失败: {stock_info['converted_code']} - 错误: {str(e)}")
                failed_count += 1

    print(f"\n重试完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {failed_count} 只")
    print(f"重试总计: {len(failed_stocks)} 只")


def test_single_stock(stock_symbol, output_folder_path, end_date="20250718"):
    """
    测试单只股票
    """
    print(f"测试单只股票: {stock_symbol}")
    factors_df = generate_factors_for_stock(stock_symbol, end_date)

    if factors_df is not None:
        # 创建输出文件夹
        from pathlib import Path

        output_folder = Path(output_folder_path)
        output_folder.mkdir(exist_ok=True)

        # 获取股票名称
        stock_name = instruments(stock_symbol).symbol

        # 保存CSV文件
        output_filename = (
            f"{stock_symbol}-{stock_name}-日线后复权及常用指标-{end_date}.csv"
        )
        output_path = output_folder / output_filename
        factors_df.to_csv(output_path, encoding="utf-8", index=False)
        print(f"成功保存: {output_filename}")
        print(f"数据形状: {factors_df.shape}")
        print(f"列名: {list(factors_df.columns)}")
        return True
    else:
        print("处理失败")
        return False


# 每日手动调整的日期
END_DATE = "20250819"  # 格式: YYYYMMDD

if __name__ == "__main__":
    end_date = END_DATE

    # 从配置文件获取路径
    csv_folder_path = RAW_DATA_DIR

    # 生成带时间戳的输出路径
    output_folder_path = get_timestamped_output_path(ENHANCED_DATA_DIR)
    timestamp = os.path.basename(output_folder_path).split("_")[-1]

    print(f"输入路径: {csv_folder_path}")
    print(f"输出路径: {output_folder_path}")
    print(f"结束日期: {end_date}")
    print(f"时间戳: {timestamp}")

    # 选择测试模式
    test_mode = "single"  # "single", "batch", 或 "retry_failed"

    if test_mode == "single":
        # 测试单只股票
        stock_code = "000001.XSHE"
        test_single_stock(stock_code, output_folder_path, end_date)

    elif test_mode == "batch":
        # 选择并行或串行处理
        use_parallel = True  # 设置为False使用串行处理
        limit = None  # 处理所有股票

        if use_parallel:
            print(f"并行批量测试所有股票")
            max_workers = 4
            batch_process_stocks_parallel(
                csv_folder_path,
                output_folder_path,
                end_date,
                limit=limit,
                max_workers=max_workers,
            )
        else:
            print(f"串行批量测试所有股票")
            batch_process_stocks(
                csv_folder_path, output_folder_path, end_date, limit=limit
            )

    elif test_mode == "retry_failed":
        # 重试失败的股票
        use_parallel_retry = True
        max_workers_retry = 2
        print(f"重试处理失败的股票")
        retry_failed_stocks(
            csv_folder_path,
            output_folder_path,
            end_date,
            use_parallel=use_parallel_retry,
            max_workers=max_workers_retry,
        )
