import sys
import os
import pandas as pd
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import yaml

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from deep_model.config.paths import RAW_DATA_DIR, ENHANCED_DATA_DIR
from deep_model.config.settings import get_timestamped_output_path

# 导入统一的factor_utils包
kdcj_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, kdcj_root)
try:
    from factor_utils import *

    print("成功导入factor_utils")
except ImportError as e:
    print(f"无法导入factor_utils: {e}")


def load_factor_config(config_path=None):
    """
    加载因子配置文件
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "factor_config.yaml"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        # 返回默认配置
        return get_default_config()


def get_default_config():
    """
    获取默认配置（作为配置文件加载失败时的备选方案）
    """
    return {
        "factor_name_mapping": {
            "date": "交易日期",
            "order_book_id": "股票代码",
            "stock_name": "股票简称",
            "close": "收盘价",
            "turnover_rate": "换手率(%)",
        },
        "fundamental_factors": ["pe_ratio_ttm", "pb_ratio_ttm", "market_cap_3"],
        "column_order": [
            "date",
            "order_book_id",
            "stock_name",
            "close",
            "turnover_rate",
        ],
        "processing_config": {
            "default_end_date": "20250901",
            "max_workers": 4,
            "decimal_places": 4,
            "test_mode": "single",
        },
    }


# 加载配置
config = load_factor_config()
factor_name_mapping = config["factor_name_mapping"]
fundamental_factors = config["fundamental_factors"]
processing_config = config["processing_config"]
column_order = config["column_order"]


def parse_stock_info_from_filename(filename):
    """
    从文件名解析股票信息
    输入: "000001.SZ-股票名称-日线后复权及常用指标-20250718.csv"
    输出: ("000001.SZ", "股票名称", "20250718")
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
    failed_stocks = []  # 记录失败的股票

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
                    failed_stocks.append(
                        {
                            "stock_code": stock_symbol,
                            "stock_name": stock_name,
                            "error": "生成因子失败",
                        }
                    )
                    print(
                        f"进度: {i}/{len(stock_list)} - 失败: {stock_symbol} - 错误: 生成因子失败"
                    )

            except Exception as e:
                failed_count += 1
                failed_stocks.append(
                    {
                        "stock_code": stock_symbol,
                        "stock_name": stock_name,
                        "error": str(e),
                    }
                )
                print(
                    f"进度: {i}/{len(stock_list)} - 失败: {stock_symbol} - 错误: {str(e)}"
                )

    print(f"\n处理完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {failed_count} 只")
    print(f"总计: {len(stock_list)} 只")

    # 保存失败股票列表
    if failed_stocks:
        save_failed_stocks(failed_stocks, "batch_process_parallel")


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
    failed_stocks = []  # 记录失败的股票

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
            failed_stocks.append(
                {
                    "stock_code": stock_info["converted_code"],
                    "stock_name": stock_info["stock_name"],
                    "error": "生成因子失败",
                }
            )
            print(f"处理失败: {stock_info['converted_code']}")
            error_count += 1

        # 添加延时避免API限制

    print(f"\n处理完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {error_count} 只")
    print(f"总计: {len(stock_list)} 只")

    # 保存失败股票列表
    if failed_stocks:
        save_failed_stocks(failed_stocks, "batch_process_single")


def save_failed_stocks(failed_stocks, process_type):
    """
    保存失败股票信息到文件

    Args:
        failed_stocks (list): 失败股票列表
        process_type (str): 处理类型 (batch_process_parallel, batch_process_single, retry_failed)
    """
    # 创建数据目录
    data_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    data_dir.mkdir(exist_ok=True)

    # 生成文件名，包含时间戳和处理类型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_{process_type}_{timestamp}.txt"
    file_path = data_dir / filename

    # 写入失败股票信息
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"失败股票记录 - {process_type}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"失败总数: {len(failed_stocks)}\n")
        f.write("=" * 60 + "\n\n")

        for i, stock in enumerate(failed_stocks, 1):
            f.write(f"{i:3d}. 股票代码: {stock['stock_code']}\n")
            f.write(f"     股票名称: {stock['stock_name']}\n")
            f.write(f"     失败原因: {stock['error']}\n")
            f.write("-" * 40 + "\n")

    print(f"失败股票记录已保存到: {file_path}")


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
    retry_failed_stocks = []  # 记录重试失败的股票

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
                        retry_failed_stocks.append(
                            {
                                "stock_code": stock_symbol,
                                "stock_name": stock_name,
                                "error": "生成因子失败",
                            }
                        )
                        print(
                            f"重试进度: {i}/{len(failed_stocks)} - 失败: {stock_symbol} - 错误: 生成因子失败"
                        )

                except Exception as e:
                    failed_count += 1
                    retry_failed_stocks.append(
                        {
                            "stock_code": stock_symbol,
                            "stock_name": stock_name,
                            "error": str(e),
                        }
                    )
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
                    retry_failed_stocks.append(
                        {
                            "stock_code": stock_info["converted_code"],
                            "stock_name": stock_info["stock_name"],
                            "error": "生成因子失败",
                        }
                    )
                    print(f"失败: {stock_info['converted_code']}")
                    failed_count += 1

                # 串行模式下添加延迟避免API限制
                time.sleep(1)

            except Exception as e:
                retry_failed_stocks.append(
                    {
                        "stock_code": stock_info["converted_code"],
                        "stock_name": stock_info["stock_name"],
                        "error": str(e),
                    }
                )
                print(f"失败: {stock_info['converted_code']} - 错误: {str(e)}")
                failed_count += 1

    print(f"\n重试完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {failed_count} 只")
    print(f"重试总计: {len(failed_stocks)} 只")

    # 保存重试失败的股票列表
    if retry_failed_stocks:
        save_failed_stocks(retry_failed_stocks, "retry_failed")


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
if __name__ == "__main__":
    end_date = processing_config["default_end_date"]

    # 从配置文件获取路径
    csv_folder_path = RAW_DATA_DIR

    # 生成带时间戳的输出路径
    output_folder_path = get_timestamped_output_path(ENHANCED_DATA_DIR)
    timestamp = os.path.basename(output_folder_path).split("_")[-1]

    print(f"输入路径: {csv_folder_path}")
    print(f"输出路径: {output_folder_path}")
    print(f"数据下载结束日期: {end_date}")
    print(f"输出文件时间戳: {timestamp}")

    # 从配置文件获取测试模式
    test_mode = processing_config["test_mode"]  # "single", "batch", 或 "retry_failed"

    if test_mode == "single":
        # 测试单只股票
        stock_code = "000002.XSHE"
        test_single_stock(stock_code, output_folder_path, end_date)

    elif test_mode == "batch":
        # 选择并行或串行处理
        use_parallel = True  # 设置为False使用串行处理
        limit = None  # 处理所有股票

        if use_parallel:
            print(f"并行批量测试所有股票")
            batch_process_stocks_parallel(
                csv_folder_path,
                output_folder_path,
                end_date,
                limit,
                max_workers=processing_config["max_workers"],
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
