"""
批处理引擎模块
负责股票因子数据的批量处理，支持并行和串行模式
"""
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_utils import (
    get_stock_list_from_csv_folder,
    save_failed_stocks,
    get_failed_stocks,
)
from factor_calculator import generate_factors_for_stock
from config_manager import get_max_workers


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
