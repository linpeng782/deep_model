import sys
import os
from pathlib import Path

# 添加项目配置路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from deep_model.config.paths import RAW_DATA_DIR, ENHANCED_DATA_DIR

# 导入统一的factor_utils包
kdcj_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, kdcj_root)
try:
    from factor_utils import *

    print("成功导入factor_utils")
except ImportError as e:
    print(f"无法导入factor_utils: {e}")

# 导入新的模块化组件
from config_manager import (
    get_default_end_date,
    get_test_mode,
    get_max_workers,
    get_timestamped_output_path,
)
from factor_calculator import generate_factors_for_stock
from batch_processor import (
    batch_process_stocks_parallel,
    batch_process_stocks,
    retry_failed_stocks,
)


def test_single_stock(stock_symbol, output_folder_path, end_date="20250718"):
    """
    测试单只股票
    """
    print(f"测试单只股票: {stock_symbol}")
    factors_df = generate_factors_for_stock(stock_symbol, end_date)

    if factors_df is not None:
        # 创建输出文件夹
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
    # 获取配置信息（简化版）
    end_date = get_default_end_date()
    test_mode = get_test_mode()

    # 从配置文件获取路径
    csv_folder_path = RAW_DATA_DIR

    # 生成带时间戳的输出路径
    output_folder_path = get_timestamped_output_path(ENHANCED_DATA_DIR)
    timestamp = os.path.basename(output_folder_path).split("_")[-1]

    print(f"输入路径: {csv_folder_path}")
    print(f"输出路径: {output_folder_path}")
    print(f"数据下载结束日期: {end_date}")
    print(f"输出文件时间戳: {timestamp}")

    # 执行相应的测试模式

    if test_mode == "single":
        # 测试单只股票
        stock_code = "000002.XSHE"
        test_single_stock(stock_code, output_folder_path, end_date)

    elif test_mode == "batch":
        # 选择并行或串行处理
        use_parallel = True  # 设置为False使用串行处理
        limit = 20  # 处理所有股票

        if use_parallel:
            print(f"并行批量测试所有股票")
            batch_process_stocks_parallel(
                csv_folder_path,
                output_folder_path,
                end_date,
                limit,
                max_workers=get_max_workers(),
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
