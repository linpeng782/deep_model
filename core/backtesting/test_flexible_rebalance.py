"""
测试灵活调仓频率的滚动回测框架
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('/Users/didi/KDCJ/deep_model/core/backtesting')

from rolling_backtest_refactor import rolling_backtest, get_performance_analysis


def get_portfolio_weights_path(end_date, rank_n):
    """获取投资组合权重文件路径"""
    return f"/Users/didi/KDCJ/deep_model/outputs/factor_weights/portfolio_weights_{end_date}_{rank_n}.pkl"


def get_bars_data_path():
    """获取股票价格数据文件路径"""
    return "/Users/didi/KDCJ/deep_model/data/stock_bars_20170101_20250819.pkl"


def test_different_frequencies():
    """测试不同调仓频率的效果"""
    
    # 加载数据
    end_date = "20250819"
    rank_n = "top50"
    
    portfolio_weights_path = get_portfolio_weights_path(end_date, rank_n)
    bars_data_path = get_bars_data_path()
    
    print("加载投资组合权重数据...")
    portfolio_weights = pd.read_pickle(portfolio_weights_path)
    
    print("加载股票价格数据...")
    bars_df = pd.read_pickle(bars_data_path)
    
    # 为了快速测试，只使用部分数据
    portfolio_weights = portfolio_weights.iloc[:100]  # 只用前100个交易日
    
    print(f"数据范围: {portfolio_weights.index.min()} 到 {portfolio_weights.index.max()}")
    print(f"总交易日数: {len(portfolio_weights)}")
    
    # 测试不同的调仓频率配置
    test_configs = [
        {"portfolio_count": 12, "rebalance_frequency": "monthly", "name": "月度调仓_12组合"},
        {"portfolio_count": 5, "rebalance_frequency": "weekly", "name": "周度调仓_5组合"},
        {"portfolio_count": 10, "rebalance_frequency": 10, "name": "10天调仓_10组合"},
        {"portfolio_count": 20, "rebalance_frequency": 5, "name": "5天调仓_20组合"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['name']}")
        print(f"组合数量: {config['portfolio_count']}")
        print(f"调仓频率: {config['rebalance_frequency']}")
        print(f"{'='*50}")
        
        try:
            # 运行回测
            account_history = rolling_backtest(
                portfolio_weights=portfolio_weights,
                bars_df=bars_df,
                portfolio_count=config['portfolio_count'],
                rebalance_frequency=config['rebalance_frequency'],
                initial_capital=10000000,  # 1000万初始资金
            )
            
            print(f"回测完成！账户历史记录长度: {len(account_history)}")
            print(f"最终资产: {account_history['total_account_asset'].iloc[-1]:,.2f}")
            print(f"总收益率: {(account_history['total_account_asset'].iloc[-1] / 10000000 - 1)*100:.2f}%")
            
            results[config['name']] = account_history
            
        except Exception as e:
            print(f"配置 {config['name']} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 比较结果
    if results:
        print(f"\n{'='*60}")
        print("各配置收益对比:")
        print(f"{'='*60}")
        
        for name, history in results.items():
            final_value = history['total_account_asset'].iloc[-1]
            total_return = (final_value / 10000000 - 1) * 100
            print(f"{name:20s}: 最终资产 {final_value:>12,.2f}, 总收益率 {total_return:>6.2f}%")


if __name__ == "__main__":
    test_different_frequencies()
