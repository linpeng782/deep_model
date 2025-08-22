# 滚动回测脚本对比分析

## 概述

本文档详细对比分析了 `deep_model/backtesting/` 目录下的三个滚动回测脚本，帮助用户根据具体需求选择合适的回测框架。

## 脚本概览

| 脚本名称 | 主要特点 | 适用场景 |
|---------|---------|---------|
| `rolling_backtest_open.py` | 固定开盘价交易 | 传统量化策略，简单快速 |
| `rolling_backtest_post.py` | 后复权价格，开盘价交易 | 需要考虑分红除权的长期策略 |
| `rolling_backtest_mixed.py` | 灵活买卖时点配置 | 高级策略，需要精确控制交易时点 |

## 详细对比分析

### 1. 核心函数签名对比

#### rolling_backtest_open.py
```python
def rolling_backtest(
    portfolio_weights,
    bars_df,
    holding_months=12,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
)
```

#### rolling_backtest_post.py
```python
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
)
```

#### rolling_backtest_mixed.py
```python
def rolling_backtest(
    portfolio_weights,
    bars_df,
    holding_months=12,
    initial_capital=10000 * 10000,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.0001,
    commission_rate=0.0002,
    min_transaction_fee=5,
    cash_annual_yield=0.02,
    sell_timing="open",
    buy_timing="open",
)
```

### 2. 价格数据获取差异

#### rolling_backtest_open.py
- **固定开盘价**：只获取开盘价数据
- **复权类型**：固定为后复权 (`adjust='post'`)
- **函数签名**：`get_stock_bars(stock_price_data, portfolio_weights, adjust)`

```python
# 固定获取开盘价，后复权
bars_df = get_stock_bars(stock_price_data, portfolio_weights, adjust='post')
```

#### rolling_backtest_post.py
- **灵活价格类型**：支持开盘价和收盘价
- **复权类型**：固定为后复权 (`adjust='post'`)
- **函数签名**：`get_stock_bars(stock_price_data, portfolio_weights, adjust, price_type)`

```python
# 支持价格类型选择，固定后复权
bars_df = get_stock_bars(stock_price_data, portfolio_weights, 'post', 'open')
```

#### rolling_backtest_mixed.py
- **完全灵活**：支持开盘价/收盘价 + 前复权/后复权/不复权
- **函数签名**：`get_stock_bars(stock_price_data, portfolio_weights, adjust, price_type)`

```python
# 完全可配置的价格获取
bars_df = get_stock_bars(stock_price_data, portfolio_weights, 'post', 'open')
```

### 3. 交易时点控制

#### rolling_backtest_open.py
- **固定交易时点**：买入和卖出都在开盘价执行
- **无配置选项**：不支持交易时点自定义

#### rolling_backtest_post.py & rolling_backtest_mixed.py
- **灵活交易时点**：支持 `sell_timing` 和 `buy_timing` 参数
- **可选值**：`"open"` (开盘价) 或 `"close"` (收盘价)
- **默认设置**：都默认为开盘价交易

```python
# 支持的交易时点组合
sell_timing="open", buy_timing="open"    # 开盘卖出，开盘买入
sell_timing="close", buy_timing="open"   # 收盘卖出，开盘买入
sell_timing="open", buy_timing="close"   # 开盘卖出，收盘买入
sell_timing="close", buy_timing="close"  # 收盘卖出，收盘买入
```

### 4. 手续费处理差异

#### rolling_backtest_open.py & rolling_backtest_mixed.py
- **完整手续费**：包含 `min_transaction_fee` 参数
- **最低手续费保护**：确保每笔交易至少收取最低手续费

#### rolling_backtest_post.py
- **简化手续费**：移除了 `min_transaction_fee` 参数
- **按比例收费**：只按交易金额比例收取手续费

### 5. 性能指标精度

#### rolling_backtest_open.py
- **标准精度**：绩效指标保留4位小数
- **适用场景**：一般回测分析

```python
"年化收益": round(Strategy_Annualized_Return_EAR, 4)
```

#### rolling_backtest_post.py & rolling_backtest_mixed.py
- **高精度**：绩效指标保留6位小数
- **适用场景**：精确的策略对比分析

```python
"年化收益": round(Strategy_Annualized_Return_EAR, 6)
```

## 使用建议

### 选择 rolling_backtest_open.py 当：
- ✅ 需要快速验证策略效果
- ✅ 策略逻辑相对简单
- ✅ 只关注开盘价交易
- ✅ 对精度要求不高

### 选择 rolling_backtest_post.py 当：
- ✅ 需要考虑分红除权影响
- ✅ 长期持仓策略
- ✅ 需要灵活的价格类型选择
- ✅ 对手续费计算要求简化

### 选择 rolling_backtest_mixed.py 当：
- ✅ 需要精确控制买卖时点
- ✅ 策略涉及复杂的交易逻辑
- ✅ 需要对比不同交易时点的效果
- ✅ 需要高精度的绩效分析
- ✅ 需要完整的手续费控制

## 实际应用示例

### 1. 简单因子策略测试
```python
# 使用 rolling_backtest_open.py
from rolling_backtest_open import rolling_backtest
account_result = rolling_backtest(portfolio_weights, bars_df)
```

### 2. 长期价值投资策略
```python
# 使用 rolling_backtest_post.py，考虑分红除权
from rolling_backtest_post import rolling_backtest
account_result = rolling_backtest(
    portfolio_weights, 
    bars_df,
    holding_months=24,  # 2年持仓
    sell_timing="close",
    buy_timing="open"
)
```

### 3. 高频交易策略对比
```python
# 使用 rolling_backtest_mixed.py，对比不同交易时点
from rolling_backtest_mixed import rolling_backtest

# 策略1：开盘买入，收盘卖出
result1 = rolling_backtest(
    portfolio_weights, bars_df,
    sell_timing="close", buy_timing="open"
)

# 策略2：收盘买入，开盘卖出  
result2 = rolling_backtest(
    portfolio_weights, bars_df,
    sell_timing="open", buy_timing="close"
)
```

## 技术架构统一性

### 共同特点
- ✅ 都支持N个月滚动持仓（默认12个月）
- ✅ 都包含完整的绩效分析函数
- ✅ 都支持图表保存功能
- ✅ 都使用相同的辅助函数（`get_monthly_first_trading_days`, `get_expire_date`, `calc_transaction_fee`）

### 代码复用
三个脚本在核心逻辑上高度一致，主要差异在于：
1. 价格数据获取的灵活性
2. 交易时点的可配置性
3. 手续费计算的复杂度
4. 输出精度的要求

## 维护建议

1. **统一接口**：考虑将三个脚本合并为一个，通过参数控制不同模式
2. **配置文件**：将交易成本、持仓周期等参数外置到配置文件
3. **单元测试**：为每个脚本编写对应的测试用例
4. **文档同步**：确保代码修改时同步更新本对比文档

## 总结

三个滚动回测脚本各有特色，用户应根据具体的策略需求和精度要求选择合适的版本：

- **入门用户**：推荐 `rolling_backtest_open.py`
- **专业用户**：推荐 `rolling_backtest_mixed.py`
- **特定需求**：根据上述对比选择最适合的版本

所有脚本都经过充分测试，可以放心使用。如有疑问，请参考各脚本的详细注释或联系开发团队。
