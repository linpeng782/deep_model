# Deep Model 量化回测系统

## 项目简介

这是一个完整的量化回测系统，支持多种买卖时点策略对比，包含数据准备、信号生成、回测执行和结果分析四个核心模块。系统采用滚动持仓策略，支持买卖价格分离，提供多场景对比分析功能。

## 项目结构

```
deep_model/
├── config/                 # 配置管理
│   └── paths.py            # 路径配置
├── tools/                  # 数据处理工具
│   ├── batch_factor_processor.py    # 批量因子处理器
│   ├── feature_comparison.py        # 特征对比工具
│   └── compare_csv_folders.py       # CSV文件夹对比
├── backtest/              # 回测框架
│   ├── rolling_backtest_mixed.py    # 混合回测策略
│   ├── rolling_backtest_open.py     # 开盘价回测
│   ├── rolling_backtest_post.py     # 后复权回测
│   ├── get_stock_data.py            # 股票数据获取
│   └── get_buy_signal.py            # 买入信号生成
├── docs/                  # 文档
│   ├── 回测系统运行流程说明.md
│   └── batch_factor_processor_logic.md
└── data/                  # 数据目录（被gitignore）
    ├── raw/               # 原始数据
    ├── enhanced/          # 增强因子数据
    └── cache/             # 缓存数据
```

## 核心功能

### 1. 数据处理
- 支持5733只股票的批量因子处理
- 自动股票代码转换（SZ↔XSHE, SH↔XSHG, BJ↔BJSE）
- 特征数据一致性验证

### 2. 回测框架
- 12个月滚动持仓策略
- 多种买卖时点对比（开盘/收盘）
- 复权/未复权价格支持
- 完整的交易成本计算

### 3. 性能分析
- 详细的回测指标计算
- 可视化图表生成
- 基准对比分析

## 快速开始

### 环境要求
- Python 3.8+
- pandas, numpy, matplotlib
- rqdatac (米筐数据接口)

### 安装依赖
```bash
pip install pandas numpy matplotlib scipy statsmodels tqdm seaborn joblib
```

### 基本使用

1. **配置路径**
```python
from config.paths import ensure_dirs
ensure_dirs()  # 确保所有必要目录存在
```

2. **批量处理因子数据**
```python
from tools.batch_factor_processor import batch_process_stocks
batch_process_stocks(csv_folder_path, output_folder_path, end_date)
```

3. **运行回测**
```python
from backtest.rolling_backtest_mixed import rolling_backtest
results = rolling_backtest(portfolio_weights, bars_df)
```

## 重要说明

⚠️ **数据目录说明**：
- `data/` 目录包含大量股票数据（约17GB），已被`.gitignore`排除
- 首次使用需要准备相应的数据文件
- 数据格式要求详见文档说明

## 开发指南

### 代码规范
- 所有注释和函数说明使用中文
- 优先使用向量化操作，避免循环
- 代码中不使用图形符号和表情

### 测试
```bash
# 单只股票测试
python tools/batch_factor_processor.py

# 回测测试
python backtest/rolling_backtest_mixed.py
```

## 版本历史

- v1.0.0 - 初始版本，完成基础回测框架
- v1.1.0 - 添加批量因子处理功能
- v1.2.0 - 完善配置管理和文档

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
