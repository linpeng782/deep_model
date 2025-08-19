# batch_factor_processor.py 代码逻辑说明

## 概述

本模块用于批量处理股票因子数据，从原始CSV文件中提取股票信息，通过米筐API获取技术和基本面因子，生成增强版的因子数据文件。

## 核心功能

1. **文件名解析**: 从CSV文件名提取股票代码、名称和日期信息
2. **代码转换**: 将股票代码转换为米筐API格式
3. **因子生成**: 获取技术指标和基本面因子数据
4. **批量处理**: 支持串行和并行两种处理模式
5. **失败重试**: 自动识别和重试处理失败的股票

## 主要执行流程

```
main()
├── 模式选择 (single/batch/retry_failed)
├── batch_process_stocks_parallel() [并行模式]
│   ├── get_stock_list_from_csv_folder()
│   │   ├── parse_stock_info_from_filename()
│   │   └── convert_stock_code()
│   └── ThreadPoolExecutor
│       └── process_single_stock_parallel()
│           └── generate_factors_for_stock()
├── batch_process_stocks() [串行模式]
│   ├── get_stock_list_from_csv_folder()
│   └── generate_factors_for_stock() (循环调用)
└── retry_failed_stocks() [重试模式]
    ├── get_failed_stocks()
    └── generate_factors_for_stock() (重试失败股票)
```

---

## 详细函数功能说明

### 1. `parse_stock_info_from_filename(filename)`
**功能**: 解析CSV文件名，提取股票信息
- **输入格式**: 
  - `filename`: 字符串类型
  - 示例: `"000001.SZ-平安银行-日线后复权及常用指标-20250718.csv"`
- **输出格式**: 
  - 成功: 三元组 `(股票代码, 股票名称, 日期)`
  - 失败: `(None, None, None)`
  - 示例: `("000001.SZ", "平安银行", "20250718")`
- **处理**: 使用正则表达式 `r'([0-9]{6}\.[A-Z]{2})-(.+?)-日线后复权及常用指标-(\d{8})\.csv'`

### 2. `convert_stock_code(original_code)`
**功能**: 转换股票代码格式以适配米筐API
- **输入格式**:
  - `original_code`: 字符串类型
  - 示例: `"000001.SZ"`, `"600000.SH"`, `"430001.BJ"`
- **输出格式**:
  - 深交所: `"XXXXXX.XSHE"`
  - 上交所: `"XXXXXX.XSHG"`
  - 北交所: `"XXXXXX.BJSE"`
  - 其他: `None`
- **转换规则**:
  - `000001.SZ` → `000001.XSHE` (深交所)
  - `600000.SH` → `600000.XSHG` (上交所)
  - `430001.BJ` → `430001.BJSE` (北交所)

### 3. `get_stock_list_from_csv_folder(csv_folder_path, limit=None)`
**功能**: 扫描CSV文件夹，生成待处理股票列表
- **输入格式**:
  - `csv_folder_path`: 字符串，CSV文件夹的绝对路径
  - `limit`: 整数或None，限制处理的股票数量
  - 示例: `("/path/to/csv/folder", 10)`
- **输出格式**:
  - 列表，每个元素为字典
  - 字典结构:
    ```python
    {
        'original_code': '000001.SZ',
        'converted_code': '000001.XSHE', 
        'stock_name': '平安银行',
        'date': '20250718'
    }
    ```
- **处理流程**:
  1. 遍历文件夹中所有`.csv`文件
  2. 调用`parse_stock_info_from_filename()`解析文件名
  3. 调用`convert_stock_code()`转换代码格式
  4. 过滤掉转换失败的股票和解析失败的文件
  5. 应用数量限制（如果指定）

### 4. `generate_factors_for_stock(stock_symbol, end_date)`
**功能**: 为单只股票生成完整的因子数据
- **输入格式**:
  - `stock_symbol`: 字符串，米筐格式股票代码 (如 `"000001.XSHE"`)
  - `end_date`: 字符串，结束日期 (如 `"20250718"`)
- **输出格式**:
  - 成功: pandas.DataFrame，索引为日期，列为各种因子
  - 失败: `None`
  - DataFrame列包括:
    ```
    技术因子: open, high, low, close, volume, total_turnover, 
             prev_close, change_amount, change_pct, amplitude, turnover_rate
    基本面因子: pe_ratio_lyr, pe_ratio_ttm, pb_ratio_lyr, pb_ratio_ttm, 
               pb_ratio_lf, ps_ratio_lyr, ps_ratio_ttm, dividend_yield_ttm,
               market_cap_3, market_cap_2
    ```
- **数据获取**:
  1. 获取股票上市日期作为开始日期
  2. 调用`get_price()`获取OHLCV技术数据
  3. 调用`get_turnover_rate()`获取换手率
  4. 调用`get_factor()`获取基本面因子
- **因子计算**:
  1. `prev_close`: 前一日收盘价 = `close.shift(1)`
  2. `change_amount`: 涨跌额 = `close - prev_close`
  3. `change_pct`: 涨跌幅 = `(close - prev_close) / prev_close`
  4. `amplitude`: 振幅 = `(high - low) / prev_close`

### 5. `process_single_stock_parallel(stock_info, output_folder_path, end_date)`
**功能**: 并行处理单只股票的辅助函数
- **输入格式**:
  - `stock_info`: 字典，包含股票信息
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
- **输出格式**:
  - 成功: `(True, stock_info)`
  - 失败: `(False, stock_info)`
- **功能**: 为ThreadPoolExecutor设计的单股票处理函数

### 6. `batch_process_stocks_parallel(csv_folder_path, output_folder_path, end_date, limit=None, max_workers=4)`
**功能**: 并行批量处理股票因子数据
- **输入格式**:
  - `csv_folder_path`: 字符串，原始CSV文件夹路径
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
  - `limit`: 整数或None，处理股票数量限制
  - `max_workers`: 整数，最大并发线程数（默认4）
- **特点**:
  - 使用ThreadPoolExecutor实现并行处理
  - 显示实时进度和预估剩余时间
  - 自动统计成功/失败数量

### 7. `batch_process_stocks(csv_folder_path, output_folder_path, end_date, limit=None)`
**功能**: 串行批量处理股票因子数据
- **输入格式**:
  - `csv_folder_path`: 字符串，原始CSV文件夹路径
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
  - `limit`: 整数或None，处理股票数量限制
- **特点**:
  - 逐个处理，稳定性更好
  - 每个股票间有延时，避免API限制
  - 适合小规模测试

### 8. `get_failed_stocks(csv_folder_path, output_folder_path, end_date)`
**功能**: 获取尚未创建输出CSV文件的股票列表
- **输入格式**:
  - `csv_folder_path`: 字符串，原始CSV文件夹路径
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
- **输出格式**:
  - 列表，包含失败股票的信息字典
- **功能**: 对比原始文件和输出文件，找出未成功处理的股票

### 9. `retry_failed_stocks(csv_folder_path, output_folder_path, end_date, use_parallel=False, max_workers=2)`
**功能**: 重试处理失败的股票
- **输入格式**:
  - `csv_folder_path`: 字符串，原始CSV文件夹路径
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
  - `use_parallel`: 布尔值，是否使用并行模式
  - `max_workers`: 整数，最大并发线程数
- **特点**:
  - 自动识别失败股票
  - 支持并行和串行两种重试模式
  - 串行模式成功率更高

### 10. `test_single_stock(stock_symbol, output_folder_path, end_date="20250718")`
**功能**: 测试单只股票的处理功能
- **输入格式**:
  - `stock_symbol`: 字符串，米筐格式股票代码
  - `output_folder_path`: 字符串，输出文件夹路径
  - `end_date`: 字符串，结束日期
- **输出格式**:
  - 成功: `True`
  - 失败: `False`
- **功能**: 用于开发和调试阶段的单股票测试

---

## 数据流向图

```
原始CSV文件夹
    ↓ (扫描文件名)
股票代码列表
    ↓ (代码转换)
米筐格式代码
    ↓ (米筐API调用)
原始数据 (OHLCV + 基本面 + 换手率)
    ↓ (因子计算和合并)
增强因子数据 (30+个因子)
    ↓ (中文列名映射)
新的CSV文件 (UTF-8编码)
```

---

## 主要执行模式

### 1. 单股票测试模式 (`test_mode = "single"`)
- **用途**: 开发和调试阶段
- **特点**: 快速验证单个股票的处理逻辑
- **适用场景**: 新功能测试、错误排查

### 2. 批量处理模式 (`test_mode = "batch"`)
- **串行模式**: 稳定性高，适合小规模数据
- **并行模式**: 效率高，适合大规模数据处理
- **可配置**: 线程数、处理数量限制

### 3. 失败重试模式 (`test_mode = "retry_failed"`)
- **用途**: 处理网络或API限制导致的失败
- **智能识别**: 自动对比原始和输出文件
- **灵活模式**: 支持串行和并行重试

---

## 关键设计特点

### 错误处理机制
- **全面包装**: 每个股票处理都有try-catch包装
- **继续执行**: 单个失败不影响整体进度
- **详细日志**: 记录成功/失败原因和统计信息
- **智能重试**: 自动识别和重试失败项目

### 性能优化
- **API限制处理**: 串行模式添加延时，并行模式控制并发数
- **内存管理**: 逐股票处理，避免大量数据同时加载
- **进度显示**: 实时显示处理进度和预估时间
- **灵活配置**: 支持数量限制和线程数调整

### 数据质量保证
- **列名映射**: 使用`factor_name_mapping`统一中文列名
- **数据清洗**: 过滤换手率为0的异常数据
- **编码统一**: 使用UTF-8编码保证中文兼容性
- **格式一致**: 保持与原始文件相同的命名格式

### 可维护性
- **模块化设计**: 函数职责单一，易于测试和维护
- **参数化配置**: 路径、日期、数量等关键参数可配置
- **清晰文档**: 详细的函数说明和使用示例
- **版本控制**: 支持不同版本的因子数据格式

---

## 因子数据说明

### 技术因子
- **价格数据**: 开盘价、最高价、最低价、收盘价、昨收价
- **成交数据**: 成交量、成交额、未复权成交量
- **技术指标**: 涨跌额、涨跌幅、振幅
- **换手率**: 总股本换手率、自由流通股换手率

### 基本面因子
- **估值指标**: PE、PB、PS等不同版本
- **市值数据**: 总市值、流通市值
- **股本数据**: 自由流通股本
- **收益指标**: 股息率TTM

### 数据覆盖
- **时间范围**: 从股票上市日到指定结束日期
- **市场覆盖**: 支持深交所、上交所、北交所
- **数据质量**: 后复权价格，处理停牌和异常值
