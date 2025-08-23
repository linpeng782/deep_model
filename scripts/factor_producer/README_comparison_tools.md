# CSV文件夹比较工具使用说明

这些工具可以帮助您比较不同文件夹中的CSV文件，支持灵活的文件夹路径配置。

## 工具列表

### 1. find_all_differences.py - 找出所有差异文件
找出两个文件夹中所有有差异的CSV文件。

**使用方法：**
```bash
# 使用默认路径（enhanced vs enhanced_20250819）
python tools/find_all_differences.py

# 比较指定文件夹
python tools/find_all_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250818_02

# 比较不同日期的文件夹
python tools/find_all_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250819
```

### 2. check_specific_differences.py - 检查特定文件差异详情
详细分析特定文件的差异情况。

**使用方法：**
```bash
# 使用默认路径和默认文件列表
python tools/check_specific_differences.py

# 比较指定文件夹
python tools/check_specific_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250818_02

# 检查特定文件
python tools/check_specific_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250818_02 --files "000001.XSHE-平安银行-日线后复权及常用指标-20250818.csv" "000002.XSHE-万科A-日线后复权及常用指标-20250818.csv"
```

### 3. sample_compare_folders.py - 抽样比较文件夹
随机抽样比较两个文件夹的CSV文件，适合快速验证。

**使用方法：**
```bash
# 使用默认路径，抽样500个文件
python tools/sample_compare_folders.py

# 比较指定文件夹，抽样100个文件
python tools/sample_compare_folders.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250818_02 --sample-size 100

# 比较不同日期，抽样1000个文件
python tools/sample_compare_folders.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250819 --sample-size 1000
```

## 参数说明

### 通用参数
- `--folder1`: 第一个文件夹路径（相对于enhanced目录）
- `--folder2`: 第二个文件夹路径（相对于enhanced目录）

### 特定参数
- `--files`: （check_specific_differences.py）要检查的具体文件列表
- `--sample-size`: （sample_compare_folders.py）抽样数量，默认500

## 文件夹路径说明

工具会自动在enhanced目录下查找指定的文件夹。例如：
- `--folder1 enhanced_factors_csv_20250818_01` 对应路径：`/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250818_01`
- `--folder2 enhanced_factors_csv_20250819` 对应路径：`/Users/didi/KDCJ/deep_model/data/enhanced/enhanced_factors_csv_20250819`

## 使用场景示例

### 场景1：比较同一天的两次运行结果
```bash
python tools/find_all_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250818_02
```

### 场景2：比较不同日期的数据
```bash
python tools/sample_compare_folders.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250819 --sample-size 200
```

### 场景3：详细检查特定股票的差异
```bash
python tools/check_specific_differences.py --folder1 enhanced_factors_csv_20250818_01 --folder2 enhanced_factors_csv_20250819 --files "000001.XSHE-平安银行-日线后复权及常用指标-20250818.csv"
```

## 输出说明

- ✅ 完全一致：文件内容完全相同
- ⚠️ 数值差异：存在微小的数值差异（通常是正常的浮点精度差异）
- ❌ 形状不匹配：文件行数或列数不同
- ❌ 列名不匹配：列名不一致
- ❌ 错误：文件读取或处理出错
