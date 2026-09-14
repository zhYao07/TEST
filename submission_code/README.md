# 浅层超稠油注采响应滞后预测

## 算法说明

算法先将注汽井和生产井记录聚合为 `WELL_GROUP_NAME × PROD_DATE` 日级底表。每个原始指标保留 sum、mean、max、min、std 和有效井数。随后对 8 个核心序列计算截至前一天的 7、14、30、60、90 天 mean、std、min、max。模型还按 `WELL_GROUP_NAME × PROD_INDICATOR` 使用测试起点前的训练标签构造冻结历史画像；整个测试期不使用也不更新测试标签，只有距离最后已知标签的天数随预测日期变化。

预测使用三个 CatBoost 模型：短/长分段分类器、0～3 天四分类器和 4～45 天回归器。当前提交策略使用 0.56 分段阈值；Short 分支取四分类条件中位数，Long 分支输出安全值 4 天，以控制真实短段误入 Long 后的评分损失。

## 运行环境

- Python 3.9.17
- numpy 1.26.3
- pandas 2.3.3
- catboost 1.2.8
- PyYAML 6.0.2

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据目录

`--data_dir` 指向赛事下发的原始 `data` 目录，结构如下：

```text
data/
├── train/
│   ├── well_group_info.csv
│   ├── well_inj_data.csv
│   ├── well_prod_data.csv
│   └── optimal_lag_days.csv
└── test/
    ├── well_inj_data.csv
    ├── well_prod_data.csv
    └── test_optimal_lag_days.csv
```

训练期动态数据用于为测试期初构造历史窗口；训练标签仅用于生成冻结在测试起点之前的历史画像。

## 运行命令

在解压后的代码包根目录执行：

```bash
python predict.py --data_dir ../data --output ./test_optimal_lag_days.csv
```

也可以指定配置文件：

```bash
python predict.py --data_dir ../data --output ./test_optimal_lag_days.csv --config ./config.yaml
```

运行后会一次性完成数据读取、井组日聚合、历史特征计算、模型加载、预测和 CSV 生成，无需修改代码或手工写入数据。程序只写入 `--output` 指定的最终提交 CSV，不保存特征底表、预测明细、特征重要性、日志或其他中间文件。

## 文件说明

```text
submission_code/
├── README.md              # 算法说明、环境和运行方法
├── predict.py             # 主预测入口
├── requirements.txt       # Python 依赖
├── config.yaml            # 模型参数和特征配置
├── src/
│   ├── feature.py         # group-day 聚合与历史特征
│   ├── model.py           # 模型加载与推理
│   └── utils.py           # 配置、输入和结果校验
├── models/
│   ├── segment.cbm        # 短/长分段分类器
│   ├── short.cbm          # 0～3 天四分类器
│   └── long.cbm           # 长段回归器，保留供诊断
└── examples/
    └── result_example.json
```

输出 CSV 固定包含以下五列，字段名、样本数和行顺序与测试模板一致：

```text
WELL_GROUP_NAME, PROD_DATE, INJ_INDICATOR, PROD_INDICATOR, OPTIMAL_LAG_DAYS
```
