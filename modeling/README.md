# 数据集与 CatBoost 分层模型

代码分成两个独立步骤，使用已有 conda NLP 环境。路径相对于脚本位置确定，从项目根目录或 modeling 目录运行都可以。

```powershell
& 'D:\anaconda\envs\NLP\python.exe' 'modeling\get_dataset.py'
& 'D:\anaconda\envs\NLP\python.exe' 'modeling\model.py'
& 'D:\anaconda\envs\NLP\python.exe' 'modeling\model2.py'
```

## get_dataset.py

函数依次处理：`prepare` → `get_group_day` → `get_feature` → `get_dataset`。

- `get_aggregate_feature`：每个原始指标计算 sum、mean、max、min、std、有效井数，各侧另有记录井数。
- `get_group_day`：构造完整井组日历，保留共享井的井组关系。全缺失 sum 为 NaN；无记录日计数为 0、数值为 NaN；std 的 ddof=1；不填充历史值。
- `get_history_feature`：仅对 `HISTORY_COLUMNS` 中的 8 个核心序列增加过去 7、14、30、60、90 个自然日的 mean、std、min、max，共 160 列。先按井组 shift(1)，各窗口均截至 t-1，不含当天；缺失值不填充，均值/极值至少需要一个有效观测，std 至少需要两个有效观测（ddof=1）。训练、验证、测试的历史连续计算，不使用标签。
- `get_past_only_label_prior`：训练样本按 `井组×采出指标` 使用严格早于样本日的历史标签，构造 count、last、均值、中位数、short 比例和最近 5 条统计。
- `get_frozen_label_prior`：验证集统一冻结在 2025-12-01 之前，测试集统一冻结在 2026-03-01 之前；冻结画像不随未来标签更新，只有 `label_days_since_last` 随样本日期变化。
- `get_feature`：后续添加特征的统一入口，目前仅调用多窗口历史统计，保留全部当日聚合值。
- `get_dataset`：连接目标样本，保留行顺序，标签放在最后。日期、注入指标只用于样本标识，不进入本版模型。

数据保存在 `modeling/dataset/`：

| 文件 | 内容 |
|---|---|
| group_day.csv.gz | 多视角 group-day 底表，pandas 可直接读取 |
| train.csv | 2025-12-01 以前的有标签样本 |
| validate.csv | 2025-12-01 至 2026-02-28 的有标签样本 |
| test.csv | 正式测试样本，不包含标签列 |
| test_template.csv | 原始测试模板，用于保留提交键值和顺序 |
| dataset_info.json | 数据形状、验证区间和底表特征列 |

## model.py

`PARAMS` 集中设置参数，当前配置：深度 7、学习率 0.02、最多 1000 轮、随机种子 42。`STAGE_PARAMS` 定义三层损失。修改参数后只需重跑模型脚本，不必重新构造数据集。

- `model_catboost(train, validate, test)`：返回预测和三个模型。
- `predict_catboost(models, test)`：推理。分段阈值 0.5，短段四分类取条件中位数，长段 MAE 回归裁剪至 4～45 后用 `np.rint` 输出整数。
- `save_models` / `load_models`：保存和加载三阶段模型；推理使用模型内保存的特征顺序。
- `contest_score`：分段准确率、真实短段和长段 MAE、三个得分及综合分。错误路由的样本也计入真实分段 MAE。

运行时先用训练集拟合，在十二月至二月整段验证集上早停；再合并 train 和 validate，用已选定树数重训并预测 test。验证参与早停，成绩属于开发验证成绩。全量重训不再把已加入训练的 validate 当成独立验证集。

当前 label-prior v1 使用 0.56 路由阈值；进入 Long 分支时最终输出固定为 4 天，以降低真实短段误入 Long 后的损失。结果在 `modeling/outputs/catboost_dec_feb_safe_long4_label_prior_v1/`。每次运行保存最终提交文件 `test_optimal_lag_days.csv`、`final_models/` 三个模型权重、`stage_diagnostics.json` 和 `training_summary.json`。汇总同时记录阈值扫描前 10 名和验证集分月成绩；`stage_diagnostics.json` 分别记录 Router、Short Expert、Long Expert 的独立能力，不参与预测。

## model2.py

使用一个 `CatBoostRegressor(loss_function='MAE')` 直接预测全部 0～45 天样本，不做短长路由。基础参数、特征和验证区间与 `model.py` 一致；连续预测裁剪至 0～45 后使用 `np.rint` 转为整数。结果保存到 `modeling/outputs/catboost_single_regressor/`，每次运行只保存 `test_optimal_lag_days.csv`、`final_model.cbm` 和 `training_summary.json`。
