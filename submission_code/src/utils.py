from pathlib import Path

import pandas as pd
import yaml


OUTPUT_COLUMNS = [
    'WELL_GROUP_NAME',
    'PROD_DATE',
    'INJ_INDICATOR',
    'PROD_INDICATOR',
    'OPTIMAL_LAG_DAYS',
]


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def check_input_files(data_dir):
    data_dir = Path(data_dir)
    required = [
        data_dir / 'train' / 'well_group_info.csv',
        data_dir / 'train' / 'well_inj_data.csv',
        data_dir / 'train' / 'well_prod_data.csv',
        data_dir / 'train' / 'optimal_lag_days.csv',
        data_dir / 'test' / 'well_inj_data.csv',
        data_dir / 'test' / 'well_prod_data.csv',
        data_dir / 'test' / 'test_optimal_lag_days.csv',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('缺少输入文件：\n' + '\n'.join(missing))


def save_result(template, prediction, output):
    result = template.copy()
    result['OPTIMAL_LAG_DAYS'] = prediction
    if list(result.columns) != OUTPUT_COLUMNS:
        raise ValueError('结果字段或字段顺序与提交要求不一致')
    if result['OPTIMAL_LAG_DAYS'].isna().any():
        raise ValueError('预测结果包含缺失值')
    if not result['OPTIMAL_LAG_DAYS'].between(0, 45).all():
        raise ValueError('预测结果超出 0～45 天范围')
    if not (result['OPTIMAL_LAG_DAYS'] % 1 == 0).all():
        raise ValueError('预测结果必须为整数')

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding='utf-8-sig')
    return result
