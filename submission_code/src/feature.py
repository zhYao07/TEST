from pathlib import Path

import pandas as pd


DAY_KEYS = ['WELL_GROUP_NAME', 'PROD_DATE']
SAMPLE_KEYS = DAY_KEYS + ['INJ_INDICATOR', 'PROD_INDICATOR']
TARGET = 'OPTIMAL_LAG_DAYS'
METRICS = {
    'inj': ['INJ_VOL_DAILY', 'INJ_LIQ_DAILY', 'WH_TEMP', 'STEAM_INJ_PRES'],
    'prod': ['LIQ_PROD_DAILY', 'OIL_PROD_DAILY', 'WATER_CUT', 'WH_TEMP', 'MAX_TUBING_PRES'],
}


def read_data(path):
    data = pd.read_csv(path)
    data['PROD_DATE'] = pd.to_datetime(data['PROD_DATE'], format='mixed').dt.normalize()
    return data


def aggregate_group_day(data, source):
    """将井级动态聚合为唯一的井组日记录。"""
    grouped = data.groupby(DAY_KEYS)
    feature = grouped['WELL_NAME'].nunique().rename(
        f'{source}__recorded_well_count').to_frame()
    for column in METRICS[source]:
        values = grouped[column]
        prefix = f'{source}__{column}__'
        feature[prefix + 'sum'] = values.sum(min_count=1)
        feature[prefix + 'mean'] = values.mean()
        feature[prefix + 'max'] = values.max()
        feature[prefix + 'min'] = values.min()
        feature[prefix + 'std'] = values.std()
        feature[prefix + 'valid_well_count'] = values.count()
    return feature.reset_index()


def build_group_day(inj_data, prod_data, well_info, test_data):
    groups = sorted(set(well_info['WELL_GROUP_NAME']).union(
        set(inj_data['WELL_GROUP_NAME']),
        set(prod_data['WELL_GROUP_NAME']),
        set(test_data['WELL_GROUP_NAME'])))
    start_date = min(inj_data.PROD_DATE.min(), prod_data.PROD_DATE.min(),
                     test_data.PROD_DATE.min())
    end_date = max(inj_data.PROD_DATE.max(), prod_data.PROD_DATE.max(),
                   test_data.PROD_DATE.max())
    dates = pd.date_range(start_date, end_date)
    group_day = pd.MultiIndex.from_product(
        [groups, dates], names=DAY_KEYS).to_frame(index=False)

    for source, data in [('inj', inj_data), ('prod', prod_data)]:
        feature = aggregate_group_day(data, source)
        group_day = pd.merge(group_day, feature, on=DAY_KEYS, how='left')

    count_columns = [column for column in group_day if column.endswith('well_count')]
    group_day[count_columns] = group_day[count_columns].fillna(0).astype(int)
    return group_day


def add_history_feature(group_day, history_columns, history_windows):
    """计算截至前一天的多窗口统计特征。"""
    data = group_day.sort_values(DAY_KEYS).reset_index(drop=True).copy()
    history = data.groupby('WELL_GROUP_NAME')[history_columns].shift(1)
    features = [data]
    for window in history_windows:
        rolling = history.groupby(
            data['WELL_GROUP_NAME'], sort=False).rolling(window, min_periods=1)
        for stat in ['mean', 'std', 'min', 'max']:
            feature = getattr(rolling, stat)().reset_index(
                level=0, drop=True).sort_index()
            feature = feature.add_suffix(f'__hist{window}_{stat}')
            features.append(feature)
    return pd.concat(features, axis=1)


def build_test_dataset(data_dir, config):
    """从赛事原始 data 目录构造测试样本特征。"""
    data_dir = Path(data_dir)
    inj_data = pd.concat([
        read_data(data_dir / split / 'well_inj_data.csv')
        for split in ['train', 'test']
    ], ignore_index=True)
    prod_data = pd.concat([
        read_data(data_dir / split / 'well_prod_data.csv')
        for split in ['train', 'test']
    ], ignore_index=True)
    well_info = pd.read_csv(data_dir / 'train' / 'well_group_info.csv')
    template_raw = pd.read_csv(data_dir / 'test' / 'test_optimal_lag_days.csv')
    test_data = template_raw.copy()
    test_data['PROD_DATE'] = pd.to_datetime(
        test_data['PROD_DATE'], format='mixed').dt.normalize()
    test_data = test_data.drop(columns=TARGET)

    group_day = build_group_day(inj_data, prod_data, well_info, test_data)
    feature_data = add_history_feature(
        group_day,
        config['feature']['history_columns'],
        config['feature']['history_windows'],
    )
    test_data['_row_id'] = range(len(test_data))
    test = pd.merge(test_data, feature_data, on=DAY_KEYS, how='left')
    test = test.sort_values('_row_id').drop(columns='_row_id').reset_index(drop=True)
    return template_raw, test
