import os
import json
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
FEATURE_VERSION = 'same_day_aggregates_hist7_14_30_60_90_selected8_label_prior_v1'
VALIDATE_START = '2025-12-01'
VALIDATE_END = '2026-03-01'  # 左闭右开，包含十二月、一月、二月
TEST_FREEZE_DATE = '2026-03-01'

DAY_KEYS = ['WELL_GROUP_NAME', 'PROD_DATE']
SAMPLE_KEYS = DAY_KEYS + ['INJ_INDICATOR', 'PROD_INDICATOR']
LABEL = 'OPTIMAL_LAG_DAYS'
LABEL_PRIOR_KEYS = ['WELL_GROUP_NAME', 'PROD_INDICATOR']
LABEL_PRIOR_COLUMNS = [
    'label_hist_count',
    'label_last_lag',
    'label_last_is_short',
    'label_hist_mean',
    'label_hist_median',
    'label_hist_short_ratio',
    'label_recent5_median',
    'label_recent5_short_ratio',
    'label_days_since_last',
]
METRICS = {
    'inj': ['INJ_VOL_DAILY', 'INJ_LIQ_DAILY', 'WH_TEMP', 'STEAM_INJ_PRES'],
    'prod': ['LIQ_PROD_DAILY', 'OIL_PROD_DAILY', 'WATER_CUT', 'WH_TEMP', 'MAX_TUBING_PRES'],
}
HISTORY_COLUMNS = [
    'inj__INJ_VOL_DAILY__sum',
    'inj__WH_TEMP__mean',
    'inj__STEAM_INJ_PRES__mean',
    'prod__LIQ_PROD_DAILY__sum',
    'prod__OIL_PROD_DAILY__sum',
    'prod__WATER_CUT__mean',
    'prod__WH_TEMP__mean',
    'prod__MAX_TUBING_PRES__max',
]
HISTORY_WINDOWS = [7, 14, 30, 60, 90]


def prepare(dataset):
    """统一日期格式，保留原始缺失值与真实零值。"""
    data = dataset.copy()
    data['PROD_DATE'] = pd.to_datetime(data['PROD_DATE'], format='mixed').dt.normalize()
    return data


def get_aggregate_feature(data, source):
    """按井组、日期聚合一个动态表，共享井在各所属井组分别参与。"""
    if data.duplicated(DAY_KEYS + ['WELL_NAME']).any():
        raise ValueError(f'{source} 存在重复井组-井-日期，请先核对原始记录')
    grouped = data.groupby(DAY_KEYS)
    feat = grouped['WELL_NAME'].nunique().rename(f'{source}__recorded_well_count').to_frame()
    for column in METRICS[source]:
        values = grouped[column]
        prefix = f'{source}__{column}__'
        feat[prefix + 'sum'] = values.sum(min_count=1)  # 全空时仍为空，而不是 0
        feat[prefix + 'mean'] = values.mean()
        feat[prefix + 'max'] = values.max()
        feat[prefix + 'min'] = values.min()
        feat[prefix + 'std'] = values.std()  # ddof=1，只有一口有效井时为空
        feat[prefix + 'valid_well_count'] = values.count()
    return feat.reset_index()


def get_group_day(inj_data, prod_data, well_info, label_data, test_data):
    """构造完整自然日历的 group-day 底表，不包含标签或跨日特征。"""
    sources = [inj_data, prod_data, label_data, test_data]
    groups = sorted(set(well_info['WELL_GROUP_NAME']).union(
        *(set(data['WELL_GROUP_NAME']) for data in sources)))
    start = min(data['PROD_DATE'].min() for data in sources)
    end = max(data['PROD_DATE'].max() for data in sources)
    dates = pd.date_range(start, end)
    group_day = pd.MultiIndex.from_product([groups, dates], names=DAY_KEYS).to_frame(index=False)
    for source, data in [('inj', inj_data), ('prod', prod_data)]:
        feat = get_aggregate_feature(data, source)
        group_day = pd.merge(group_day, feat, on=DAY_KEYS, how='left', validate='one_to_one')
    count_columns = [column for column in group_day if column.endswith('well_count')]
    group_day[count_columns] = group_day[count_columns].fillna(0).astype(int)
    return group_day


def get_history_feature(group_day, windows=HISTORY_WINDOWS):
    """对指定核心序列计算多个历史窗口的 mean/std/min/max，不包含当天。"""
    data = group_day.sort_values(DAY_KEYS).reset_index(drop=True).copy()
    # 底表已补齐自然日；先按井组 shift(1)，使 t 日窗口截至 t-1。
    history = data.groupby('WELL_GROUP_NAME')[HISTORY_COLUMNS].shift(1)
    features = [data]
    for window in windows:
        rolling = history.groupby(data['WELL_GROUP_NAME'], sort=False).rolling(window, min_periods=1)
        for stat in ['mean', 'std', 'min', 'max']:
            feat = getattr(rolling, stat)().reset_index(level=0, drop=True).sort_index()
            feat = feat.add_suffix(f'__hist{window}_{stat}')
            features.append(feat)
    # 忽略窗口内 NaN；无有效观测仍为 NaN，std 少于两个有效值时为 NaN。
    return pd.concat(features, axis=1)


def get_feature(group_day):
    """统一特征入口，后续新特征函数在这里依次调用。"""
    data = get_history_feature(group_day, windows=HISTORY_WINDOWS)
    return data


def get_past_only_label_prior(label_data):
    """为训练样本构造严格 date < t 的历史标签先验。"""
    columns = SAMPLE_KEYS + [LABEL]
    data = label_data[columns].sort_values(
        LABEL_PRIOR_KEYS + ['PROD_DATE']).reset_index(drop=True)
    prior = data[SAMPLE_KEYS].copy()
    for column in LABEL_PRIOR_COLUMNS:
        prior[column] = float('nan')

    for _, indices in data.groupby(LABEL_PRIOR_KEYS, sort=False).groups.items():
        indices = list(indices)
        values = data.loc[indices, LABEL].astype(float)
        dates = data.loc[indices, 'PROD_DATE']
        past = values.shift(1)
        past_short = past.le(3).where(past.notna()).astype(float)

        prior.loc[indices, 'label_hist_count'] = range(len(indices))
        prior.loc[indices, 'label_last_lag'] = past
        prior.loc[indices, 'label_last_is_short'] = past_short
        prior.loc[indices, 'label_hist_mean'] = past.expanding(min_periods=1).mean()
        prior.loc[indices, 'label_hist_median'] = past.expanding(min_periods=1).median()
        prior.loc[indices, 'label_hist_short_ratio'] = \
            past_short.expanding(min_periods=1).mean()
        prior.loc[indices, 'label_recent5_median'] = \
            past.rolling(5, min_periods=1).median()
        prior.loc[indices, 'label_recent5_short_ratio'] = \
            past_short.rolling(5, min_periods=1).mean()
        prior.loc[indices, 'label_days_since_last'] = \
            (dates - dates.shift(1)).dt.days

    prior['label_hist_count'] = prior['label_hist_count'].astype(int)
    return prior


def get_frozen_label_prior(samples, label_data, freeze_date):
    """使用冻结点之前的标签画像，为整个未来区间生成不更新的先验。"""
    freeze_date = pd.Timestamp(freeze_date)
    history = label_data[label_data['PROD_DATE'] < freeze_date].sort_values(
        LABEL_PRIOR_KEYS + ['PROD_DATE'])
    rows = []
    for keys, group in history.groupby(LABEL_PRIOR_KEYS, sort=False):
        values = group[LABEL].astype(float)
        recent = values.tail(5)
        rows.append({
            'WELL_GROUP_NAME': keys[0],
            'PROD_INDICATOR': keys[1],
            'label_hist_count': len(values),
            'label_last_lag': values.iloc[-1],
            'label_last_is_short': float(values.iloc[-1] <= 3),
            'label_hist_mean': values.mean(),
            'label_hist_median': values.median(),
            'label_hist_short_ratio': values.le(3).mean(),
            'label_recent5_median': recent.median(),
            'label_recent5_short_ratio': recent.le(3).mean(),
            '_label_last_date': group['PROD_DATE'].iloc[-1],
        })
    frozen = pd.DataFrame(rows)

    prior = samples[SAMPLE_KEYS].copy()
    prior['_row_id'] = range(len(prior))
    prior = pd.merge(prior, frozen, on=LABEL_PRIOR_KEYS, how='left',
                     validate='many_to_one')
    prior['label_hist_count'] = prior['label_hist_count'].fillna(0).astype(int)
    prior['label_days_since_last'] = \
        (prior['PROD_DATE'] - prior['_label_last_date']).dt.days
    prior = prior.sort_values('_row_id').drop(
        columns=['_row_id', '_label_last_date']).reset_index(drop=True)
    return prior


def get_dataset(label_data, feature_data, label_prior=None):
    """连接目标样本，保持样本原始顺序，标签放在最后。"""
    data = label_data.copy()
    data['_row_id'] = range(len(data))
    data = pd.merge(data, feature_data, on=DAY_KEYS, how='left', validate='many_to_one')
    if label_prior is not None:
        data = pd.merge(data, label_prior, on=SAMPLE_KEYS, how='left',
                        validate='one_to_one')
    data = data.sort_values('_row_id').drop(columns='_row_id').reset_index(drop=True)
    if LABEL in data.columns:
        label = data.pop(LABEL)
        data[LABEL] = label.astype(int)
    return data


def main():
    inj_data = pd.concat([prepare(pd.read_csv(os.path.join(DATA_DIR, split, 'well_inj_data.csv')))
                          for split in ['train', 'test']], ignore_index=True)
    prod_data = pd.concat([prepare(pd.read_csv(os.path.join(DATA_DIR, split, 'well_prod_data.csv')))
                           for split in ['train', 'test']], ignore_index=True)
    well_info = pd.read_csv(os.path.join(DATA_DIR, 'train', 'well_group_info.csv'))
    label_data = prepare(pd.read_csv(os.path.join(DATA_DIR, 'train', 'optimal_lag_days.csv')))
    test_template = pd.read_csv(os.path.join(DATA_DIR, 'test', 'test_optimal_lag_days.csv'))
    test_data = prepare(test_template.drop(columns=LABEL))

    print('构造 group-day 底表', flush=True)
    group_day = get_group_day(inj_data, prod_data, well_info, label_data, test_data)
    feature_data = get_feature(group_day)
    train_label = label_data[label_data['PROD_DATE'] < VALIDATE_START]
    validate_label = label_data[(label_data['PROD_DATE'] >= VALIDATE_START) &
                               (label_data['PROD_DATE'] < VALIDATE_END)]
    train_prior = get_past_only_label_prior(label_data)
    validate_prior = get_frozen_label_prior(validate_label, label_data, VALIDATE_START)
    test_prior = get_frozen_label_prior(test_data, label_data, TEST_FREEZE_DATE)
    train = get_dataset(train_label, feature_data, train_prior)
    validate = get_dataset(validate_label, feature_data, validate_prior)
    test = get_dataset(test_data, feature_data, test_prior)

    os.makedirs(DATASET_DIR, exist_ok=True)
    group_day.to_csv(os.path.join(DATASET_DIR, 'group_day.csv.gz'), index=False, encoding='utf-8-sig')
    for name, data in [('train', train), ('validate', validate), ('test', test)]:
        data.to_csv(os.path.join(DATASET_DIR, name + '.csv'), index=False, encoding='utf-8-sig')
        print(f'{name}: {data.shape}, 日期 {data.PROD_DATE.min().date()} 至 {data.PROD_DATE.max().date()}', flush=True)
    # 模型脚本只读 dataset 目录；此文件用于恢复提交日期字符串及行顺序。
    test_template.to_csv(os.path.join(DATASET_DIR, 'test_template.csv'), index=False, encoding='utf-8-sig')
    info = {'feature_version': FEATURE_VERSION, 'validate_start': VALIDATE_START,
            'history_window': {'days': HISTORY_WINDOWS, 'include_today': False, 'min_periods': 1,
                               'statistics': ['mean', 'std', 'min', 'max'], 'std_ddof': 1,
                               'source_columns': HISTORY_COLUMNS},
            'label_prior': {'keys': LABEL_PRIOR_KEYS, 'columns': LABEL_PRIOR_COLUMNS,
                            'train_mode': 'strict_past_only',
                            'validate_freeze_exclusive': VALIDATE_START,
                            'test_freeze_exclusive': TEST_FREEZE_DATE,
                            'recent_observations': 5},
            'validate_end_exclusive': VALIDATE_END, 'group_day_shape': list(group_day.shape),
            'train_shape': list(train.shape), 'validate_shape': list(validate.shape),
            'test_shape': list(test.shape), 'feature_columns': list(feature_data.columns)}
    with open(os.path.join(DATASET_DIR, 'dataset_info.json'), 'w', encoding='utf-8') as file:
        json.dump(info, file, ensure_ascii=False, indent=2)
    print('数据集已保存到:', DATASET_DIR, flush=True)


if __name__ == '__main__':
    main()
