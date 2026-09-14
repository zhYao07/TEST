from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


CAT_FEATURES = ['WELL_GROUP_NAME', 'PROD_INDICATOR']


def load_models(model_dir):
    model_dir = Path(model_dir)
    models = {
        'segment': CatBoostClassifier(),
        'short': CatBoostClassifier(),
        'long': CatBoostRegressor(),
    }
    for name, model in models.items():
        model.load_model(str(model_dir / f'{name}.cbm'))
    return models


def get_model_data(data, feature_columns):
    feature = data[feature_columns].copy()
    for column in CAT_FEATURES:
        feature[column] = feature[column].fillna('UNKNOWN').astype(str)
    return feature


def short_median(probability):
    return (probability.cumsum(axis=1) >= 0.5).argmax(axis=1)


def predict(models, test, short_threshold, long_safe_value):
    feature_columns = models['segment'].feature_names_
    feature = get_model_data(test, feature_columns)

    segment_model = models['segment']
    p_short = segment_model.predict_proba(feature)[
        :, list(segment_model.classes_).index(1)]

    short_model = models['short']
    short_probability = np.zeros((len(test), 4))
    short_probability[:, short_model.classes_.astype(int)] = \
        short_model.predict_proba(feature)
    short_prediction = short_median(short_probability)

    # Long 回归模型随代码包保留，但当前评分适配策略使用安全值 4。
    prediction = np.where(
        p_short >= short_threshold,
        short_prediction,
        long_safe_value,
    )
    return prediction.astype(int)
