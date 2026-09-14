import os
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from get_dataset import DATASET_DIR, BASE_DIR, SAMPLE_KEYS, LABEL, FEATURE_VERSION


OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'catboost_single_regressor')
CAT_FEATURES = ['WELL_GROUP_NAME', 'PROD_INDICATOR']
DROP_COLUMNS = ['PROD_DATE', 'INJ_INDICATOR', LABEL]
EARLY_STOPPING_ROUNDS = 80

PARAMS = {
    'iterations': 1000,
    'depth': 7,
    'learning_rate': 0.01,
    'l2_leaf_reg': 5,
    'loss_function': 'MAE',
    'random_seed': 42,
    'thread_count': 8,
    'task_type': 'CPU',
    'allow_writing_files': False,
    'verbose': False,
}


def contest_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    short = y_true <= 3
    accuracy = float(np.mean(short == (y_pred <= 3)))
    mae_short = float(np.abs(y_true[short] - y_pred[short]).mean()) if short.any() else None
    mae_long = float(np.abs(y_true[~short] - y_pred[~short]).mean()) if (~short).any() else None
    s_seg = 60 * accuracy
    s_short = max(0, 30 - 9 * mae_short) if mae_short is not None else 0
    s_long = max(0, 10 - 0.2 * mae_long) if mae_long is not None else 0
    return {'n': len(y_true), 'segment_accuracy': accuracy,
            'MAE_short': mae_short, 'MAE_long': mae_long,
            'S_seg': s_seg, 'S_short': s_short, 'S_long': s_long,
            'score': s_seg + s_short + s_long}


def get_model_data(data, feature_columns):
    feature = data[feature_columns].copy()
    for column in CAT_FEATURES:
        feature[column] = feature[column].fillna('UNKNOWN').astype(str)
    return feature


def predict_catboost(model, test):
    feature = get_model_data(test, model.feature_names_)
    prediction_raw = np.clip(model.predict(feature), 0, 45)
    prediction = np.rint(prediction_raw).astype(int)
    result = test[SAMPLE_KEYS].reset_index(drop=True).copy()
    result['prediction_raw'] = prediction_raw
    result['prediction'] = prediction
    return result


def model_catboost(train, validate, test, iterations=None, feature_columns=None):
    if feature_columns is None:
        feature_columns = [column for column in train if column not in DROP_COLUMNS and
                           (column in CAT_FEATURES or train[column].nunique(dropna=True) > 1)]
    params = {**PARAMS, 'cat_features': CAT_FEATURES}
    if iterations is not None:
        params['iterations'] = iterations
    model = CatBoostRegressor(**params)
    x_train = get_model_data(train, feature_columns)
    fit_params = {}
    if validate is not None:
        x_validate = get_model_data(validate, feature_columns)
        fit_params = {'eval_set': (x_validate, validate[LABEL]),
                      'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                      'use_best_model': True}
    model.fit(x_train, train[LABEL], **fit_params)
    result = predict_catboost(model, test)
    return result, model


def save_model(model, path):
    model.save_model(path)


def load_model(path):
    model = CatBoostRegressor()
    model.load_model(path)
    return model


def main():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    validate = pd.read_csv(os.path.join(DATASET_DIR, 'validate.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('train / validate / test:', train.shape, validate.shape, test.shape, flush=True)
    start_time = time.time()

    print('训练统一回归模型', flush=True)
    result_valid, model_valid = model_catboost(
        train, validate, validate.drop(columns=LABEL))
    result_valid[LABEL] = validate[LABEL].to_numpy()
    metrics = contest_score(result_valid[LABEL], result_valid['prediction'])
    reference = contest_score(validate[LABEL], np.ones(len(validate)))
    iterations = model_valid.tree_count_
    feature_columns = model_valid.feature_names_
    print('实际树数:', iterations, flush=True)
    print('验证结果:', metrics, flush=True)

    print('使用全部标签重训统一回归模型', flush=True)
    big_train = pd.concat([train, validate], ignore_index=True)
    result, model = model_catboost(
        big_train, None, test, iterations=iterations, feature_columns=feature_columns)
    save_model(model, os.path.join(OUTPUT_DIR, 'final_model.cbm'))

    submission = pd.read_csv(os.path.join(DATASET_DIR, 'test_template.csv'))
    submission[LABEL] = result['prediction'].to_numpy()
    submission.to_csv(os.path.join(OUTPUT_DIR, 'test_optimal_lag_days.csv'),
                      index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time
    summary = {
        'model': 'CatBoostRegressor',
        'feature_version': FEATURE_VERSION,
        'params': PARAMS,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'selected_iterations': iterations,
        'feature_columns': feature_columns,
        'feature_count': len(feature_columns),
        'train_rows': len(train),
        'validate_rows': len(validate),
        'test_rows': len(test),
        'validate_start': validate.PROD_DATE.min(),
        'validate_end': validate.PROD_DATE.max(),
        'metrics': metrics,
        'constant_one_reference': reference,
        'seconds': elapsed,
        'prediction_rule': 'clip to [0, 45], then np.rint',
        'output_files': ['test_optimal_lag_days.csv', 'final_model.cbm',
                         'training_summary.json'],
    }
    with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f'训练及推理完成，用时 {elapsed:.2f} 秒，输出目录: {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
