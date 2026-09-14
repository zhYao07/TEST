import os
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from get_dataset import DATASET_DIR, BASE_DIR, SAMPLE_KEYS, LABEL, FEATURE_VERSION


OUTPUT_DIR = os.path.join(
    BASE_DIR, 'outputs', 'catboost_dec_feb_safe_long4_label_prior_v1')
CAT_FEATURES = ['WELL_GROUP_NAME', 'PROD_INDICATOR']
DROP_COLUMNS = ['PROD_DATE', 'INJ_INDICATOR', LABEL]
SHORT_THRESHOLD = 0.56
LONG_SAFE_VALUE = 4
EARLY_STOPPING_ROUNDS = 150

# 在这里修改参数；三个模型共享基础参数，各自使用对应的损失函数。
PARAMS = {
    'iterations': 1000,
    'depth': 7,
    'learning_rate': 0.02,
    'l2_leaf_reg': 5,
    'random_seed': 42,
    'thread_count': 8,
    'task_type': 'CPU',
    'allow_writing_files': False,
    'verbose': False,
}
STAGE_PARAMS = {
    'segment': {'loss_function': 'Logloss'},
    'short': {'loss_function': 'MultiClass', 'classes_count': 4},
    'long': {'loss_function': 'MAE'},
}


def contest_score(y_true, y_pred):
    """按真实标签分段计算 MAE，包括路由错误的样本。"""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError('预测包含 NaN 或无穷值')
    short = y_true <= 3
    accuracy = float(np.mean(short == (y_pred <= 3)))
    mae_short = float(np.abs(y_true[short] - y_pred[short]).mean()) if short.any() else None
    mae_long = float(np.abs(y_true[~short] - y_pred[~short]).mean()) if (~short).any() else None
    s_seg = 60 * accuracy
    s_short = max(0, 30 - 9 * mae_short) if mae_short is not None else 0
    s_long = max(0, 10 - 0.2 * mae_long) if mae_long is not None else 0
    return {'n': len(y_true), 'segment_accuracy': accuracy, 'MAE_short': mae_short,
            'MAE_long': mae_long, 'S_seg': s_seg, 'S_short': s_short,
            'S_long': s_long, 'score': s_seg + s_short + s_long}


def scan_short_thresholds(result, y_true):
    """固定两个专家输出，只在内存中按官方总分扫描 Router 阈值。"""
    rows = []
    for threshold in np.arange(0.20, 0.801, 0.01):
        prediction = np.where(
            result['p_short'].to_numpy() >= threshold,
            result['short_prediction'].to_numpy(),
            LONG_SAFE_VALUE,
        )
        rows.append({
            'threshold': round(float(threshold), 2),
            **contest_score(y_true, prediction),
        })
    return sorted(rows, key=lambda row: row['score'], reverse=True)


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def binary_auc(y_true, probability):
    """使用平均秩计算二分类 AUC；不额外引入 sklearn 依赖。"""
    y_true = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    positive_count = int(y_true.sum())
    negative_count = int((~y_true).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = pd.Series(probability).rank(method='average').to_numpy()
    rank_sum = ranks[y_true].sum()
    return float((rank_sum - positive_count * (positive_count + 1) / 2) /
                 (positive_count * negative_count))


def stage_diagnostics(result, y_true, short_threshold=SHORT_THRESHOLD,
                      long_safe_value=LONG_SAFE_VALUE):
    """分别衡量 Router、Short Expert 和 Long Expert 的独立验证能力。"""
    y_true = np.asarray(y_true, dtype=int)
    true_short = y_true <= 3
    predicted_short = result['p_short'].to_numpy() >= short_threshold
    p_short = np.clip(result['p_short'].to_numpy(dtype=float), 1e-15, 1 - 1e-15)

    router_confusion = np.array([
        [np.sum(true_short & predicted_short), np.sum(true_short & ~predicted_short)],
        [np.sum(~true_short & predicted_short), np.sum(~true_short & ~predicted_short)],
    ], dtype=int)
    short_tp, short_fn = router_confusion[0]
    short_fp, long_tp = router_confusion[1]
    short_precision = safe_divide(short_tp, short_tp + short_fp)
    short_recall = safe_divide(short_tp, short_tp + short_fn)
    long_precision = safe_divide(long_tp, long_tp + short_fn)
    long_recall = safe_divide(long_tp, long_tp + short_fp)
    router = {
        'threshold': short_threshold,
        'n': len(y_true),
        'true_short_count': int(true_short.sum()),
        'true_long_count': int((~true_short).sum()),
        'predicted_short_count': int(predicted_short.sum()),
        'predicted_long_count': int((~predicted_short).sum()),
        'accuracy': float(np.mean(true_short == predicted_short)),
        'official_s_seg': float(60 * np.mean(true_short == predicted_short)),
        'balanced_accuracy': float((short_recall + long_recall) / 2),
        'short_precision': short_precision,
        'short_recall': short_recall,
        'short_f1': (
            safe_divide(2 * short_precision * short_recall,
                        short_precision + short_recall)
            if short_precision is not None and short_recall is not None else None
        ),
        'long_precision': long_precision,
        'long_recall': long_recall,
        'long_f1': (
            safe_divide(2 * long_precision * long_recall,
                        long_precision + long_recall)
            if long_precision is not None and long_recall is not None else None
        ),
        'auc_short': binary_auc(true_short, p_short),
        'logloss': float(-np.mean(true_short * np.log(p_short) +
                                  (~true_short) * np.log(1 - p_short))),
        'brier_score': float(np.mean((p_short - true_short.astype(float)) ** 2)),
        'confusion_matrix_rows_true_cols_pred_short_long': router_confusion.tolist(),
    }

    short_true = y_true[true_short]
    short_pred = result.loc[true_short, 'short_prediction'].to_numpy(dtype=int)
    short_probability = result.loc[
        true_short, [f'p{i}_given_short' for i in range(4)]].to_numpy(dtype=float)
    short_error = short_pred - short_true
    short_confusion = np.zeros((4, 4), dtype=int)
    np.add.at(short_confusion, (short_true, short_pred), 1)
    true_class_probability = np.clip(
        short_probability[np.arange(len(short_true)), short_true], 1e-15, 1)
    short_by_class = {}
    for value in range(4):
        mask = short_true == value
        short_by_class[str(value)] = {
            'n': int(mask.sum()),
            'accuracy': float(np.mean(short_pred[mask] == short_true[mask])) if mask.any() else None,
            'mae': float(np.mean(np.abs(short_error[mask]))) if mask.any() else None,
            'mean_prediction': float(np.mean(short_pred[mask])) if mask.any() else None,
        }
    short_expert = {
        'evaluation_scope': '仅真实标签为0至3的样本，假设Router全部路由正确',
        'n': len(short_true),
        'accuracy': float(np.mean(short_pred == short_true)),
        'mae': float(np.mean(np.abs(short_error))),
        'official_s_short_if_oracle_routed': float(
            max(0, 30 - 9 * np.mean(np.abs(short_error)))),
        'rmse': float(np.sqrt(np.mean(short_error ** 2))),
        'mean_error_prediction_minus_truth': float(np.mean(short_error)),
        'within_1_day_rate': float(np.mean(np.abs(short_error) <= 1)),
        'multiclass_logloss': float(-np.mean(np.log(true_class_probability))),
        'confusion_matrix_rows_true_cols_pred_0_1_2_3': short_confusion.tolist(),
        'by_true_class': short_by_class,
    }

    long_true = y_true[~true_short]
    long_raw = result.loc[~true_short, 'long_prediction_raw'].to_numpy(dtype=float)
    long_pred = result.loc[~true_short, 'long_prediction'].to_numpy(dtype=int)
    long_error_raw = long_raw - long_true
    long_error = long_pred - long_true
    safe_error = long_safe_value - long_true
    long_by_range = {}
    for lower, upper in [(4, 10), (11, 20), (21, 30), (31, 45)]:
        mask = (long_true >= lower) & (long_true <= upper)
        long_by_range[f'{lower}-{upper}'] = {
            'n': int(mask.sum()),
            'raw_mae': float(np.mean(np.abs(long_error_raw[mask]))) if mask.any() else None,
            'rounded_mae': float(np.mean(np.abs(long_error[mask]))) if mask.any() else None,
            'safe_constant_mae': float(np.mean(np.abs(safe_error[mask]))) if mask.any() else None,
            'mean_prediction': float(np.mean(long_pred[mask])) if mask.any() else None,
        }
    long_expert = {
        'evaluation_scope': '仅真实标签为4至45的样本，假设Router全部路由正确',
        'n': len(long_true),
        'raw_mae': float(np.mean(np.abs(long_error_raw))),
        'rounded_mae': float(np.mean(np.abs(long_error))),
        'official_s_long_if_oracle_routed': float(
            max(0, 10 - 0.2 * np.mean(np.abs(long_error)))),
        'rounded_rmse': float(np.sqrt(np.mean(long_error ** 2))),
        'mean_error_prediction_minus_truth': float(np.mean(long_error)),
        'within_1_day_rate': float(np.mean(np.abs(long_error) <= 1)),
        'within_3_days_rate': float(np.mean(np.abs(long_error) <= 3)),
        'within_5_days_rate': float(np.mean(np.abs(long_error) <= 5)),
        'prediction_min': int(long_pred.min()),
        'prediction_max': int(long_pred.max()),
        'prediction_mean': float(np.mean(long_pred)),
        'safe_constant_value': long_safe_value,
        'safe_constant_mae': float(np.mean(np.abs(safe_error))),
        'safe_constant_official_s_long_if_oracle_routed': float(
            max(0, 10 - 0.2 * np.mean(np.abs(safe_error)))),
        'by_true_range': long_by_range,
    }
    return {'router': router, 'short_expert': short_expert, 'long_expert': long_expert}


def short_median(probability):
    """短段输出条件分布中位数，最小化期望绝对误差。"""
    return (probability.cumsum(axis=1) >= 0.5).argmax(axis=1)


def get_model_data(data, feature_columns):
    feature = data[feature_columns].copy()
    for column in CAT_FEATURES:
        feature[column] = feature[column].fillna('UNKNOWN').astype(str)
    return feature


def predict_catboost(models, test):
    """使用三个已训练模型推理，返回样本键、概率和最终整数预测。"""
    feature = get_model_data(test, models['segment'].feature_names_)
    segment_model = models['segment']
    p_short = segment_model.predict_proba(feature)[:, list(segment_model.classes_).index(1)]
    short_model = models['short']
    short_probability = np.zeros((len(test), 4))
    short_probability[:, short_model.classes_.astype(int)] = short_model.predict_proba(feature)
    short_prediction = short_median(short_probability)
    long_raw = np.clip(models['long'].predict(feature), 4, 45)
    long_prediction = np.rint(long_raw).astype(int)

    result = test[SAMPLE_KEYS].reset_index(drop=True).copy()
    result['p_short'] = p_short
    for i in range(4):
        result[f'p{i}_given_short'] = short_probability[:, i]
    result['short_prediction'] = short_prediction
    result['long_prediction_raw'] = long_raw
    result['long_prediction'] = long_prediction
    # Long 回归到约 30 天会放大误入 Long 的真实短段误差。
    # 当前安全解码统一输出 4，原 Long 回归结果仍保留用于诊断。
    result['prediction'] = np.where(p_short >= SHORT_THRESHOLD, short_prediction, LONG_SAFE_VALUE)
    return result


def model_catboost(train, validate, test, iterations=None, feature_columns=None):
    """验证训练时传 validate；全量重训时传 None，并传入已选定的各阶段树数。"""
    if feature_columns is None:
        feature_columns = [column for column in train if column not in DROP_COLUMNS and
                           (column in CAT_FEATURES or train[column].nunique(dropna=True) > 1)]
    x_train = get_model_data(train, feature_columns)
    y_train = train[LABEL].to_numpy()
    if validate is not None:
        x_validate = get_model_data(validate, feature_columns)
        y_validate = validate[LABEL].to_numpy()

    models = {}
    for stage in ['segment', 'short', 'long']:
        params = {**PARAMS, **STAGE_PARAMS[stage], 'cat_features': CAT_FEATURES}
        if iterations is not None:
            params['iterations'] = iterations[stage]
        model_class = CatBoostRegressor if stage == 'long' else CatBoostClassifier
        model = model_class(**params)
        if stage == 'segment':
            mask = np.ones(len(train), dtype=bool)
            target = (y_train <= 3).astype(int)
        else:
            mask = y_train <= 3 if stage == 'short' else y_train >= 4
            target = y_train[mask]

        fit_params = {}
        if validate is not None:
            if stage == 'segment':
                valid_mask = np.ones(len(validate), dtype=bool)
                valid_target = (y_validate <= 3).astype(int)
            else:
                valid_mask = y_validate <= 3 if stage == 'short' else y_validate >= 4
                valid_target = y_validate[valid_mask]
            fit_params = {'eval_set': (x_validate.loc[valid_mask], valid_target),
                          'early_stopping_rounds': EARLY_STOPPING_ROUNDS, 'use_best_model': True}
        print(f'训练 {stage}: {mask.sum()} 条样本', flush=True)
        model.fit(x_train.loc[mask], target, **fit_params)
        models[stage] = model
        print(f'{stage} 实际树数: {model.tree_count_}', flush=True)
    return predict_catboost(models, test), models


def save_models(models, directory):
    os.makedirs(directory, exist_ok=True)
    for stage, model in models.items():
        model.save_model(os.path.join(directory, stage + '.cbm'))


def load_models(directory):
    """加载保存的模型；之后可直接调用 predict_catboost(models, test)。"""
    models = {}
    for stage in ['segment', 'short', 'long']:
        model = CatBoostRegressor() if stage == 'long' else CatBoostClassifier()
        model.load_model(os.path.join(directory, stage + '.cbm'))
        models[stage] = model
    return models


def main():
    train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    validate = pd.read_csv(os.path.join(DATASET_DIR, 'validate.csv'))
    test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('train / validate / test:', train.shape, validate.shape, test.shape, flush=True)
    start_time = time.time()

    # 先做线下验证，验证集不加入这一步的训练。
    result_valid, models_valid = model_catboost(train, validate, validate.drop(columns=LABEL))
    result_valid[LABEL] = validate[LABEL].to_numpy()
    metrics = contest_score(result_valid[LABEL], result_valid['prediction'])
    threshold_scan = scan_short_thresholds(result_valid, result_valid[LABEL])
    validate_month = pd.to_datetime(validate['PROD_DATE']).dt.to_period('M').astype(str)
    monthly_metrics = {
        month: contest_score(
            result_valid.loc[validate_month.eq(month), LABEL],
            result_valid.loc[validate_month.eq(month), 'prediction'],
        )
        for month in sorted(validate_month.unique())
    }
    diagnostics = stage_diagnostics(result_valid, result_valid[LABEL])
    reference = contest_score(validate[LABEL], np.ones(len(validate)))
    iterations = {stage: model.tree_count_ for stage, model in models_valid.items()}
    features = models_valid['segment'].feature_names_
    print('验证结果:', metrics, flush=True)
    diagnostics_path = os.path.join(OUTPUT_DIR, 'stage_diagnostics.json')
    with open(diagnostics_path, 'w', encoding='utf-8') as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)
    print('三阶段独立能力诊断已保存:', diagnostics_path, flush=True)

    # 用全部已知标签重训，沿用验证选定的特征和树数，不再对同一验证集早停。
    big_train = pd.concat([train, validate], ignore_index=True)
    result, models = model_catboost(big_train, None, test, iterations, features)
    save_models(models, os.path.join(OUTPUT_DIR, 'final_models'))
    submission = pd.read_csv(os.path.join(DATASET_DIR, 'test_template.csv'))
    submission[LABEL] = result['prediction'].to_numpy()
    submission.to_csv(os.path.join(OUTPUT_DIR, 'test_optimal_lag_days.csv'),
                      index=False, encoding='utf-8-sig')

    elapsed = time.time() - start_time
    summary = {'feature_version': FEATURE_VERSION,
               'params': PARAMS, 'stage_params': STAGE_PARAMS, 'short_threshold': SHORT_THRESHOLD,
               'long_output_mode': 'safe_constant', 'long_safe_value': LONG_SAFE_VALUE,
               'early_stopping_rounds': EARLY_STOPPING_ROUNDS, 'selected_iterations': iterations,
               'feature_columns': features, 'feature_count': len(features),
               'excluded_columns': [c for c in train if c not in features],
               'train_rows': len(train), 'validate_rows': len(validate), 'test_rows': len(test),
               'validate_start': validate.PROD_DATE.min(), 'validate_end': validate.PROD_DATE.max(),
               'metrics': metrics, 'monthly_metrics': monthly_metrics,
               'threshold_scan_top10': threshold_scan[:10],
               'constant_one_reference': reference, 'seconds': elapsed,
               'output_files': ['test_optimal_lag_days.csv', 'final_models/',
                                'stage_diagnostics.json',
                                'training_summary.json'],
               'note': '十二月至二月参与各阶段早停；Long回归保留作诊断，最终Long路由固定输出4天。'}
    with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f'训练及推理完成，用时 {elapsed:.2f} 秒，输出目录: {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
