import argparse
from pathlib import Path

from src.feature import build_test_dataset
from src.model import load_models, predict
from src.utils import check_input_files, load_config, save_result


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='浅层超稠油注采响应滞后预测')
    parser.add_argument('--data_dir', default='./data',
                        help='赛事原始 data 目录，内部包含 train/ 和 test/')
    parser.add_argument('--output', default='./test_optimal_lag_days.csv',
                        help='预测结果 CSV 保存路径')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'),
                        help='配置文件路径')
    args = parser.parse_args()

    check_input_files(args.data_dir)
    config = load_config(args.config)
    template, test = build_test_dataset(args.data_dir, config)
    model_dir = ROOT / config['model']['model_dir']
    models = load_models(model_dir)
    prediction = predict(
        models,
        test,
        config['model']['short_threshold'],
        config['model']['long_safe_value'],
    )
    result = save_result(template, prediction, args.output)
    print(f'预测完成：{len(result)} 行')
    print(f'结果文件：{Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
