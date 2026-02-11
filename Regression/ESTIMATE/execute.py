import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict

import numpy as np

from parser import build_config_from_args
from utils.common import seed, get_class

def run_model(config):
    model = get_class(os.path.join(args.model_path_folder, config["model"]["path"]))(config)
    metrics = {}
    if args.train:
        model.train()
    if args.test:
        metrics = model.test()
    return metrics

def get_symbols():
    symbols = np.load("data_estimate/preprocess/baseline_data.npy", allow_pickle=True).item()
    return list(map(str, symbols.keys()))

if __name__=='__main__':
    sys.path.append(os.getcwd())

    args = ArgumentParser()

    # Project
    args.add_argument('--run_name', default='sp500_estimate')
    args.add_argument('--model', type=str, default='estimate')
    args.add_argument('--model_path_folder', type=str, default='model/estimate/framework/')
    args.add_argument('--train', action='store_true', default=False)
    args.add_argument('--test', action='store_true', default=False)

    # Data (flat)
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--symbols', nargs='*', default=[])
    args.add_argument('--start_train', default='2015-01-01')
    args.add_argument('--end_train', default='2018-12-31')
    args.add_argument('--start_valid', default='2019-01-01')
    args.add_argument('--end_valid', default='2020-12-31')
    args.add_argument('--start_test', default='2021-01-01')
    args.add_argument('--end_test', default='2021-01-01')
    args.add_argument('--n_step_ahead', type=int, default=5)
    args.add_argument('--target_col', default='trend_return')
    args.add_argument('--include_target', action='store_true')
    args.add_argument('--history_window', type=int, default=20)
    args.add_argument('--outlier_threshold', type=float, default=1000)

    # Indicator args (flat)
    args.add_argument('--close_sma_medium', type=int, default=10)
    args.add_argument('--close_sma_slow', type=int, default=20)
    args.add_argument('--rsi_medium', type=int, default=10)
    args.add_argument('--rsi_slow', type=int, default=20)
    args.add_argument('--macd_medium', type=int, default=10)
    args.add_argument('--macd_slow', type=int, default=20)
    args.add_argument('--mfi_medium', type=int, default=10)
    args.add_argument('--mfi_slow', type=int, default=20)

    # Model (flat)
    args.add_argument('--path', default='supervisor.Supervisor')
    args.add_argument('--confidence_threshold', type=float, default=0.90)
    args.add_argument('--earlystop', type=int, default=16)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--hidden_dim', type=int, default=32)
    args.add_argument('--rnn_units', type=int, default=16)
    args.add_argument('--learning_rate', type=float, default=0.0001)
    args.add_argument('--cuda', action='store_true', default=True)
    args.add_argument('--resume', action='store_true')
    args.add_argument('--save_best', action='store_true')
    args.add_argument('--lr_decay', type=float, default=0.005)
    args.add_argument('--eval_iter', type=int, default=10)
    args.add_argument('--dropout', type=float, default=0.4)
    args.add_argument('--verify_threshold', type=float, default=0.08)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--job_id', type=int, default=0)

    # Backtest
    args.add_argument('--config_path', default='backtest/config/normal.yaml')

    args = args.parse_args()

    config = asdict(build_config_from_args(args))
    config['data']['symbols'] = get_symbols()

    seed(args.seed)

    base_pretrained_run_folder = "pretrained/{}/{}".format(args.model, args.run_name)
    pretrained_folder = base_pretrained_run_folder + "-{}/rb".format(args.job_id)
    config['model']['pretrained_log'] = pretrained_folder
    os.makedirs(pretrained_folder, exist_ok=True)

    performances = run_model(config)

    print('Test performances:', performances)