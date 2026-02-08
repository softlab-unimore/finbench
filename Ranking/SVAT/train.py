import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch

from experiments.exp_svat import Exp_SVAT
from utils.load_data import save_dates
from utils.metric import get_metrics
from utils.optimize import set_seed


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sp500', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='SVAT', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=20, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=60, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2015-01-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2018-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2019-01-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2019-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2020-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2020-12-31', help='End date for the dataset')
    args.add_argument('--seed', type=int, default=42, help='Random seed')

    args.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    args.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    args.add_argument('--adv_eps', type=float, default=0.005, help='Epsilon for adversarial training')
    args.add_argument('--reg_alpha', type=float, default=0.5, help='Regularization alpha for ranking loss')
    args.add_argument('--kl_lambda', type=float, default=1.0, help='KL divergence lambda')

    args = args.parse_args()

    metrics_path = f'./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    gpus = 0 if device.type == 'cuda' else None

    param_dict = dict(
        data_path = [args.data_path],
        universe = [args.universe],
        start_date = [args.start_date],
        end_train_date = [args.end_train_date],
        start_valid_date = [args.start_valid_date],
        end_valid_date = [args.end_valid_date],
        start_test_date = [args.start_test_date],
        end_date = [args.end_date],
        history_len = [args.seq_len],
        pred_len = [args.pred_len],
        fea_dim = [5],
        hid_size = [32],
        drop_rate = [0.1],
        z_dim = [32],
        adv_hid_size = [128],
        learning_rate = [0.0001],
        adv_lr = [0.0001],
        adv_eps = [args.adv_eps],
        kl_lambda = [args.kl_lambda],
        reg_alpha = [args.reg_alpha],
        eta = [10.],
        adv = ['Attention'],
        epochs = [args.epochs],
        lradj = ['decayed'],
        patience = [2],
        clip = [10.],
        sample_size = [50],
        rank_sign = [True]
    )

    set_seed(args.seed)
    exp = Exp_SVAT(param_dict, gpus=gpus)

    test_pred, test_gt, mask, valid_pred, valid_gt, valid_mask = exp.train()

    valid_pred = valid_pred.swapaxes(0,1)
    valid_gt = valid_gt.swapaxes(0,1)
    valid_metrics = get_metrics(valid_gt, valid_pred)
    valid_metrics = {k: float(v) for k, v in valid_metrics.items()}
    print(f'Validation Metrics: {valid_metrics}')

    test_pred = test_pred.swapaxes(0,1)
    test_gt = test_gt.swapaxes(0,1)
    test_metrics = get_metrics(test_gt, test_pred)
    test_metrics = {k: float(v) for k, v in test_metrics.items()}
    dates_gt, last_seq_date, tickers = save_dates(exp.dates, exp.tickers, exp.test_index, args.pred_len, args.seq_len)

    test_pred = [np.expand_dims(test_pred[i], axis=1) for i in range(test_pred.shape[0])]
    test_gt = [np.expand_dims(test_gt[i], axis=1) for i in range(test_gt.shape[0])]

    results = {
        'metrics': test_metrics,
        'preds': test_pred,
        'labels': test_gt,
        'pred_date': dates_gt,
        'last_date': last_seq_date,
        'tickers': tickers
    }

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(valid_metrics, f, indent=4)

    with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)

    with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f'Metrics: {test_metrics}')

    torch.cuda.empty_cache()