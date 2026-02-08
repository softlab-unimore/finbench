# -*-Encoding: utf-8 -*-
import argparse
import glob
import json
import pickle
import shutil

import torch
import numpy as np
import random

from sklearn.metrics import r2_score

from exp.exp_model import Exp_Model
import os

from preprocessing import load_dataset


def get_metrics(preds, labels):
    preds_flat = np.array(preds)
    labels_flat = np.array(labels)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat)
    }
    return metrics


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='generating')

    # Load data
    parser.add_argument('--universe', type=str, default='sx5e')
    parser.add_argument('--model_name', type=str, default='DVa')
    parser.add_argument('--root_path', type=str, default='./data/sx5e', help='root path of the data files')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--job_id', type=int, default=0, help='job id')

    parser.add_argument('--sequence_length', type=int, default=2, help='length of input sequence')
    parser.add_argument('--prediction_length', type=int, default=None, help='prediction sequence length')
    parser.add_argument('--start_date', type=str, default='2016-01-01', help='start date')
    parser.add_argument('--end_train_date', type=str, default='2019-12-31', help='train end date')
    parser.add_argument('--start_valid_date', type=str, default='2020-01-01', help='validation start date')
    parser.add_argument('--end_valid_date', type=str, default='2020-12-31', help='validation end date')
    parser.add_argument('--start_test_date', type=str, default='2021-01-01', help='test start date')
    parser.add_argument('--end_date', type=str, default='2021-12-31', help='end date')

    parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
    parser.add_argument('--input_dim', type=int, default=6, help='dimension of input')
    parser.add_argument('--hidden_size', type=int, default=128, help='encoder dimension')
    parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

    # Diffusion process
    parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
    parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
    parser.add_argument('--beta_end', type=float, default=1.0, help='end of the beta')
    parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

    # Bidirectional VAE
    parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')
    parser.add_argument('--mult', type=float, default=1, help='mult of channels')
    parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
    parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
    parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
    parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
    parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
    parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

    # Training settings
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
    parser.add_argument('--zeta', type=float, default=0.5, help='trade off parameter zeta')
    parser.add_argument('--eta', type=float, default=1.0, help='trade off parameter eta')
    parser.add_argument('--seed', type=int, default=100, help='random seed for reproducibility')

    # Device
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()

    metrics_path = f"./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split('-')[0]}"
    os.makedirs(metrics_path, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.prediction_length is None:
        args.prediction_length = args.sequence_length

    print('Args in experiment:')
    print(args)

    os.makedirs(f'{args.root_path}/preprocessed_{args.job_id}', exist_ok=True)
    tickers = load_dataset(args.start_date, args.end_date, args.end_train_date, args.end_valid_date, args.start_test_date,
                           args.root_path, args.universe, args.job_id, args.batch_size, args.sequence_length, args.prediction_length)

    Exp = Exp_Model
    train_setting = f'tp{args.root_path.split(os.sep)[-1]}_sl{ args.sequence_length}_seed{args.seed}_jobid{args.job_id}'

    preds = []
    labels = []
    pred_dates = []
    last_dates = []
    all_tickers = []

    val_preds = []
    val_labels = []
    val_pred_dates = []
    val_last_dates = []
    val_tickers = []

    args.root_path = f'{args.root_path}/preprocessed_{args.job_id}'

    for ticker in tickers:
        print('-----------------------------------------')
        print('Training model for ticker:', ticker)
        print('-----------------------------------------')

        args.data_path = f"{ticker.replace('.', '_')}.csv"

        setting = args.data_path + '_' + train_setting
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        pred, label, pred_date, last_date = exp.test(setting, args.sequence_length, args.prediction_length)

        if np.isnan(pred).any() or np.isnan(label).any():
            print(f'NaN predictions for ticker {ticker}, skipping...')
            continue

        preds.extend(pred)
        labels.extend(label)
        pred_dates.extend(pred_date)
        last_dates.extend(last_date)
        all_tickers.extend([ticker] * len(pred))

        print('>>>>>>>start validation : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        val_pred, val_label, val_pred_date, val_last_date = exp.validate(setting, args.sequence_length, args.prediction_length)

        val_preds.extend(val_pred)
        val_labels.extend(val_label)
        val_pred_dates.extend(val_pred_date)
        val_last_dates.extend(val_last_date)
        val_tickers.extend([ticker] * len(val_pred))

        torch.cuda.empty_cache()

        os.remove(os.path.join(args.root_path, args.data_path))
        shutil.rmtree(os.path.join(args.checkpoints, setting))

    metrics = get_metrics(preds, labels)
    metrics = {k: float(v) for k, v in metrics.items()}

    val_metrics = get_metrics(val_preds, val_labels)
    val_metrics = {k: float(v) for k, v in val_metrics.items()}

    results = {
        'metrics': metrics,
        'preds': np.array(preds),
        'labels': np.array(labels),
        'pred_date': pred_dates,
        'last_date': last_dates,
        'tickers': all_tickers
    }

    with open(f'{metrics_path}/results_sl{args.sequence_length}_pl{args.prediction_length}.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open(f'{metrics_path}/metrics_sl{args.sequence_length}_pl{args.prediction_length}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(f'{metrics_path}/val_metrics_sl{args.sequence_length}_pl{args.prediction_length}.json', 'w') as f:
        json.dump(val_metrics, f, indent=4)


