import os
from argparse import ArgumentParser
from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from load_dataset import RobustZScoreNormalization, CustomDataset
from utils import build_config_from_args, filter_constituents_by_date
from finformer import trainer


def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df


def preprocess_matrix(adj_matrix):
    return (adj_matrix.max(axis=0) > 0).astype(np.float32)


def custom_collate(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == 'instruments':
            batch_dict[key] = list(chain.from_iterable(item[key] for item in batch))
        else:
            batch_dict[key] = torch.utils.data._utils.collate.default_collate(
                [item[key] for item in batch]
            )
    return batch_dict


if __name__ == '__main__':
    args = ArgumentParser()

    # Dataset parameters
    args.add_argument('--data_path', type=str, default='./data')
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--model_name', type=str, default='FinFormer')

    # Prediction parameters
    args.add_argument('--seq_len', type=int, default=None)
    args.add_argument('--pred_len', type=int, default=5)
    args.add_argument('--start_date', type=str, default='2015-10-01')
    args.add_argument('--end_train_date', type=str, default='2018-12-31')
    args.add_argument('--start_valid_date', type=str, default='2019-10-01')
    args.add_argument('--end_valid_date', type=str, default='2019-12-31')
    args.add_argument('--start_test_date', type=str, default='2020-10-01')
    args.add_argument('--end_date', type=str, default='2020-12-31')

    # Model Hyperparameters
    args.add_argument('--d_feat', type=int, default=6)
    args.add_argument('--hidden_size', type=int, default=64)
    args.add_argument('--num_layers', type=int, default=2)
    args.add_argument('--temporal_dropout', type=float, default=0.4)
    args.add_argument('--snum_head', type=int, default=4)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_epochs', type=int, default=200)
    args.add_argument('--lr', type=float, default=0.0002)

    args = args.parse_args()
    data_config, model_config = build_config_from_args(args)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    metrics_path = f"./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split('-')[0]}"
    os.makedirs(metrics_path, exist_ok=True)

    df_alpha = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_alpha360.csv')
    tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date)
    df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

    # Extract labels
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')[['date', 'instrument', 'adj_close']]
    df_close = df_close.sort_values(['instrument', 'date'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-args.pred_len) - x) / x)
    df_alpha = df_alpha.merge(df_close[['date', 'instrument', 'Label']], on=['date', 'instrument'], how='left')

    df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

    robust_z_score = RobustZScoreNormalization(df_alpha[(df_alpha['date'] >= args.start_date) & (df_alpha['date'] <= args.end_train_date)])

    df_train = CustomDataset(
        df_alpha,
        d_feat=args.d_feat,
        pred_len=args.pred_len,
        start_date=args.start_date,
        end_date=args.end_train_date,
        z_score=robust_z_score,
        period='train'
    )
    train_loader = DataLoader(df_train, shuffle=True, drop_last=True, collate_fn=custom_collate)

    df_valid = CustomDataset(
        df_alpha,
        d_feat=args.d_feat,
        pred_len=args.pred_len,
        start_date=args.start_valid_date,
        end_date=args.end_valid_date,
        z_score=robust_z_score,
        period='valid'
    )
    valid_loader = DataLoader(df_valid, shuffle=False, drop_last=False, collate_fn=custom_collate)

    df_test = CustomDataset(
        df_alpha,
        d_feat=args.d_feat,
        pred_len=args.pred_len,
        start_date=args.start_test_date,
        end_date=args.end_date,
        z_score=robust_z_score,
        period='test'
    )
    test_loader = DataLoader(df_test, shuffle=False, drop_last=False, collate_fn=custom_collate)

    matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_sector_industry_matrix.npz')
    adj_matrix, ticker_list = matrix['adj_matrix'][:11, :, :], matrix['tickers']
    adj_matrix = preprocess_matrix(adj_matrix)

    trainer(args, train_loader, valid_loader, test_loader, model_config, adj_matrix, ticker_list, metrics_path, device)

    print(f"Training completed. Metrics saved to {metrics_path}")

