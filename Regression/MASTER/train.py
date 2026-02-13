import json
import os
import pickle
from argparse import ArgumentParser
import time

import pandas as pd
import torch

from load_dataset import RobustZScoreNormalization, CSVDataset
from master import MASTERModel
from utils import filter_constituents_by_date


def save_datasets(dl_train, dl_valid, dl_test, universe, seq_len, pred_len):
    with open(f'./checkpoints/{universe}_dl_train_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_train, f)

    with open(f'./checkpoints/{universe}_dl_valid_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_valid, f)

    with open(f'./checkpoints/{universe}_dl_test_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_test, f)

def open_datasets(universe, seq_len, pred_len):
    with open(f'./checkpoints/{universe}_dl_train_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_train = pickle.load(f)

    with open(f'./checkpoints/{universe}_dl_valid_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_valid = pickle.load(f)

    with open(f'./checkpoints/{universe}_dl_test_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_test = pickle.load(f)

    return dl_train, dl_valid, dl_test

def create_saving_path(args):
    model_save_path = f"./model_params/{args.universe}/{args.model_name}"
    metrics_path = f"./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split('-')[0]}"
    log_dir = f"./logs/{args.model_name}/{args.seed}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path, exist_ok=True)

    return model_save_path, metrics_path, log_dir


def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df

def extract_labels(df):
    df_close = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")[['date', 'instrument', 'adj_close']]
    df_close = df_close.sort_values(['instrument', 'date'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-args.pred_len) - x) / x)
    df = df.merge(df_close[['date', 'instrument', 'Label']], on=['date', 'instrument'], how='left')
    return df


if __name__ == '__main__':

    args = ArgumentParser()

    # Dataset Parameters
    args.add_argument('--model_name', type=str, default='MASTER')
    args.add_argument('--universe', type=str, default='sx5e')
    args.add_argument('--data_path', type=str, default='../../Evaluation/data')
    args.add_argument('--data_preprocessing', action='store_true', default=False)
    args.add_argument('--nation', type=str, default=None)

    # Prediction Parameters
    args.add_argument('--seq_len', type=int, default=8)
    args.add_argument('--pred_len', type=int, default=10)
    args.add_argument('--start_date', type=str, default='2010-01-01')
    args.add_argument('--end_train_date', type=str, default='2010-01-31')
    args.add_argument('--start_valid_date', type=str, default='2010-02-01')
    args.add_argument('--end_valid_date', type=str, default='2010-02-28')
    args.add_argument('--start_test_date', type=str, default='2010-03-01')
    args.add_argument('--end_date', type=str, default='2010-03-31')

    # Model Hyperparameters
    args.add_argument('--n_epoch', type=int, default=40)
    args.add_argument('--d_model', type=int, default=256)
    args.add_argument('--lr', type=float, default=1e-5)
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--t_nhead', type=int, default=4)
    args.add_argument('--s_nhead', type=int, default=2)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--gate_input_start_index', type=int, default=157)
    args.add_argument('--gate_input_end_index', type=int, default=None)
    args.add_argument('--beta', type=int, default=5)
    args.add_argument('--train_stop_loss_thred', type=float, default=0.95)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--num_workers', type=int, default=4)

    args = args.parse_args()

    # Mapping universe -> default nation
    universe_to_nation = {
        'sxxp': 'eu',
        'sp500': 'us',
        'nasdaq100': 'us',
        'dji': 'us',
        'sx5e': 'eu',
    }

    if args.nation is None:
        args.nation = universe_to_nation.get(args.universe.lower(), 'us')  # fallback 'us'

    # Setting args.gate_input_end_index based on the number of features in market and alpha CSVs
    market_csv = f"{args.data_path}/{args.universe}/{args.nation}_market.csv"
    shape_m = pd.read_csv(market_csv).shape[1]
    alpha_csv = f"{args.data_path}/{args.universe}/{args.universe}_alpha158.csv"
    shape_a = pd.read_csv(alpha_csv).shape[1]
    args.gate_input_end_index = shape_m + shape_a - 3


    print(args)

    model_save_path, metrics_path, log_dir = create_saving_path(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data_preprocessing:
        os.makedirs('./checkpoints/', exist_ok=True)

        df_alpha = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_alpha158.csv")
        tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date)
        df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

        market_index = pd.read_csv(f'{args.data_path}/{args.nation}_market.csv')
        df_alpha = pd.merge(df_alpha, market_index, how='left', on='date')

        df_alpha = extract_labels(df_alpha)
        df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

        robust_z_score = RobustZScoreNormalization(df_alpha[(df_alpha['date'] >= args.start_date) & (df_alpha['date'] <= args.end_train_date)])

        dl_train = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_date,
            end_date=args.end_train_date,
            z_score=robust_z_score
        )

        dl_valid = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_valid_date,
            end_date=args.end_valid_date,
            z_score=robust_z_score,
            period='valid'
        )

        dl_test = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_test_date,
            end_date=args.end_date,
            z_score=robust_z_score,
            period='test'
        )

        # save_datasets(dl_train, dl_valid, dl_test, args.universe, args.seq_len, args.pred_len)
    else:
        dl_train, dl_valid, dl_test = open_datasets(args.universe, args.seq_len, args.pred_len)


    model = MASTERModel(
        d_feat=args.gate_input_start_index, d_model=args.d_model, t_nhead=args.t_nhead, s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout, S_dropout_rate=args.dropout, beta=args.beta, n_epochs=args.n_epoch, lr=args.lr,
        gate_input_end_index=args.gate_input_end_index, gate_input_start_index=args.gate_input_start_index,
        save_path=f'{model_save_path}_{args.seed}', GPU=args.gpu, train_stop_loss_thred=args.train_stop_loss_thred
    )

    start = time.time()

    model.fit(dl_train, metrics_path, args, dl_valid, args.num_workers)

    print("Model Trained.")
    predictions, labels, metrics = model.predict(dl_test, args.num_workers)

    running_time = time.time() - start

    print('Seed: {:d} time cost : {:.2f} sec'.format(args.seed, running_time))

    metrics = {k: float(v) for k, v in metrics.items()}
    print(metrics)

    results = {
        'metrics': metrics,
        'preds': predictions,
        'labels': labels,
        'pred_date': dl_test.output_dates,
        'last_date': dl_test.input_dates,
        'tickers': dl_test.tickers_to_date
    }

    with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(metrics, f)