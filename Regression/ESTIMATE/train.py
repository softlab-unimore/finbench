import importlib.util
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd

def filter_constituents_by_date(constituents: pd.DataFrame, test_start_date: str) -> pd.DataFrame:
    """
    Filters a DataFrame of constituents to include only those active on a given test start date

    Args:
        constituents: DataFrame with 'StartDate' and 'EndDate' columns
        test_start_date: The date to check for active constituents (YYYY-MM-DD format)

    Returns:
        A new DataFrame containing only the active constituents.
    """
    if not all(col in constituents.columns for col in ['StartDate', 'EndDate']):
        raise ValueError('The "constituents" DataFrame must contain StartDate and EndDate columns')

    start_dates = pd.to_datetime(constituents['StartDate'])
    end_dates = pd.to_datetime(constituents['EndDate'])
    test_start = pd.to_datetime(test_start_date)

    # Fill missing dates with logical defaults: very early for start dates, a future date for end dates
    start_dates = start_dates.fillna(pd.Timestamp.min)
    end_dates = end_dates.fillna(pd.Timestamp.max)

    is_active = (start_dates < test_start) & (end_dates >= test_start)

    return constituents[is_active].copy()

def parse_args():
    args = ArgumentParser()

    # Preprocess dataset
    args.add_argument('--universe', type=str, default='sxxp')
    args.add_argument('--data_path', type=str, default='./data')
    args.add_argument('--train', action='store_true', default=False)
    args.add_argument('--test', action='store_true', default=False)
    args.add_argument('--job_id', type=int, default=0)

    # Project
    args.add_argument('--run_name', default='sxxp_estimate')
    args.add_argument('--model', type=str, default='estimate')
    args.add_argument('--model_path_folder', type=str, default='model/estimate/framework/')

    # Data (flat)
    args.add_argument('--symbols', nargs='*', default=[])
    args.add_argument('--start_train', default='2018-10-01')
    args.add_argument('--end_train', default='2018-12-31')
    args.add_argument('--start_valid', default='2019-06-01')
    args.add_argument('--end_valid', default='2019-12-31')
    args.add_argument('--start_test', default='2020-01-01')
    args.add_argument('--end_test', default='2020-12-31')
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
    args.add_argument('--save_best', action='store_true', default=True)
    args.add_argument('--lr_decay', type=float, default=0.005)
    args.add_argument('--eval_iter', type=int, default=10)
    args.add_argument('--dropout', type=float, default=0.4)
    args.add_argument('--verify_threshold', type=float, default=0.08)
    args.add_argument('--seed', type=int, default=42)

    # Backtest
    args.add_argument('--config_path', default='backtest/config/normal.yaml')

    args = args.parse_args()
    return args

def select_valid_ticker(df, start_date, start_test, end_date, seq_len=20):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df_test = df[(df['date'] >= start_test) & (df['date'] <= end_date)]
    df_test = df_test.groupby('instrument').filter(lambda x: len(x) > seq_len)

    tickers_train = set(df_train['instrument'].unique())
    tickers_test = set(df_test['instrument'].unique())
    valid_tickers = tickers_train.intersection(tickers_test)

    df = df[df['instrument'].isin(valid_tickers)]
    return df, valid_tickers

def preprocess_dataset(args):
    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test)

    df = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    df = df[df['instrument'].isin(tickers['EODHD'].tolist())]

    df, valid_tickers = select_valid_ticker(df, args.start_train, args.start_test, args.end_test, args.history_window)
    df = df[['open', 'close', 'high', 'low', 'volume', 'instrument', 'date', 'adj_close']]
    dates = df['date'].drop_duplicates().sort_values()

    ticker_dict = {}
    final_valid_tickers = []

    for ticker, sub_df in df.groupby('instrument'):
        sub_df = sub_df.sort_values('date')
        train_mask = ((sub_df['date'] >= args.start_train) & (sub_df['date'] <= args.end_train))
        n_train_obs = train_mask.sum()
        if n_train_obs < args.n_step_ahead + args.history_window:
            continue

        sub_df = pd.merge(pd.DataFrame({'date': dates}), sub_df, on='date', how='left').ffill().infer_objects()
        ticker_dict[ticker] = (sub_df.set_index('date').drop(columns=['instrument']))
        final_valid_tickers.append(ticker)

    np.save(f'{args.data_path}_estimate/preprocess/baseline_data.npy', ticker_dict)
    return final_valid_tickers

def exctract_sector_info(args, tickers):
    info = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_info.csv')
    info = info[info['instrument'].isin(tickers)]
    info = info[['instrument', 'Name', 'GicSector']]
    info = info.rename({'instrument': 'Symbol', 'GicSector': 'Sector'}, axis=1)
    info.to_csv(f'{args.data_path}_estimate/preprocess/ticker_info.csv', index=False)


def run_script(script_path, args):
    sys.argv = [script_path]
    for k, v in vars(args).items():
        if isinstance(v, bool):
            if v:
                sys.argv.append(f"--{k}")
        else:
            sys.argv.extend([f"--{k}", str(v)])

    spec = importlib.util.spec_from_file_location("__main__", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

if __name__=='__main__':

    args = parse_args()

    os.makedirs(f'{args.data_path}_estimate/preprocess/', exist_ok=True)
    valid_tickers = preprocess_dataset(args)
    exctract_sector_info(args, valid_tickers)

    delattr(args, 'data_path')

    run_script('execute.py', args)
