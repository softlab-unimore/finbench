import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def filter_constituents_by_date(constituents: pd.DataFrame, test_start_date: str) -> pd.DataFrame:
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

def find_max_values(df):
    max_values = {
        'open': df['open'].max(),
        'close': df['close'].max(),
        'high': df['high'].max(),
        'low': df['low'].max(),
        'volume': df['volume'].max(),
        'ma_5': df['ma_5'].max(),
        'ma_10': df['ma_10'].max(),
        'ma_20': df['ma_20'].max(),
        'ma_30': df['ma_30'].max()
    }
    return max_values

def dataset_normalization(dataset, max_values):
    dataset = dataset.copy()
    dataset.loc[:, 'open'] = dataset['open'] / max_values['open']
    dataset.loc[:, 'close'] = dataset['close'] / max_values['close']
    dataset.loc[:, 'high'] = dataset['high'] / max_values['high']
    dataset.loc[:, 'low'] = dataset['low'] / max_values['low']
    dataset['volume'] = dataset['volume'].astype(float)
    dataset.loc[:, 'volume'] = dataset['volume'] / max_values['volume']
    dataset.loc[:, 'ma_5'] = dataset['ma_5'] / max_values['ma_5']
    dataset.loc[:, 'ma_10'] = dataset['ma_10'] / max_values['ma_10']
    dataset.loc[:, 'ma_20'] = dataset['ma_20'] / max_values['ma_20']
    dataset.loc[:, 'ma_30'] = dataset['ma_30'] / max_values['ma_30']
    return dataset


def load_data(args):
    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    constituents = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_constituents.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    dates = dataset['date'].drop_duplicates().sort_values()
    dates = dates[(dates >= args.start_date) & (dates <= args.end_date)]

    min_days = int(len(dates) * 0.98)

    valid_instruments = dataset[dataset['date'].isin(dates)].groupby('instrument')['date'].nunique()
    valid_instruments = valid_instruments[valid_instruments >= min_days].index.tolist()
    dataset = dataset[dataset['instrument'].isin(valid_instruments)]

    dataset = dataset.sort_values(['instrument', 'date'])
    dataset = dataset[['instrument', 'date', 'open', 'high', 'low', 'close', 'volume']]

    all_data = []
    gt_data = []

    for ticker in tqdm(valid_instruments):
        df = dataset[dataset['instrument'] == ticker].copy()

        # Exctract labels
        df['label'] = (df['close'].shift(-args.pred_len) - df['close']) / df['close']
        df['label'] = np.where(df['label'] >= 0.0055, 1, np.where(df['label'] < -0.005, 2, 0))

        # Filtering
        df = df[(df['date'] >= args.start_date) & (df['date'] <= args.end_date)]
        df['date'], dates = pd.to_datetime(df['date']), pd.to_datetime(dates)
        df = pd.merge_asof(pd.DataFrame({'date': dates}), df.sort_values('date'), on='date', direction='backward')

        for window in [5, 10, 20, 30]:
            df[f'ma_{window}'] = df['close'].transform(lambda x: x.rolling(window).mean())

        ma_cols = [f"ma_{w}" for w in [5, 10, 20, 30]]
        df_raw_ma = df[ma_cols].copy()

        # Normalization
        max_values = find_max_values(df[(df['date'] >= args.start_date) & (df['date'] <= args.end_train_date)])
        df = dataset_normalization(df, max_values)

        for col in ma_cols:
            mask = df_raw_ma[col].isna()
            df.loc[mask, col] = df_raw_ma[col].bfill()[mask]

        gt_data.append(df['label'])
        df = df.drop(columns=['instrument', 'date', 'label'])
        all_data.append(df)

    dates = dates.reset_index(drop=True)
    valid_index = pd.to_datetime(dates[dates > args.end_train_date]).idxmin() - args.seq_len - args.pred_len + 1
    test_index = pd.to_datetime(dates[dates > args.end_valid_date]).idxmin() - args.seq_len - args.pred_len + 1

    return np.array(all_data), np.array(gt_data), valid_index, test_index, dates, valid_instruments


def extract_dates(test_index, dates, args):
    pred_dates = dates[dates >= args.start_test_date]
    last_date = dates[test_index + args.seq_len - 1 : -args.pred_len]
    return pred_dates, last_date


def get_batch(eod_data, gt_data, offset, seq_len, pred_len):
    return eod_data[:, offset:offset + seq_len, :], gt_data[:, offset + seq_len + pred_len - 1]


def load_matrices(tickers, args):
    # Adj matrix: industry relations
    adj = np.load(f'{args.data_path}/{args.universe}/{args.universe}_sector_industry_matrix.npz')
    matr_adj, all_tickers = adj['adj_matrix'], adj['tickers']
    matr_adj = matr_adj[11:] #only industry features
    mask = np.isin(all_tickers, tickers)
    matr_adj = matr_adj[:, mask, :][:, :, mask]
    matr_adj = matr_adj.max(axis=0)

    # Fund matrix
    H = np.zeros((len(tickers), 62))

    return torch.Tensor(matr_adj), torch.Tensor(H)



















