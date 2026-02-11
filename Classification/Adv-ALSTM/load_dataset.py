import os
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm


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

    is_active = (start_dates < test_start) & (end_dates > test_start)

    return constituents[is_active].copy()


def extract_data_divided_per_ticker(data, raw_data, start_date, end_date, tickers, pred_len):
    number_of_ticker_available = 0

    data_ticker = []
    available_tickers = []

    for ticker in tqdm(tickers):
        df = data[data['instrument'] == ticker].copy()
        raw_df = raw_data[raw_data['instrument'] == ticker].copy()
        raw_df = raw_df[['date', 'adj_close']]
        df = df.merge(raw_df, how='inner', on='date')

        df['Label'] = ((df['adj_close'].shift(-pred_len) / df['adj_close']) - 1) * 100
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.dropna()
        df = df.drop(['instrument'], axis=1)

        if len(df) > 0:
            number_of_ticker_available = number_of_ticker_available + 1
            data_ticker.append(df)
            available_tickers.append(ticker)

    print(f"Number of tickers available: {number_of_ticker_available} / {len(tickers)}")
    return data_ticker, available_tickers


def extract_labels(df, df_label, dates=None, test_dates=None, flag='train'):
    if flag == 'train':
        df_label = np.where(df_label >= 0.55, 1, np.where(df_label <= -0.5, 0, -1))
        mask_label = (df_label != -1).squeeze(1)
        mask_values = df[:, :, 1:].min(axis=(1,2)) > -123320 # from the original code
        mask = mask_label & mask_values
        df, df_label = df[mask], df_label[mask]
        if dates is not None:
            dates = dates[mask]
            test_dates = test_dates[mask]
    else:
        df_label = np.where(df_label >= 0.5, 1, 0)
    return df, df_label, dates, test_dates


def train_val_test_split(df, end_train_date, end_valid_date, seq_len, pred_len):
    dates = pd.to_datetime(df['date'])

    # Extract training data
    df_train = df[df['date'] <= end_train_date]
    df_train_label = df_train.iloc[seq_len - 1:-pred_len, df_train.columns.get_loc('Label')].to_numpy().reshape(-1, 1)

    if len(df_train) > seq_len:
        df_train = np.array([df_train[i:i + seq_len] for i in range(len(df_train) - seq_len - pred_len + 1)])
        df_train, df_train_label, _, _ = extract_labels(df_train, df_train_label)

        if df[dates.dt.year == pd.to_datetime(end_train_date).year + 1].shape[0] > 0:
            end_train_date_idx = pd.to_datetime(df[dates.dt.year == pd.to_datetime(end_train_date).year + 1]['date']).idxmin()
            start_valid_date = df.loc[end_train_date_idx - seq_len - pred_len]['date']
        else:
            start_valid_date = end_train_date
    else:
        df_train = None
        df_train_label = None
        start_valid_date = end_train_date

    df_valid = df[(df['date'] > start_valid_date) & (df['date'] <= end_valid_date)]
    df_valid_label = df_valid.iloc[seq_len - 1:-pred_len, df_valid.columns.get_loc('Label')].to_numpy().reshape(-1, 1)

    if len(df_valid) > seq_len:
        df_valid = np.array([df_valid[i:i + seq_len] for i in range(len(df_valid) - seq_len - pred_len + 1)])
        df_valid, df_valid_label, _, _ = extract_labels(df_valid, df_valid_label, flag='valid')

        if df[dates.dt.year == pd.to_datetime(end_valid_date).year + 1].shape[0] > 0:
            start_test_date_idx = pd.to_datetime(df[dates.dt.year == pd.to_datetime(end_valid_date).year + 1]['date']).idxmin()
            start_test_date = df.loc[start_test_date_idx - seq_len - pred_len]['date']
        else:
            start_test_date = end_valid_date
    else:
        df_valid = None
        df_valid_label = None
        start_test_date = end_valid_date

    df_test = df[df['date'] > start_test_date]
    df_test_label = df_test.iloc[seq_len - 1:-pred_len, df_test.columns.get_loc('Label')].to_numpy().reshape(-1, 1)

    test_dates = df_test['date'].iloc[seq_len + pred_len - 1:].to_numpy()

    if len(df_test) > seq_len:
        df_test = np.array([df_test[i:i + seq_len] for i in range(len(df_test) - seq_len - pred_len + 1)])
        df_test, df_test_label, last_test_dates, test_dates = extract_labels(df_test, df_test_label, df_test[:, -1, 0], test_dates, flag='test')
    else:
        df_test = None
        df_test_label = None
        last_test_dates = None

    return df_train, df_train_label, df_valid, df_valid_label, df_test, df_test_label, last_test_dates, test_dates


def date_to_weekday_onehot(date_str):
    weekday = datetime.strptime(date_str, '%Y-%m-%d').weekday()
    if weekday > 4:
        return np.zeros(5, dtype=bool)
    onehot = np.zeros(5, dtype=bool)
    onehot[weekday] = True
    return onehot


def date_to_weekday(df):
    dates = df[:, :, 0].flatten()
    dates = np.array([date_to_weekday_onehot(date) for date in dates])
    dates = dates.reshape(df.shape[0], df.shape[1], 5)
    return dates


def load_dataset(data_path, universe, start_date, end_date, end_train_date, end_valid_date, seq_len=5, pred_len=1):
    
    constituents = pd.read_csv(f'{data_path}/constituents/eodhd/{universe}.csv')
    constituents = filter_constituents_by_date(constituents, end_valid_date)

    data = pd.read_csv(f'{data_path}/{universe}/{universe}_tech.csv')
    data = data[data['instrument'].isin(constituents['EODHD'].tolist())]

    tickers = data['instrument'].unique().tolist()
    data = data[['date', 'instrument', 'c_open', 'c_high', 'c_low', 'close_price_change', 'adj_close_price_change', 'z_d5',
         'z_d10', 'z_d15', 'z_d20', 'z_d25', 'z_d30']]

    raw_data = pd.read_csv(f'{data_path}/{universe}/{universe}.csv')
    raw_data = raw_data[['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]

    print("Loading data...")
    data_ticker, available_tickers = extract_data_divided_per_ticker(data, raw_data, start_date, end_date, tickers, pred_len)

    train_samples = []
    train_labels = []
    train_dates = []
    valid_samples = []
    valid_labels = []
    valid_dates = []
    test_samples = []
    test_labels = []
    test_dates = []
    test_tickers = []
    test_dates_str = []
    test_last_dates_str = []

    print("Splitting data into train, validation, and test sets...")

    for i in tqdm(range(len(data_ticker))):

        df_train, df_train_label, df_valid, df_valid_label, df_test, df_test_label, _, test_dates_ticker = train_val_test_split(data_ticker[i], end_train_date, end_valid_date, seq_len, pred_len)

        if df_train is not None:
            train_samples.append(df_train[:, :, 1:-1])
            train_labels.append(df_train_label)
            train_dates.append(date_to_weekday(df_train))

        if df_valid is not None:
            valid_samples.append(df_valid[:, :, 1:-1])
            valid_labels.append(df_valid_label)
            valid_dates.append(date_to_weekday(df_valid))

        if df_test is not None:
            test_samples.append(df_test[:, :, 1:-1])
            test_labels.append(df_test_label)
            test_dates.append(date_to_weekday(df_test))
            test_tickers.append([available_tickers[i]]*len(df_test))
            test_dates_str.append(test_dates_ticker)
            test_last_dates_str.append(df_test[:, -1, 0].tolist())

    train_samples = np.concatenate(train_samples, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_dates = np.concatenate(train_dates, axis=0)
    valid_samples = np.concatenate(valid_samples, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    valid_dates = np.concatenate(valid_dates, axis=0)
    test_samples = np.concatenate(test_samples, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_dates = np.concatenate(test_dates, axis=0)
    test_tickers = list(chain.from_iterable(test_tickers))
    test_dates_str = list(chain.from_iterable(test_dates_str))
    test_last_dates_str = list(chain.from_iterable(test_last_dates_str))

    return train_samples, train_dates, train_labels, valid_samples, valid_dates, valid_labels, test_samples, test_dates, test_labels, test_tickers, test_last_dates_str, test_dates_str

