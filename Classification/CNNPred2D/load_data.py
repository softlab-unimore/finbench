import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
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


def read_datasets(args):
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    df_close = df_close[['instrument', 'date', 'close']]

    df_tech = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_tech.csv')
    df_tech = df_tech[['instrument', 'date', 'vol', 'mom1', 'mom2', 'mom3', 'roc5', 'roc10', 'roc15', 'roc20', 'ema10',
                       'ema20', 'ema50', 'ema200']]

    df_cnnpred = pd.read_csv(f'{args.data_path}/cnnpred_market.csv')

    dataset = df_tech.merge(df_close, how='inner', on=['instrument', 'date'])
    dataset = dataset.merge(df_cnnpred, on='date', how='inner')

    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    dataset = dataset[(dataset['date'] >= args.start_date) & (dataset['date'] <= args.end_date)]
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    return dataset


def construct_data_warehouse_2d(args):
    dataset = read_datasets(args)

    df_grouped = dataset.groupby('instrument')
    data_warehouse = {inst: df.drop(columns=['instrument']).reset_index(drop=True) for inst, df in df_grouped}
    stocks = len(data_warehouse)
    features = len(dataset.columns) - 2

    return data_warehouse, stocks, features


def split_train_val_test_ticker(df, args):
    # Train data
    train_data = df[(df.index >= args.start_date) & (df.index <= args.end_train_date)]
    if len(train_data) > args.seq_len + args.pred_len:
        train_data = train_data[:-args.pred_len]
        train_labels = df[(df.index >= args.start_date) & (df.index <= args.end_train_date)]['label']
        train_labels = train_labels[args.seq_len + args.pred_len - 1:]
        train_data = scale(train_data.iloc[:, :-1])
    else:
        return None

    # Valid data
    valid_start_date = df.index[df.index >= args.start_valid_date].min()
    if isinstance(valid_start_date, str):
        valid_index = df.index.get_loc(valid_start_date) - args.seq_len - args.pred_len + 1
        valid_data = df[valid_index:]
        valid_data = valid_data[valid_data.index <= args.end_valid_date][:-args.pred_len]
        valid_labels = df[(df.index >= args.start_valid_date) & (df.index <= args.end_valid_date)]['label']
        valid_data = scale(valid_data.iloc[:, :-1])
    else:
        return None

    # Test data
    test_start_date = df.index[df.index >= args.start_test_date].min()
    if isinstance(test_start_date, str):
        test_index = df.index.get_loc(test_start_date) - args.seq_len - args.pred_len + 1
        test_data = df[test_index:]
        test_data = test_data[test_data.index <= args.end_date][:-args.pred_len]
        test_labels = df[(df.index >= args.start_test_date) & (df.index <= args.end_date)]['label']
        test_data = test_data.iloc[:, :-1]
    else:
        return None

    return [train_data, train_labels, valid_data, valid_labels, test_data, test_labels]


def extract_dates(test_data, test_labels, seq_len):
    return {
        'last_date': test_data[seq_len-1:].to_list(),
        'pred_date': test_labels.to_list()
    }


def split_train_val_test(data_warehouse, args):

    dates = {}
    splitted_data_warehouse = {}

    for ticker, df in tqdm(data_warehouse.items()):
        # Set index and label extraction
        # df = df[200:] # Because one of the features is a moving avg of 200 days
        df = df.fillna(0).set_index('date')
        df['label'] = (df['close'][args.pred_len:] / df['close'][:-args.pred_len].values).astype(int)

        splitted_dataset = split_train_val_test_ticker(df, args)

        if splitted_dataset is None:
            continue

        dates[ticker] = extract_dates(splitted_dataset[4].index, splitted_dataset[5].index, args.seq_len)
        splitted_data_warehouse[ticker] = splitted_dataset

    return splitted_data_warehouse, dates


def extract_sequences_2d(data_warehouse, seq_len, dates=None, idx=(0,1)):

    sequences = []
    labels = []
    tickers = []
    tot_pred_dates = []
    tot_last_dates = []

    for key, data in data_warehouse.items():
        features = data[idx[0]]
        for i in range(len(features) - seq_len + 1):
            seq = features[i:i + seq_len]
            sequences.append(seq)

        if dates is not None:
            tot_pred_dates.extend(dates[key]['pred_date'])
            tot_last_dates.extend(dates[key]['last_date'])

        labels.extend(data[idx[1]])
        tickers.extend([key] * (len(features) - seq_len + 1))

    sequences = np.expand_dims(sequences, axis=-1)
    return sequences, np.array(labels), tickers, tot_last_dates, tot_pred_dates

