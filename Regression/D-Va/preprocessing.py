import glob
import os

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

    is_active = (start_dates < test_start) & (end_dates >= test_start)

    return constituents[is_active].copy()


def load_dataset(start_date, end_date, train_end_date, val_end_date, start_test_date, path, universe, job_id, batch_size, pred_len, seq_len):
    data = pd.read_csv(f'{path}/{universe}.csv')
    constituents = pd.read_csv(f'{path}/{universe}_constituents.csv')
    tickers = filter_constituents_by_date(constituents, start_test_date)['EODHD'].tolist()
    data = data[data['volume'] > 0]

    available_tickers = []

    if os.path.exists(f'{path}/preprocessed_{job_id}'):
        files = glob.glob(os.path.join(f'{path}/preprocessed_{job_id}', '*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs(path)

    for ticker in tqdm(tickers, desc='Processing tickers'):
        df = data[data['instrument'] == ticker]
        df = df[['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']]

        df['adj_open'] = df['adj_open'] / df['adj_close']
        df['adj_high'] = df['adj_high'] / df['adj_close']
        df['adj_low'] = df['adj_low'] / df['adj_close']
        df['adj_abs_ret'] = df['adj_close'] - df['adj_close'].shift(1)
        df['target'] = df['adj_close'] / df['adj_close'].shift(1)
        df = df.drop(['adj_close'], axis=1)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.dropna(axis=0)

        train_df = df[df['date'] <= train_end_date]
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)]
        test_df = df[df['date'] > val_end_date]

        if len(df) > 0 and len(train_df) >= 251 and len(val_df) >= len(train_df) * 0.1 and len(test_df) - pred_len - seq_len >= batch_size:
            df.to_csv(f'{path}/preprocessed_{job_id}/{ticker.replace(".","_")}.csv', index=False)
            available_tickers.append(ticker)

    return available_tickers


