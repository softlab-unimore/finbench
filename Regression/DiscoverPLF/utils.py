import torch
import pandas as pd
import numpy as np


def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)


def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


def cal_convariance(x, y): # the 2nd dimension of x and y are the same
    e_x = torch.mean(x, dim = 1).reshape(-1, 1)
    e_y = torch.mean(y, dim = 1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend*y_extend, dim = 2)
    return e_xy - e_x_e_y


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic


def cal_beta(epoch):
    if epoch<20: return 1.
    elif epoch<30: return 5.
    else: return 10.


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


def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df

