import numpy as np
import pandas as pd
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
        'close': df['close'].max(),
    }
    return max_values


def dataset_normalization(dataset, max_values):
    dataset = dataset.copy()
    dataset.loc[:, 'close'] = dataset['close'] / max_values['close']
    dataset.loc[:, 'ma5'] = dataset['ma5'] / max_values['close']
    dataset.loc[:, 'ma10'] = dataset['ma10'] / max_values['close']
    dataset.loc[:, 'ma20'] = dataset['ma20'] / max_values['close']
    dataset.loc[:, 'ma30'] = dataset['ma30'] / max_values['close']
    return dataset


def load_stock_features(args):
    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    constituents = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]
    dataset = dataset[['instrument', 'date', 'close']]
    dataset = dataset.sort_values(by=['date', 'instrument'])

    all_dates = dataset['date'].drop_duplicates().sort_values()
    #all_dates = all_dates[(all_dates >= args.start_date) & (all_dates <= args.end_date)]

    stock_features = []
    return_ratios = []
    base_price = []
    masks = []
    available_tickers = []

    for ticker, df in tqdm(dataset.groupby('instrument')):
        df = pd.DataFrame({'date': all_dates}).merge(df, on='date', how='left').ffill().infer_objects()

        if ticker == 'SBNY.US':
            continue

        if df['date'].min() <= args.start_date and df['date'].max() >= args.end_date:

            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma30'] = df['close'].rolling(30).mean()

            mask = (df['date'] >= args.start_date) & (df['date'] <= args.end_date)
            df = df[mask]

            df = df.bfill().infer_objects()

            return_ratio = (df['close'] - df['close'].shift(args.pred_len)) / df['close'].shift(args.pred_len).fillna(0)
            mask = np.where(df['close'] < 1e-8, 0.0, 1.0)

            max_values = find_max_values(df[(df['date'] >= args.start_date) & (df['date'] <= args.end_train_date)])
            df = dataset_normalization(df, max_values)
            df = df.drop(columns=['instrument', 'date'])

            df = df[['ma5', 'ma10', 'ma20', 'ma30', 'close']]

            stock_features.append(df)
            base_price.append(df['close'] * 0.2)
            masks.append(mask)
            return_ratios.append(return_ratio)
            available_tickers.append(ticker)

    stock_features = np.array(stock_features)
    masks = np.array(masks)
    return_ratios = np.array(return_ratios)
    base_price = np.array(base_price)

    all_dates = all_dates[(all_dates >= args.start_date) & (all_dates <= args.end_date)].reset_index(drop=True)
    valid_index = pd.to_datetime(all_dates[all_dates > args.end_train_date]).idxmin() - args.history_len - args.pred_len + 1
    test_index = pd.to_datetime(all_dates[all_dates > args.end_valid_date]).idxmin() - args.history_len - args.pred_len + 1

    return stock_features, masks, return_ratios, base_price, valid_index, test_index, all_dates.to_list(), available_tickers


def load_inci_matrix(tickers, args):
    # inci_matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_inc_matrix.npz')
    inci_matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_inc_matrix_sect_ind.npz')
    matr = inci_matrix['inc_matrix']
    matr_tickers = inci_matrix['tickers']

    matr_tickers = [mt in tickers for mt in matr_tickers]
    matr = matr[matr_tickers]

    # matr = generate_safe_incidence_matrix(
    #     num_nodes=matr.shape[0],
    #     num_hyperedges=2000,
    #     min_edges_per_node=3,
    #     min_nodes_per_edge=5
    # )

    return matr


def save_dates(dates, tickers, start_index, pred_len, seq_len):
    last_seq_date = dates[start_index + seq_len - 1: - pred_len]
    dates_gt = dates[start_index + seq_len + pred_len - 1:]
    tickers = [tickers] * len(dates_gt)
    return dates_gt, last_seq_date, tickers

#
# def generate_safe_incidence_matrix(num_nodes=500, num_hyperedges=2000, min_edges_per_node=3, min_nodes_per_edge=5):
#     # matrice inizializzata a zero
#     mat = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)
#
#     # garantisci che ogni nodo partecipi ad almeno 'min_edges_per_node' iperedge
#     for node in range(num_nodes):
#         edges = np.random.choice(
#             num_hyperedges,
#             size=min_edges_per_node,
#             replace=False
#         )
#         mat[node, edges] = 1
#
#     # garantisci che ogni iperedge abbia almeno 'min_nodes_per_edge' nodi
#     for edge in range(num_hyperedges):
#         nodes = np.random.choice(
#             num_nodes,
#             size=min_nodes_per_edge,
#             replace=False
#         )
#         mat[nodes, edge] = 1
#
#     return mat
