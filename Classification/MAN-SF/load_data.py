import json
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy
import torch
from dateutil import parser
from tqdm import tqdm

import embedder
from multiprocessing import Pool, cpu_count

import pandas as pd


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


def load_prices(args):
    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    constituents = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_constituents.csv')
    tickers = filter_constituents_by_date(constituents, args.start_test_date)['EODHD'].tolist()
    dataset = dataset[dataset['instrument'].isin(tickers)]

    dataset = dataset[(dataset['date'] >= args.start_date) & (dataset['date'] <= args.end_date)]
    dates = dataset['date'].drop_duplicates().sort_values()

    tickers = dataset['instrument'].drop_duplicates().sort_values().tolist()
    dataset = dataset[['date', 'instrument', 'high', 'low', 'adj_close']]

    total_data = []
    valid_tickers = []
    valid_dates = {}

    print('Extract prices...')

    for ticker in tqdm(tickers):
        df = dataset[dataset['instrument'] == ticker].copy()
        df = df.drop(columns=['instrument'])
        df = df.set_index('date')

        if not dates.isin(df.index).all():
            continue

        df['high'] = df['high'] / df['adj_close'].shift(1)
        df['low'] = df['low'] / df['adj_close'].shift(1)
        df['return'] = (df['adj_close'].shift(-args.pred_len) - df['adj_close']) / df['adj_close']
        df['label'] = np.where(df['return'] >= 0.0055, 1, np.where(df['return'] <= -0.005, 0, np.nan))
        df['label_test'] = np.where(df['return'] > 0, 1, 0)
        df = df.drop(columns=['return'])

        total_data.append(df.to_numpy())
        valid_tickers.append(ticker)
        valid_dates[ticker] = df.index.tolist()

    return total_data, valid_tickers, valid_dates


def process_news_file(args_tuple):
    filename, dir_path, args, batch_size, valid_dates = args_tuple

    ticker_name = filename.replace(".json", "")

    with open(os.path.join(dir_path, filename), 'r') as f:
        news = json.load(f)

    texts = []
    dates = []

    for item in news:
        date = parser.parse(item["date"]).strftime("%Y-%m-%d")
        if date in valid_dates[ticker_name]:
            dates.append(date)
            texts.append(item["title"] + "\n" + item["content"])

    if len(texts) == 0:
        print(f"[Worker {os.getpid()}] No news in date range for {ticker_name}")
        return ticker_name, None, None

    print(f"[Worker {os.getpid()}] Embedding {ticker_name} ({len(texts)} items)")
    embeddings = embedder.embed_news_async(texts, embedder._global_embed, embedder._global_sess, batch_size)

    return ticker_name, dates, embeddings


def encode_news(tickers, dates, args, batch_size=512):

    dir_path = f'{args.data_path}/{args.universe}/news'

    start_year = args.start_date.split('-')[0]
    end_year = args.end_date.split('-')[0]
    file_emb = f'{args.data_path}_news/{args.universe}/news_embeddings_y{start_year}_y{end_year}.pkl'

    if os.path.exists(file_emb):
        print(f"[load_news] Loading precomputed news embeddings")
        with open(file_emb, 'rb') as f:
             return pickle.load(f)

    jobs = [
        (filename, dir_path, args, batch_size, dates)
        for filename in os.listdir(dir_path)
        if filename.endswith(".json") and filename.replace(".json", "") in tickers
    ]

    num_workers = min(cpu_count(), 10)
    print(f"[load_news] Using {num_workers} workers…")

    news_dict = {}
    with Pool(num_workers, initializer=embedder.worker_initializer, maxtasksperchild=5) as pool:
        for ticker_name, dates, embeddings in tqdm(pool.imap_unordered(process_news_file, jobs), total=len(jobs)):
            news_dict[ticker_name] = {
                "dates": dates,
                "file": embeddings
            }

    with open(file_emb, 'wb') as f:
        pickle.dump(news_dict, f)

    return news_dict


def get_news_for_day(ticker_dict, date_index, dates, num_news, num_features):

    collected = []

    while len(collected) < num_news and date_index >= 0:
        d = dates[date_index]
        if d in ticker_dict:
            collected.extend(ticker_dict[d])
        date_index -= 1

    collected = collected[:num_news]

    while len(collected) < num_news:
        collected.append(np.zeros(num_features))

    return np.stack(collected)


def build_news_array(news_dict, dates, tickers, args, num_news=5):

    start_year = args.start_date.split('-')[0]
    end_year = args.end_date.split('-')[0]
    filename = f'{args.data_path}_news/{args.universe}/news_list_y{start_year}_y{end_year}.pkl'

    sample_ticker = tickers[0]
    sample_arr = np.array(news_dict[sample_ticker]["file"][0])
    num_features = sample_arr.shape[-1]

    grouped = {}

    for ticker in tickers:
        if ticker in news_dict.keys():
            d = news_dict[ticker]
            if d['file'] is not None and d['dates'] is not None:
                td = defaultdict(list)
                for date, arr in zip(d["dates"], d["file"]):
                    td[date].append(np.array(arr))
                grouped[ticker] = td

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            tensors = pickle.load(f)
        return tensors, grouped.keys()

    print("Create news array....")

    tensors = [np.zeros((len(dates[ticker]), num_news, num_features)) for ticker in grouped.keys()]

    for i, ticker in enumerate(grouped.keys()):
        ticker_dict = grouped[ticker]
        ticker_dates = dates[ticker]
        T = tensors[i]

        for j, date in enumerate(ticker_dates):
            T[j] = get_news_for_day(
                ticker_dict=ticker_dict,
                date_index=j,
                dates=ticker_dates,
                num_news=num_news,
                num_features=num_features
            )

    with open(filename, 'wb') as f:
        pickle.dump(tensors, f)

    return tensors, grouped.keys()


def load_adj(tickers, args):
    matr = np.load(f'{args.data_path}/{args.universe}/{args.universe}_wikidata_matrix.npz')
    adj = matr['adj_matrix']
    dates = matr['dates']
    all_tickers = matr['tickers']

    dates = pd.to_datetime(dates)
    target = datetime.strptime(args.end_train_date, "%Y-%m-%d")
    target_month = target.month
    target_year = target.year

    mask_date = (dates.year == target_year) & (dates.month == target_month)
    adj = adj[mask_date].squeeze()
    adj = np.max(adj, axis=0)

    mask_tickers = [tick in tickers for tick in all_tickers]
    adj = adj[mask_tickers]
    adj = adj[:, mask_tickers]
    ordered_tickers = all_tickers[mask_tickers]

    return adj, ordered_tickers


def normalize_adj(mx):
    row_sum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = scipy.sparse.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def adj_to_dense_matrix(graph):
    adj = scipy.sparse.coo_matrix(graph, dtype=np.float32)
    adj = normalize_adj(adj + scipy.sparse.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def filter_adj_matrix(adj, adj_tickers, tickers):
    mask_adj = [t in tickers for t in adj_tickers]
    adj_tickers = [t for t, m in zip(adj_tickers, mask_adj) if m]
    adj = adj[mask_adj]
    adj = adj[:, mask_adj]
    return adj, adj_tickers


def filter_data_by_available_news_tickers(news_tickers, tickers, prices, adj, adj_tickers):
    mask = [tick in news_tickers for tick in tickers]
    prices = [p for p, m in zip(prices, mask) if m]
    adj, adj_tickers = filter_adj_matrix(adj, adj_tickers, tickers)

    return prices, adj, news_tickers, adj_tickers


def train_valid_test_split(prices_array, news_list, tickers, dates, args):

    train_prices = []
    valid_prices = []
    test_prices = []

    train_news = []
    valid_news = []
    test_news = []

    train_dates = {}
    valid_dates = {}
    test_dates = {}

    for i, tick in enumerate(tickers):
        end_train_idx = next((i for i, d in enumerate(dates[tick]) if d > args.end_train_date), None)
        start_valid_idx = end_train_idx - args.seq_len - args.pred_len + 1
        end_valid_idx = next((i for i, d in enumerate(dates[tick]) if d > args.end_valid_date), None)
        start_test_idx = end_valid_idx - args.seq_len - args.pred_len + 1

        train_prices.append(prices_array[i][:end_train_idx])
        train_news.append(news_list[i][:end_train_idx])
        train_dates[tick] = dates[tick][:end_train_idx]

        valid_prices.append(prices_array[i][start_valid_idx:end_valid_idx])
        valid_news.append(news_list[i][start_valid_idx:end_valid_idx])
        valid_dates[tick] = dates[tick][start_valid_idx:end_valid_idx]

        test_prices.append(prices_array[i][start_test_idx:])
        test_news.append(news_list[i][start_test_idx:])
        test_dates[tick] = dates[tick][start_test_idx:]

    splitted_prices = [np.array(train_prices), np.array(valid_prices), np.array(test_prices)]
    splitted_news = [np.array(train_news), np.array(valid_news), np.array(test_news)]
    dates = [train_dates, valid_dates, test_dates]
    return splitted_prices, splitted_news, dates


def windows_extraction(splitted_prices, splitted_news, dates, args):

    last_dates = None
    pred_dates = None

    for i, (p, n, d) in enumerate(zip(splitted_prices, splitted_news, dates)):
        p = p[:, :-args.pred_len]
        n = n[:, :-args.pred_len]
        p = torch.from_numpy(p).float()
        n = torch.from_numpy(n).float()
        p = p.unfold(dimension=1, size=args.seq_len, step=1)
        n = n.unfold(dimension=1, size=args.seq_len, step=1)
        splitted_prices[i] = p.permute(0, 1, 3, 2)
        splitted_news[i] = n.permute(0, 1, 4, 2, 3)

        if i == 2:
            last_dates = {k: v[args.seq_len - 1: - args.pred_len] for k,v in d.items()}
            pred_dates = {k: v[args.seq_len + args.pred_len - 1:] for k,v in d.items()}

    return splitted_prices, splitted_news, last_dates, pred_dates


def filter_data_by_available_news(splitted_prices, splitted_news, last_dates, pred_dates, tickers, adj_tickers, adj):

    ticker_without_news = []
    for i, (p, n) in enumerate(zip(splitted_prices, splitted_news)):
        mask = (n == 0).all(dim=(1, 2, 3, 4))
        ticker_without_news.append(mask)

    ticker_without_news = ~torch.stack(ticker_without_news).any(dim=0)

    for i, (p, n) in enumerate(zip(splitted_prices, splitted_news)):
        p = p[ticker_without_news]
        n = n[ticker_without_news]

        mask = ~(n == 0).all(dim=(0, 2, 3, 4))
        splitted_prices[i] = p[:, mask]
        splitted_news[i] = n[:, mask]

        if i == 2:
            tickers = [t for t, m in zip(tickers, ticker_without_news) if m]
            adj, adj_tickers = filter_adj_matrix(adj, adj_tickers, tickers)
            last_dates = {k: v for (k, v), m in zip(last_dates.items(), ticker_without_news) if m}
            last_dates = {k: [d for d, m in zip(v, mask) if m] for k, v in last_dates.items()}
            pred_dates = {k: v for (k, v), m in zip(pred_dates.items(), ticker_without_news) if m}
            pred_dates = {k: [d for d, m in zip(v, mask) if m] for k, v in pred_dates.items()}

    return splitted_prices, splitted_news, last_dates, pred_dates, tickers, adj_tickers, adj


def filter_nan_windows(splitted_prices, splitted_news):

    labels = splitted_prices[0][:, :, -1, -2]
    mask = ~torch.isnan(labels)
    mask = mask.all(dim=0)
    splitted_prices[0] = splitted_prices[0][:, mask]
    splitted_news[0] = splitted_news[0][:, mask]

    return splitted_prices, splitted_news





