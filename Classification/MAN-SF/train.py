import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from torch import optim

from load_data import encode_news, load_prices, load_adj, build_news_array, filter_data_by_available_news_tickers, \
    train_valid_test_split, windows_extraction, adj_to_dense_matrix, filter_nan_windows, filter_data_by_available_news
from model.model import GAT


def get_metrics(preds, labels):
    preds = np.argmax(preds, axis=2).reshape(-1)
    labels = np.concatenate(labels, axis=0).reshape(-1)
    mask = ~np.isnan(labels)
    preds = preds[mask]
    labels = labels[mask]

    metrics = {
        'F1': f1_score(labels, preds, average='macro'),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall': recall_score(labels, preds),
        'MCC': matthews_corrcoef(labels, preds)
    }
    return metrics


def train_model(model, prices, news):
    model.train()

    i = np.random.randint(prices.shape[1])
    text = news[:, i].to(device)
    price = prices[:, i, :, :-2].to(device)
    labels = prices[:, i, -1, -2].long().to(device)

    output = model(text, price, adj)

    return output, labels


def eval_model(model, prices, news):
    model.eval()

    preds = []

    for i in range(prices.shape[1]):
        text = news[:, i].to(device)
        price = prices[:, i, :, :-2].to(device)

        output = model(text, price, adj)
        preds.append(output.detach().cpu().numpy())

    labels = prices[:, :, -1, -1].cpu().numpy()
    labels = labels.swapaxes(0, 1)

    metrics = get_metrics(preds, labels)

    return metrics, preds, labels


if __name__=='__main__':

    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='sp500', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='MAN-SF', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=10, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2019-01-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--start_valid_date', type=str, default='2022-01-01', help='Start date for validation set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--n_news', type=int, default=5, help='Number of news considered for each trading day')

    args.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    args.add_argument('--seed', type=int, default=42, help='Random seed.')
    args.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    args.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    args.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    args.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    args.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    args.add_argument('--dropout', type=float, default=0.38, help='Dropout rate (1 - keep probability).')
    args.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    args.add_argument('--early_stopping', type=int, default=50, help='Early stopping criteria.')

    args = args.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    os.makedirs(f'{args.data_path}_news/{args.universe}/', exist_ok=True)

    metrics_path = f'./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    ######################
    # DATA PREPROCESSING #
    ######################

    prices_array, tickers, dates = load_prices(args)

    if len(tickers) == 0:
        raise ValueError("No tickers found in the specified universe and date range.")

    adj, ordered_tickers = load_adj(tickers, args)
    news_dict = encode_news(tickers, dates, args)
    news_list, news_tickers = build_news_array(news_dict, dates, tickers, args)

    prices_array, adj, tickers, ordered_tickers = filter_data_by_available_news_tickers(news_tickers, tickers, prices_array, adj, ordered_tickers)

    splitted_prices, splitted_news, dates = train_valid_test_split(prices_array, news_list, tickers, dates, args)
    splitted_prices, splitted_news, last_dates, pred_dates = windows_extraction(splitted_prices, splitted_news, dates, args)
    splitted_prices, splitted_news, last_dates, pred_dates, tickers, ordered_tickers, adj = filter_data_by_available_news(
        splitted_prices, splitted_news, last_dates, pred_dates, tickers, ordered_tickers, adj)

    splitted_prices, splitted_news = filter_nan_windows(splitted_prices, splitted_news)

    if splitted_news[0].shape[1] == 0 or splitted_news[1].shape[1] == 0 or splitted_news[2].shape[1] == 0:
        raise ValueError("No news data available for training/validation/test after preprocessing.")

    ################################
    # MODEL INIZIALIZATION & TRAIN #
    ################################

    model = GAT(
        nfeat=64,
        nhid=args.hidden,
        nclass=2,
        dropout=args.dropout,
        nheads=args.nb_heads,
        alpha=args.alpha,
        stock_num=len(tickers),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.00, 1.00]).cuda())

    adj = adj_to_dense_matrix(adj)
    adj = adj.to(device)

    best_mcc = None
    stop = 0

    for epoch in range(args.epochs):
        print(f'[Epoch {epoch}] Training...')

        output, labels = train_model(model, splitted_prices[0], splitted_news[0])
        loss_train = cross_entropy(output, labels)

        print(f'Train Loss: {loss_train}')

        valid_metrics, _, _ = eval_model(model, splitted_prices[1], splitted_news[1])
        print(f'Valid metrics: F1 {valid_metrics["F1"]}, Acc {valid_metrics["Accuracy"]}, MCC {valid_metrics["MCC"]}')

        if best_mcc is None or best_mcc < valid_metrics['MCC']:
            best_mcc = valid_metrics['MCC']
            stop = 0

            valid_metrics = {k: float(v) for k, v in valid_metrics.items()}

            test_metrics, preds, test_labels = eval_model(model, splitted_prices[2], splitted_news[2])
            test_metrics = {k: float(v) for k, v in test_metrics.items()}

            test_labels = [np.expand_dims(test_labels[i], axis=1) for i in range(test_labels.shape[0])]

            pred_dates = sorted({item for sublist in pred_dates.values() for item in sublist})
            last_dates = sorted({item for sublist in last_dates.values() for item in sublist})

            results = {
                'metrics': test_metrics,
                'preds': preds,
                'labels': test_labels,
                'pred_date': pred_dates,
                'last_date': last_dates,
                'tickers': [tickers] * len(preds)
            }

            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(valid_metrics, f, indent=4)

            with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(test_metrics, f, indent=4)

            with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
                pickle.dump(results, f)

            print('Test Metrics:\n', test_metrics)

        else:
            stop += 1

        if stop >= args.early_stopping:
            break

        loss_train.backward()
        optimizer.step()




