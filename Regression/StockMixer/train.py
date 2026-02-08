import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch as torch
from load_datasets import load_dataset
from evaluator import evaluate_regression
from model import get_loss, StockMixer


def save_dates(dates, tickers, start_index, pred_len, seq_len):
    last_seq_date = dates[start_index + seq_len - 1: - pred_len]
    dates_gt = dates[start_index + seq_len + pred_len - 1:]
    tickers = [tickers] * len(dates_gt)
    return dates_gt, last_seq_date, tickers


def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = []
        cur_valid_gt = []
        cur_valid_mask = []

        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index, end_index - args.seq_len - args.pred_len + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(

                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, args.alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()

            cur_valid_pred.append(cur_rr.cpu().numpy())
            cur_valid_gt.append(gt_batch.cpu().numpy())
            cur_valid_mask.append(mask_batch.cpu().numpy())

        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)

        cur_valid_pred_flat = np.array(cur_valid_pred).reshape(cur_valid_pred[0].shape[0], -1)
        cur_valid_gt_flat = np.array(cur_valid_gt).reshape(cur_valid_gt[0].shape[0], -1)
        cur_valid_mask_flat = np.array(cur_valid_mask).reshape(cur_valid_mask[0].shape[0], -1)

        cur_valid_perf = evaluate_regression(cur_valid_pred_flat, cur_valid_gt_flat, cur_valid_mask_flat)
    return loss, reg_loss, rank_loss, cur_valid_perf, cur_valid_pred, cur_valid_gt


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = args.seq_len
    mask_batch = mask_data[:, offset: offset + seq_len + args.pred_len]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len - 1], axis=1))



if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    args.add_argument('--universe', type=str, default='nasdaq100', help='Universe of stocks to use')
    args.add_argument('--model_name', type=str, default='StockMixer', help='Name of the model to use')

    args.add_argument('--pred_len', type=int, default=5, help='Steps for future prediction')
    args.add_argument('--seq_len', type=int, default=20, help='Lookback length for the model')
    args.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for the dataset')
    args.add_argument('--end_train_date', type=str, default='2021-12-31', help='End date for training set')
    args.add_argument('--end_valid_date', type=str, default='2022-12-31', help='End date for validation set')
    args.add_argument('--start_test_date', type=str, default='2023-01-01', help='Start date for the test set')
    args.add_argument('--end_date', type=str, default='2023-12-31', help='End date for the dataset')

    args.add_argument('--reletion_name', type=str, default='wikidata', help='Name of the relation')
    args.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args.add_argument('--alpha', type=float, default=0.1, help='Regularization parameter')
    args.add_argument('--activation', type=str, default='GELU', help='Activation function to use')
    args.add_argument('--market_num', type=int, default=20, help='Number of markets')
    args.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = args.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    metrics_path = f'./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    print('Preprocessing data...')
    eod_data, gt_data, price_data, valid_index, test_index, dates, tickers = load_dataset(
        args.data_path, args.universe, args.start_date, args.end_train_date, args.end_valid_date, args.start_test_date,
        args.end_date, args.pred_len, args.seq_len)
    stock_num = eod_data.shape[0]
    fea_num = eod_data.shape[-1]
    mask_data = np.ones((stock_num, eod_data.shape[1]), dtype=float)
    trade_dates = mask_data.shape[1]

    model = StockMixer(
        stocks=stock_num,
        time_steps=args.seq_len,
        channels=fea_num,
        market=args.market_num,
        scale=(args.seq_len // 2)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None

    dates_gt, last_seq_date, tickers = save_dates(dates, tickers, test_index, args.pred_len, args.seq_len)

    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    for epoch in range(args.epochs):
        print("epoch{}##########################################################".format(epoch + 1))
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        for j in range(valid_index):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            prediction = model(data_batch)

            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, args.alpha)

            cur_loss = cur_loss
            cur_loss.backward()
            optimizer.step()

            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()

            if j % 500 == 0:
                print(f"Iteration j: {j}")

        tra_loss = tra_loss / (valid_index - args.seq_len - args.pred_len + 1)
        tra_reg_loss = tra_reg_loss / (valid_index - args.seq_len - args.pred_len + 1)
        tra_rank_loss = tra_rank_loss / (valid_index - args.seq_len - args.pred_len + 1)
        print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

        val_loss, val_reg_loss, val_rank_loss, val_perf, _, _ = validate(valid_index, test_index)
        print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

        test_loss, test_reg_loss, test_rank_loss, test_perf, pred, gt = validate(test_index, trade_dates)
        print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf

            test_perf = {k: float(v) for k, v in test_perf.items()}

            val_perf = {k: float(v) for k, v in val_perf.items()}

            results = {
                'metrics': test_perf,
                'preds': pred,
                'labels': gt,
                'pred_date': dates_gt.to_list(),
                'last_date': last_seq_date.to_list(),
                'tickers': tickers
            }

            with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
                pickle.dump(results, f)

            with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(test_perf, f, indent=4)

            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(val_perf, f, indent=4)

        print('Valid performance:\n', 'mse:{:.2e}, mae:{:.2e}, rmse:{:.2e}, r2:{:.2e}'.format(val_perf['MSE'], val_perf['MAE'],
                                                         val_perf['RMSE'], val_perf['R2']))
        print('Test performance:\n', 'mse:{:.2e}, mae:{:.2e}, rmse:{:.2e}, r2:{:.2e}'.format(test_perf['MSE'], test_perf['MAE'],
                                                                                test_perf['RMSE'], test_perf['R2']), '\n\n')

    print('----------------------------------------------------')
    print('Best test performance:\n', 'mse:{:.2e}, mae:{:.2e}, rmse:{:.2e}, r2:{:.2e}'.format(best_test_perf['MSE'], best_test_perf['MAE'],
                                                                                                         best_test_perf['RMSE'], best_test_perf['R2']), '\n\n')
