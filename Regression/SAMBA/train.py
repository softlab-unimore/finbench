import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from training import print_model_parameters, Trainer
from config.model_config import ModelArgs
from models.samba import SAMBA
from preprocessing import CustomDataset
from utils import filter_constituents_by_date


def apply_normalization(df, start_date, end_date):
    # Handle of infinity values
    df.iloc[:, 1:] = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan)

    # Apply Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].iloc[:, 1:]
    scaler.fit(train_df)
    df = scaler.transform(df.iloc[:, 1:])
    return df

def get_metrics(preds, labels):
    preds_flat = np.array(preds)
    labels_flat = np.array(labels)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat)
    }
    return metrics


if __name__=='__main__':
    args = ArgumentParser()

    # Dataset parameters
    args.add_argument('--data_path', type=str, default='../../Evaluation/data')
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--model_name', type=str, default='SAMBA')

    # Prediction parameters
    args.add_argument('--seq_len', type=int, default=5)
    args.add_argument('--pred_len', type=int, default=1)
    args.add_argument('--start_date', type=str, default='2016-01-01')
    args.add_argument('--end_train_date', type=str, default='2019-12-31')
    args.add_argument('--start_valid_date', type=str, default='2020-01-01')
    args.add_argument('--end_valid_date', type=str, default='2020-12-31')
    args.add_argument('--start_test_date', type=str, default='2021-01-01')
    args.add_argument('--end_date', type=str, default='2021-12-31')

    # Model Hyperparameters
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--seed', type=int, default=1)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--epochs', type=int, default=1500)
    args.add_argument('--d_model', type=int, default=32)
    args.add_argument('--n_layer', type=int, default=3)
    args.add_argument('--embed_dim', type=int, default=10)
    args.add_argument('--d_conv', type=int, default=3)
    args.add_argument('--loss_func', type=str, default='mae', choices=('mse', 'mae'))
    args.add_argument('--lr_decay', type=bool, default=True)
    args.add_argument('--lr_decay_step', type=str, default=[40, 70, 100])
    args.add_argument('--early_stop', type=bool, default=True)
    args.add_argument('--early_stop_patience', type=int, default=200)
    args.add_argument('--grad_norm', type=bool, default=False)
    args.add_argument('--max_grad_norm', type=float, default=5.0)
    args.add_argument('--mae_thresh', type=float, default=None)
    args.add_argument('--mape_thresh', type=float, default=0.0)
    args.add_argument('--log_dir', type=str, default='./')
    args.add_argument('--log_step', type=int, default=20)
    args.add_argument('--num_workers', type=int, default=4)

    args = vars(args.parse_args())

    metrics_path = f"./results/{args.get('universe')}/{args.get('model_name')}/y{args.get('start_test_date').split('-')[0]}"
    os.makedirs(metrics_path, exist_ok=True)

    df_close = pd.read_csv(f"{args.get('data_path')}/{args.get('universe')}/{args.get('universe')}.csv")
    df_close = df_close[['date', 'instrument', 'close']]

    df_tech = pd.read_csv(f'{args.get("data_path")}/{args.get("universe")}/{args.get("universe")}_tech.csv')
    df_tech = df_tech[['instrument', 'date', 'vol', 'mom1', 'mom2', 'mom3', 'roc5', 'roc10', 'roc15', 'roc20', 'ema10',
                       'ema20', 'ema50', 'ema200']]

    df_cnnpred = pd.read_csv(f'{args.get("data_path")}/cnnpred_market.csv')

    df = df_tech.merge(df_close, how='inner', on=['instrument', 'date'])
    df = df.merge(df_cnnpred, on='date', how='inner')

    constituents = pd.read_csv(f'{args.get("data_path")}/constituents/eodhd/{args.get("universe")}.csv')
    tickers = filter_constituents_by_date(constituents, args.get("start_test_date"))['EODHD'].tolist()
    df = df[df['instrument'].isin(tickers)]

    df = df[(df['date'] >= args.get("start_date")) & (df['date'] <= args.get("end_date"))]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred_date = []
    last_date = []
    labels = []
    preds = []
    ticker_pred = []
    val_labels = []
    val_preds = []

    for ticker in tickers:
        print('-----------------------------------------')
        print('Training model for ticker:', ticker)
        print('-----------------------------------------')

        df_ticker = df[df['instrument'] == ticker].copy()

        # filtro periodo di training
        train_df = df_ticker[
            (df_ticker['date'] >= args.get('start_date')) &
            (df_ticker['date'] <= args.get('end_train_date'))
            ]

        # controllo minimo numero di campioni
        if train_df.shape[0] <= args.get('seq_len') + args.get('pred_len'):
            print(f"Skip {ticker}: only {train_df.shape[0]} samples (<={args.get('seq_len') + args.get('pred_len')})")
            continue

        df_ticker = df_ticker.drop(columns=['instrument'])
        df_ticker['Label'] = (df_ticker['close'].shift(-args.get('pred_len')) - df_ticker['close']) / df_ticker['close']

        df_ticker['vol'] = df_ticker['vol'].astype(float)
        df_ticker.iloc[:, 1:-1] = apply_normalization(df_ticker.iloc[:, :-1], args.get('start_date'),
                                                      args.get('end_train_date'))

        train_dataset = CustomDataset(
            X=df_ticker,
            seq_len=args.get('seq_len'),
            pred_len=args.get('pred_len'),
            start_date=args.get('start_date'),
            end_date=args.get('end_train_date'),
        )

        valid_dataset = CustomDataset(
            X=df_ticker,
            seq_len=args.get('seq_len'),
            pred_len=args.get('pred_len'),
            start_date=args.get('start_valid_date'),
            end_date=args.get('end_valid_date'),
            period='valid'
        )

        test_dataset = CustomDataset(
            X=df_ticker,
            seq_len=args.get('seq_len'),
            pred_len=args.get('pred_len'),
            start_date=args.get('start_test_date'),
            end_date=args.get('end_date'),
            period='test'
        )

        if valid_dataset.idx <= 0 or test_dataset.idx <= 0:
            continue

        if (len(train_dataset.windows) >= args.get('batch_size') or len(valid_dataset.windows) >= args.get('batch_size')
                or len(test_dataset.windows) >= args.get('batch_size')):
            train_loader = DataLoader(train_dataset, batch_size=args.get('batch_size'), shuffle=True, drop_last=True, num_workers=args.get("num_workers"), pin_memory=True)
            if len(train_loader) == 0:
                continue
            else:
                print(f'No data for ticker {ticker}. Skipping...')
            val_loader = DataLoader(valid_dataset, batch_size=args.get('batch_size'), shuffle=False, drop_last=False, num_workers=args.get("num_workers"), pin_memory=True)
            if len(val_loader) == 0:
                continue
            else:
                print(f'No data for ticker {ticker}. Skipping...')
            test_loader = DataLoader(test_dataset, batch_size=args.get('batch_size'), shuffle=False, drop_last=False, num_workers=args.get("num_workers"), pin_memory=True)
            if len(test_loader) == 0:
                continue
            else:
                print(f'No data for ticker {ticker}. Skipping...')
        else:
            print(f'No data for ticker {ticker}. Skipping...')
            continue

        model = SAMBA(
            args=ModelArgs(
                d_model=args.get('d_model'),
                n_layer=args.get('n_layer'),
                vocab_size=train_dataset.windows.shape[2],
                seq_in=args.get('seq_len'),
                seq_out=args.get('pred_len'),
            ),
            hidden=args.get('d_model'),
            inp=args.get('seq_len'),
            out=args.get('pred_len'),
            embed=args.get('embed_dim'),
            cheb_k=args.get('d_conv'),
        ).to(args.get('device'))

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        print_model_parameters(model, only_num=False)

        if args.get('loss_func') == 'mae':
            loss = torch.nn.L1Loss().to(args.get('device'))
        elif args.get('loss_func') == 'mse':
            loss = torch.nn.MSELoss().to(args.get('device'))
        else:
            raise ValueError

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr'), eps=1.0e-8, weight_decay=0,
                                     amsgrad=False)

        lr_scheduler = None
        if args.get('lr_decay'):
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in args.get('lr_decay_step')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[0.5 * args.get('epochs'), 0.7 * args.get('epochs'), 0.9 * args.get('epochs')],
                gamma=0.1
            )

        trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, args=args,
                          lr_scheduler=lr_scheduler)

        trainer.train()

        y_pred, y_true = trainer.test(test_loader)

        val_y_pred, val_y_true = trainer.test(val_loader)

        preds.extend(y_pred[:, 0, 0].cpu().numpy().tolist())
        labels.extend(y_true.numpy().tolist())

        val_preds.extend(val_y_pred[:, 0, 0].cpu().numpy().tolist())
        val_labels.extend(val_y_true.numpy().tolist())

        ticker_pred.extend([ticker] * len(y_pred))
        pred_date.extend(test_dataset.dates)
        last_date.extend(test_dataset.last_dates)

    results = {
        'metrics': get_metrics(preds, labels),
        'preds': np.expand_dims(np.array(preds), axis=1),
        'labels': np.expand_dims(np.array(labels), axis=1),
        'pred_date': pred_date,
        'last_date': last_date,
        'tickers': ticker_pred
    }

    with open(f'{metrics_path}/results_sl{args.get("seq_len")}_pl{args.get("pred_len")}.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open(f'{metrics_path}/metrics_sl{args.get("seq_len")}_pl{args.get("pred_len")}.json', 'w') as f:
        json.dump(get_metrics(preds, labels), f, indent=4)

    with open(f'{metrics_path}/val_metrics_sl{args.get("seq_len")}_pl{args.get("pred_len")}.json', 'w') as f:
        json.dump(get_metrics(val_preds, val_labels), f, indent=4)