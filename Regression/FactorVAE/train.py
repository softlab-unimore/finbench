import json
import pickle

import numpy as np
import torch
import pandas as pd
import os
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import init_data_loader, RobustZScoreNormalization
from model import train, validate, test
from utils import set_seed, DataArgument, filter_constituents_by_date
import wandb

def create_saving_path(args):
    model_save_path = f"./model_params/{args.universe}/{args.model_name}"
    metrics_path = f"./results/{args.universe}/{args.model_name}/y{args.start_test_date.split('-')[0]}"
    log_dir = f"./logs/{args.model_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    return model_save_path, metrics_path, log_dir

def extract_labels(df, args, pred_len):
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')[['date', 'instrument', 'adj_close']]
    df_close.rename(columns={'date': 'datetime'}, inplace=True)
    df_close['datetime'] = pd.to_datetime(df_close['datetime'])
    df_close = df_close.sort_values(['instrument', 'datetime'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-pred_len) - x) / x)
    df = df.merge(df_close[['datetime', 'instrument', 'Label']], on=['datetime', 'instrument'], how='left')
    return df

def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df

def cs_zscore(df, clip=None):
    def zscore(group):
        mean = group.mean()
        std = group.std()
        z = (group - mean) / std
        if clip is not None:
            z = z.clip(lower=-clip, upper=clip)
        return z

    return df.groupby(level='datetime')['Label'].transform(zscore)

def main(args, metrics_path, data_args):
    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create model
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio,
                                   hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)

    # Load dataset
    dataset = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_alpha158.csv')
    tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date)
    dataset = dataset[dataset['instrument'].isin(tickers['EODHD'].tolist())].copy()

    dataset.rename(columns={'date': 'datetime'}, inplace=True)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    # Convert date strings to datetime objects
    args.start_date = pd.to_datetime(args.start_date)
    args.end_train_date = pd.to_datetime(args.end_train_date)
    args.start_valid_date = pd.to_datetime(args.start_valid_date)
    args.end_valid_date = pd.to_datetime(args.end_valid_date)
    args.start_test_date = pd.to_datetime(args.start_test_date)
    args.end_date = pd.to_datetime(args.end_date)

    # Extract labels and select valid tickers
    dataset = extract_labels(dataset, args, pred_len=args.seq_len)
    dataset = select_valid_ticker(dataset, args.start_date, args.end_train_date)

    # Filter dataset by date range and apply robust z-score normalization
    dataset.set_index(['datetime', 'instrument'], inplace=True)
    dataset = dataset.reorder_levels(['datetime', 'instrument']).sort_index()
    train_mask = (dataset.index.get_level_values('datetime') >= args.start_date) & \
                 (dataset.index.get_level_values('datetime') <= args.end_train_date)
    robust_z = RobustZScoreNormalization(dataset[train_mask])
    dataset = robust_z.robust_zscore(dataset)

    # Filter out rows with NaN labels and fill NaN values with 0
    dataset = dataset[dataset['Label'].notna()].fillna(0)
    dataset['Label'] = cs_zscore(dataset, clip=3.0)

    # Extract dates for validation and test sets considering the seq_len parameter
    all_dates = dataset.index.get_level_values('datetime').drop_duplicates().sort_values()
    mask_test = all_dates <= args.end_valid_date
    idx = np.where(mask_test)[0][np.argmax(all_dates[mask_test])]
    last_valid_date = all_dates[idx - args.seq_len - args.pred_len + 1]
    test_dates = all_dates[(all_dates >= last_valid_date) & (all_dates <= args.end_date)].to_list()
    dates = test_dates[args.seq_len + args.pred_len:]
    last_seq_date = test_dates[args.seq_len: -args.pred_len]

    index = dataset[dataset.index.get_level_values('datetime').isin(pd.to_datetime(dates))].index
    unique_dates = sorted(index.get_level_values('datetime').unique())
    ticker_per_date = [
        index[index.get_level_values('datetime') == date].get_level_values('instrument').tolist()
        for date in unique_dates
    ]

    train_dataloader = init_data_loader(dataset,
                                        shuffle=True,
                                        step_len=data_args.seq_len,
                                        start=data_args.start_time,
                                        end=data_args.fit_end_time,
                                        select_feature=data_args.select_feature)

    valid_dataloader = init_data_loader(dataset,
                                        shuffle=False,
                                        step_len=data_args.seq_len,
                                        start=data_args.val_start_time,
                                        end=data_args.val_end_time,
                                        select_feature=data_args.select_feature)

    test_dataloader = init_data_loader(dataset,
                                        shuffle=False,
                                        step_len=data_args.seq_len,
                                        start=args.start_test_date.strftime('%Y-%m-%d'),
                                        end=args.end_date.strftime('%Y-%m-%d'),
                                        select_feature=data_args.select_feature)

    T_max = len(train_dataloader) * args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************")
    args.device = device

    factorVAE.to(device)
    best_val_loss = 10000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if args.wandb:
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
        wandb.config.update(args)

    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, scheduler, args)
        val_loss, val_metrics, val_preds, val_labels = validate(factorVAE, valid_dataloader, args)
        test_loss, test_metrics, preds, labels = test(factorVAE, test_dataloader, args)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # save_root = os.path.join(args.save_dir, f'{args.run_name}_factor_{args.num_factor}_hdn_{args.hidden_size}_port_{args.num_portfolio}_seed_{args.seed}.pt')
            # torch.save(factorVAE.state_dict(), save_root)
            # print(f"Model saved at {save_root}")

            test_metrics = {k: float(v) for k, v in test_metrics.items()}
            val_metrics = {k: float(v) for k, v in val_metrics.items()}

            results = {
                'metrics': test_metrics,
                'preds': preds,
                'labels': labels,
                'pred_date': dates,
                'last_date': last_seq_date,
                'tickers': ticker_per_date
            }

            with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
                pickle.dump(results, f)

            with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(test_metrics, f, indent=4)

            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(val_metrics, f, indent=4)

        if args.wandb:
            wandb.log(
                {"Train Loss": train_loss, "Validation Loss": val_loss, "Learning Rate": scheduler.get_last_lr()[0]})

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")


    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

    parser.add_argument('--data_path', type=str, default='./data', help='path to the dataset')
    parser.add_argument('--universe', type=str, default='sp500')
    parser.add_argument('--model_name', type=str, default='FactorVAE1_sp500')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--num_latent', type=int, default=157, help='number of variables')
    parser.add_argument('--num_portfolio', type=int, default=128, help='number of stocks')

    parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction length')
    parser.add_argument('--num_factor', type=int, default=96, help='number of factors')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')

    parser.add_argument('--start_date', type=str, default='2010-01-01', help='start time')
    parser.add_argument('--end_train_date', type=str, default='2010-01-31', help='fit end time')
    parser.add_argument('--start_valid_date', type=str, default='2010-02-01', help='validation start time')
    parser.add_argument('--end_valid_date', type=str, default='2010-02-28', help='validation end time')
    parser.add_argument('--start_test_date', type=str, default='2010-03-01', help='test start time')
    parser.add_argument('--end_date', type=str, default='2010-03-31', help='end time')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--run_name', type=str, default='VAE-Revision2', help='name of the run')
    parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    args = parser.parse_args()

    # Create saving paths
    model_save_path, metrics_path, log_dir = create_saving_path(args)

    data_args = DataArgument(
        start_time=args.start_date,
        end_time=args.end_date,
        fit_end_time=args.end_train_date,
        val_start_time=args.start_valid_date,
        val_end_time=args.end_valid_date,
        seq_len=args.seq_len
    )

    main(args, metrics_path, data_args)
