import pickle

import torch
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gats import GATModel
from qlib.contrib.model.pytorch_sfm import SFM_Model
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.model.pytorch_transformer import Transformer
from model import MLP, HIST
from utils import metric_fn, mse, filter_constituents_by_date, select_valid_ticker
from dataloader import DataLoader, RobustZScoreNormalization

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    if model_name.upper() == 'GATS':
        return GATModel

    if model_name.upper() == 'SFM':
        return SFM_Model

    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'HIST':
        return HIST

    raise ValueError('unknown model name `%s`'%model_name)


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    if pred.shape == torch.Size([]):
        pred = pred.unsqueeze(0)

    return mse(pred[mask], label[mask])


global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)


def zscore(x):
    return (x - x.mean()).div(x.std())

global_step = -1
def train_epoch(model, optimizer, train_loader, args, stock2concept_matrix = None):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value , stock_index, _ = train_loader.get(slc)

        if label.shape[0] == 1:
            continue

        label = zscore(label)
        if args.model_name == 'HIST':
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
        else:
            pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(model, test_loader, args, stock2concept_matrix=None, prefix='Test'):

    model.eval()

    losses = []
    preds_ic = []
    preds = []
    labels = []
    stock_indexes = []
    mask = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
        feature, label, market_value, stock_index, index = test_loader.get(slc)
        label = zscore(label)

        if label.shape[0] == 1:
            mask.append(False)
            continue

        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            else:
                pred = model(feature)
            loss = loss_fn(pred, label, args)
            preds_ic.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())

        labels.append(label.cpu().numpy())
        preds.append(pred.cpu().numpy())
        stock_indexes.append(stock_index.cpu().numpy())
        mask.append(True)

    preds_ic = pd.concat(preds_ic, axis=0)
    _, _, ic, _ = metric_fn(preds_ic)

    return preds, labels, stock_indexes, np.mean(losses), ic, mask

def inference(model, data_loader, stock2concept_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        label = zscore(label)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            else:
                pred = model(feature)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args):

    df_alpha = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_alpha360.csv')
    tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_constituents.csv'), args.start_test_date)
    df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

    # Extract labels
    df_close = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv', usecols=['date', 'instrument', 'adj_close'])
    df_close = df_close.sort_values(['instrument', 'date'])
    df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-args.pred_len) - x) / x)
    df_alpha, df_close = df_alpha.rename(columns={'date': 'datetime'}), df_close.rename(columns={'date': 'datetime'})
    df_alpha = df_alpha.merge(df_close[['datetime', 'instrument', 'Label']], on=['datetime', 'instrument'], how='left')

    # Apply normalization
    df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)
    robust_z_score = RobustZScoreNormalization(df_alpha[(df_alpha['datetime'] >= args.start_date) & (df_alpha['datetime'] <= args.end_train_date)])
    df_alpha = df_alpha.groupby('instrument', group_keys=False).apply(lambda g: g.iloc[:-args.pred_len])
    df_alpha = robust_z_score.robust_zscore(df_alpha)
    df_alpha = df_alpha.fillna(0)

    df_alpha = df_alpha[(df_alpha['datetime'] >= args.start_date) & (df_alpha['datetime'] <= args.end_date)]
    df_alpha = df_alpha.set_index(['datetime', 'instrument'])

    # Exctract concept matrix
    stock2concept_matrix = np.load(f'{args.data_path}/{args.universe}/{args.universe}_inc_matrix.npz')

    tickers_df = set(df_alpha.index.get_level_values('instrument').unique())
    tickers_matrix = set(stock2concept_matrix['tickers'])
    missing_in_matrix = tickers_df - tickers_matrix

    df_alpha = df_alpha[~df_alpha.index.get_level_values('instrument').isin(missing_in_matrix)]
    stock_index = {f'{tick}': i for i, tick in enumerate(stock2concept_matrix['tickers'])}
    df_alpha['stock_index'] = df_alpha.index.get_level_values('instrument').map(stock_index).astype(int)

    # Exctract market value
    df_market_value = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}_market_cap.csv')
    df_market_value = df_market_value.rename(columns={'date': 'datetime'}).set_index(['datetime', 'instrument'])
    df_market_value = df_market_value[['market_cap']]
    df_market_value = df_market_value / 1000000000

    all_dates = df_alpha.index.get_level_values('datetime').unique().sort_values()
    all_dates = pd.to_datetime(all_dates)

    valid_start_idx = all_dates.searchsorted(pd.to_datetime(args.start_valid_date))
    test_start_idx = all_dates.searchsorted(pd.to_datetime(args.start_test_date))
    valid_ctx_idx = max(0, valid_start_idx - args.pred_len)
    test_ctx_idx = max(0, test_start_idx - args.pred_len)
    start_valid_date = all_dates[valid_ctx_idx].strftime('%Y-%m-%d')
    test_date = all_dates[test_ctx_idx].strftime('%Y-%m-%d')

    # Train
    df_train = df_alpha[df_alpha.index.get_level_values('datetime') <= args.end_train_date]
    df_train = df_train.join(df_market_value, how='inner').rename(columns={'market_cap': 'market_value'})
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train = df_train.groupby(level='instrument', group_keys=False).apply(lambda g: g.iloc[:-args.pred_len])
    df_train = df_train.sort_index(level=['datetime', 'instrument'])
    start_index = 0
    train_loader = DataLoader(df_train.iloc[:, :-3], pd.DataFrame(df_train["Label"]), df_train['market_value'], df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    # Validation
    df_valid = df_alpha[(df_alpha.index.get_level_values('datetime') >= start_valid_date) & (df_alpha.index.get_level_values('datetime') <= args.end_valid_date)]
    df_valid = df_valid.join(df_market_value, how='inner').rename(columns={'market_cap': 'market_value'})
    df_valid['market_value'] = df_valid['market_value'].fillna(df_valid['market_value'].mean())
    df_valid = df_valid.groupby(level='instrument', group_keys=False).apply(lambda g: g.iloc[:-args.pred_len])
    df_valid = df_valid.sort_index(level=['datetime', 'instrument'])
    start_index += len(df_valid.groupby(level=0).size())
    valid_loader = DataLoader(df_valid.iloc[:, :-3], pd.DataFrame(df_valid["Label"]), df_valid['market_value'], df_valid['stock_index'],pin_memory=True, start_index=start_index, device=device)

    # Test
    df_test = df_alpha[(df_alpha.index.get_level_values('datetime') >= test_date) & (df_alpha.index.get_level_values('datetime') <= args.end_date)]
    df_test = df_test.join(df_market_value, how='inner').rename(columns={'market_cap': 'market_value'})
    df_test['market_value'] = df_test['market_value'].fillna(df_test['market_value'].mean())
    df_test = df_test.sort_index(level=['datetime', 'instrument'])
    test_last_dates = df_test.index.get_level_values('datetime').unique().tolist()[:-args.pred_len]
    pred_dates = df_test.index.get_level_values('datetime').unique().tolist()[args.pred_len:]
    df_test = df_test.groupby(level='instrument', group_keys=False).apply(lambda g: g.iloc[:-args.pred_len])
    df_test = df_test.sort_index(level=['datetime', 'instrument'])
    start_index += len(df_test.groupby(level=0).size())
    test_loader = DataLoader(df_test.iloc[:, :-3], pd.DataFrame(df_test["Label"]), df_test['market_value'], df_test['stock_index'], pin_memory=True, start_index=start_index, device=device)

    return train_loader, valid_loader, test_loader, stock2concept_matrix['inc_matrix'], stock_index, test_last_dates, pred_dates


def get_metrics(preds, labels):
    preds_flat = np.concatenate(preds, axis=0)
    labels_flat = np.concatenate(labels, axis=0)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat),
    }
    return metrics


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics_path = f"./results/{args.universe}/{args.run_name}/{args.seed}/y{args.start_test_date.split('-')[0]}"
    os.makedirs(metrics_path, exist_ok=True)

    pprint('create loaders...')
    train_loader, valid_loader, test_loader, stock2concept_matrix, stock_index, test_last_dates, pred_dates = create_loaders(args)
    stock_index = {v: k for k, v in stock_index.items()}

    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    pprint('create model...')
    if args.model_name == 'SFM':
        model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
    elif args.model_name == 'ALSTM':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
    else:
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_score = -np.inf
    best_epoch = 0
    stop_round = 0

    params_list = collections.deque(maxlen=args.smooth_steps)
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)

        pprint('training...')
        train_epoch(model, optimizer, train_loader, args, stock2concept_matrix)

        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        model.load_state_dict(avg_params)

        pprint('evaluating...')
        valid_preds, valid_labels, _, val_loss, val_score, _ = test_epoch(model, valid_loader, args, stock2concept_matrix, prefix='Valid')
        test_preds, test_labels, test_stock_indexes, test_loss, _, mask = test_epoch(model, test_loader, args, stock2concept_matrix, prefix='Test')

        pprint(f'Valid loss: {val_loss:.4f}, Test loss: {test_loss:.4f}')

        model.load_state_dict(params_ckpt)

        if val_score > best_score:
            best_score = val_score
            stop_round = 0
            best_epoch = epoch

            tickers = [[stock_index.get(num, num) for num in sublist] for sublist in test_stock_indexes]

            best_metrics = get_metrics(test_preds, test_labels)
            best_metrics = {k: float(v) for k, v in best_metrics.items()}

            pred_dates = [date for i, date in enumerate(pred_dates) if mask[i]]
            test_last_dates = [date for i, date in enumerate(test_last_dates) if mask[i]]
            test_labels = [np.expand_dims(lab, axis=1) for lab in test_labels]
            test_preds = [np.expand_dims(pred, axis=1) for pred in test_preds]

            print('-------------------------------------------------------')
            print(f'Model updated with this results: {best_metrics}')
            print('-------------------------------------------------------')

            results = {
                'metrics': best_metrics,
                'preds': test_preds,
                'labels': test_labels,
                'pred_date': pred_dates,
                'last_date': test_last_dates,
                'tickers': tickers
            }

            with open(f'{metrics_path}/results_sl1_pl{args.pred_len}.pkl', 'wb') as f:
                pickle.dump(results, f)

            with open(f'{metrics_path}/metrics_sl1_pl{args.pred_len}.json', 'w') as f:
                json.dump(best_metrics, f, indent=4)

        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best score:', best_score, '@', best_epoch)
    #torch.save(best_param, output_path+'/model.bin')


def parse_args():

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_path', type=str, default='../../Evaluation/data')
    parser.add_argument('--model_name', type=str, default='HIST')
    parser.add_argument('--run_name', type=str, default='HIST')
    parser.add_argument('--universe', type=str, default='sp500')

    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--label', default='') # specify other labels

    # dates
    parser.add_argument('--start_date', default='2015-01-01')
    parser.add_argument('--end_train_date', default='2018-12-31')
    parser.add_argument('--start_valid_date', default='2019-01-01')
    parser.add_argument('--end_valid_date', default='2019-12-31')
    parser.add_argument('--start_test_date', default='2020-01-01')
    parser.add_argument('--end_date', default='2020-12-31')
    parser.add_argument('--pred_len', type=int, default=5)

    # model
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--loss', default='mse')

    # other
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
