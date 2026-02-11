import copy
import json
import pickle
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from load_dataset import CSVDataset, DailyBatchSamplerRandom, RobustZScoreNormalization
from lr_scheduler import ChainedScheduler
from model.MATCC import MATCC
from utils import filter_constituents_by_date

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def save_datasets(dl_train, dl_valid, dl_test, universe, seq_len, pred_len):
    with open(f'./checkpoints/{universe}_dl_train_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_train, f)

    with open(f'./checkpoints/{universe}_dl_valid_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_valid, f)

    with open(f'./checkpoints/{universe}_dl_test_sl{seq_len}_pl{pred_len}.pkl', 'wb') as f:
        pickle.dump(dl_test, f)

def open_datasets(universe, seq_len, pred_len):
    with open(f'./checkpoints/{universe}_dl_train_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_train = pickle.load(f)

    with open(f'./checkpoints/{universe}_dl_valid_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_valid = pickle.load(f)

    with open(f'./checkpoints/{universe}_dl_test_sl{seq_len}_pl{pred_len}.pkl', 'rb') as f:
        dl_test = pickle.load(f)

    return dl_train, dl_valid, dl_test

def create_saving_path(args):
    model_save_path = f"./model_params/{args.universe}/{args.model_name}_{args.seed}"
    metrics_path = f"./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split('-')[0]}"
    log_dir = f"./logs/{args.model_name}/{args.seed}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path, exist_ok=True)

    return model_save_path, metrics_path, log_dir


def create_logger_and_writer(args, log_dir):
    logging.basicConfig(filename=os.path.join(log_dir, f"{args.model_name}_seed_{args.seed}.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(f"Train {args.model_name}")
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f"{args.model_name}_{args.seed}_")
    return logger, writer

def logging_info(logger, writer, args, train_optimizer, lr_scheduler):
    logger.info(msg=f"\n===== Model {args.model_name} =====\n"
                    f"n_epochs: {args.n_epoch}\n"
                    f"start_lr: {args.lr}\n"
                    f"T_0: {args.T_0}\n"
                    f"T_mult: {args.T_mult}\n"
                    f"gamma: {args.gamma}\n"
                    f"coef: {args.coef}\n"
                    f"cosine_period: {args.cosine_period}\n"
                    f"eta_min: {args.eta_min}\n"
                    f"seed: {args.seed}\n"
                    f"optimizer: {train_optimizer}\n"
                    f"lr_scheduler: {lr_scheduler}\n"
                    f"description: train {args.model_name}\n\n")

    writer.add_text("train_optimizer", train_optimizer.__str__())
    writer.add_text("lr_scheduler", lr_scheduler.__str__())
    writer.add_text("model_name", args.model_name)
    writer.add_text("seed", str(args.seed))
    writer.add_text("n_head", str(args.n_head))
    writer.add_text("learning rate", str(args.lr))
    writer.add_text("T_0", str(args.T_0))
    writer.add_text("T_mult", str(args.T_mult))
    writer.add_text("gamma", str(args.gamma))
    writer.add_text("coef", str(args.coef))
    writer.add_text("eta_min", str(args.eta_min))
    writer.add_text("weight_decay", str(args.weight_decay))
    writer.add_text("cosine_period", str(args.cosine_period))


def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


def _init_data_loader(data, shuffle=True, drop_last=True, num_workers=2):
    sampler = DailyBatchSamplerRandom(data, shuffle)
    data_loader = DataLoader(
        data, sampler=sampler, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    return data_loader


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    loss = (pred[mask] - label[mask]) ** 2
    return torch.mean(loss)

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)

    # Exclude top 2.5% and bottom 2.5% values
    if percent_2_5 != 0:
        filtered_indices = indices[percent_2_5:-percent_2_5]
        mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
        mask[filtered_indices] = True
        return mask, x[mask]

    return torch.ones_like(x, device=x.device, dtype=torch.bool), x

def train_epoch(data_loader, train_optimizer, lr_scheduler, model, device):
    model.train()
    losses = []

    for data in data_loader:
        data = torch.squeeze(data, dim=0)
        '''
        data.shape: (N, T, F)
        N - number of stocks
        T - length of lookback_window, 8
        F - 158 factors + 63 market information + 1 label           
        '''
        feature = data[:, :, 0:-1].to(device)
        label = data[:, -1, -1].to(device)

        mask, label = drop_extreme(label)
        feature = feature[mask, :, :]
        label = zscore(label)

        pred = model(feature.float())

        loss = loss_fn(pred, label)
        losses.append(loss.item())

        train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        train_optimizer.step()
    lr_scheduler.step()

    return float(np.mean(losses))


def valid_epoch(data_loader, model, device):
    model.eval()
    losses = []
    ic = []
    ric = []
    preds = []
    labels = []
    with torch.no_grad():
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(device)
            label = data[:, -1, -1].to(device)
            with torch.no_grad():
                pred = model(feature.float())
            loss = loss_fn(pred, label)
            losses.append(loss.item())

            daily_ic, daily_ric = calc_ic(
                pred.detach().cpu().numpy(), label.detach().cpu().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

            pred = pred.unsqueeze(-1)
            label = label.unsqueeze(-1)

            preds.append(pred.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    preds_flat = np.concatenate(preds, axis=0)
    labels_flat = np.concatenate(labels, axis=0)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat)
    }

    return float(np.mean(losses)), metrics, preds, labels

def select_valid_ticker(df, start_date, end_date):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    tickers = df_train['instrument'].drop_duplicates().tolist()
    df = df[df['instrument'].isin(tickers)]
    return df


def train(args, model_save_path, metrics_path, train_optimizer, lr_scheduler):
    if not os.path.exists(args.data_path):
        raise FileExistsError('Data dir not exists')

    if args.data_preprocessing:
        os.makedirs('./checkpoints', exist_ok=True)

        df_alpha = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}_alpha158.csv")
        tickers = filter_constituents_by_date(pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv'), args.start_test_date)
        df_alpha = df_alpha[df_alpha['instrument'].isin(tickers['EODHD'].tolist())]

        market_index = pd.read_csv(f'{args.data_path}/{args.nation}_market.csv')
        df_alpha = pd.merge(df_alpha, market_index, how='left', on='date')

        # Extract labels
        df_close = pd.read_csv(f"{args.data_path}/{args.universe}/{args.universe}.csv")[['date', 'instrument', 'adj_close']]
        df_close = df_close.sort_values(['instrument', 'date'])
        df_close['Label'] = df_close.groupby('instrument')['adj_close'].transform(lambda x: (x.shift(-args.pred_len) - x) / x)
        df_alpha = df_alpha.merge(df_close[['date', 'instrument', 'Label']], on=['date', 'instrument'], how='left')

        df_alpha = select_valid_ticker(df_alpha, args.start_date, args.end_train_date)

        robust_z_score = RobustZScoreNormalization(df_alpha[(df_alpha['date'] >= args.start_date) & (df_alpha['date'] <= args.end_train_date)])

        dl_train = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_date,
            end_date=args.end_train_date,
            z_score=robust_z_score
        )

        dl_valid = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_valid_date,
            end_date=args.end_valid_date,
            z_score=robust_z_score,
            period='valid'
        )

        dl_test = CSVDataset(
            df_alpha=df_alpha,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            start_date=args.start_test_date,
            end_date=args.end_date,
            z_score=robust_z_score,
            period='test'
        )

        #save_datasets(dl_train, dl_valid, dl_test, args.universe, args.seq_len, args.pred_len)

    else:
        dl_train, dl_valid, dl_test = open_datasets(args.universe, args.seq_len, args.pred_len)

    train_loader = _init_data_loader(dl_train, shuffle=True, drop_last=True, num_workers=args.num_workers)
    valid_loader = _init_data_loader(dl_valid, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader = _init_data_loader(dl_test, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print("==" * 10 + f" Now is Training {args.model_name}_{args.seed} " + "==" * 10 + "\n")

    # best_valid_loss = float('inf')

    for step in range(1, args.n_epoch + 1):
        train_loss = train_epoch(train_loader, train_optimizer=train_optimizer, lr_scheduler=lr_scheduler, model=model, device=device)

        val_loss, valid_metrics, _, _, = valid_epoch(valid_loader, model, device)
        test_loss, test_metrics, preds, labels = valid_epoch(test_loader, model, device)

        if writer is not None:
            writer.add_scalars(
                "Valid metrics", valid_metrics, global_step=step)
            writer.add_scalars("Test metrics", test_metrics, global_step=step)
            writer.add_scalar("Train loss", train_loss, global_step=step)
            writer.add_scalar("Valid loss", val_loss, global_step=step)
            writer.add_scalar("Test loss", test_loss, global_step=step)
            writer.add_scalars("All loss Comparison",
                               {"train loss": train_loss,
                                "val loss": val_loss, "test loss": test_loss},
                               global_step=step)
            writer.add_scalar("Learning rate", train_optimizer.param_groups[0]['lr'], global_step=step)

        print("==" * 10 + f" {args.model_name}_{args.seed} Epoch {step} " + "==" * 10)
        print("Epoch %d, train_loss %.6f, valid_loss %.6f, test_loss %.6f " %(step, train_loss, val_loss, test_loss))
        print("Valid Dataset Metrics performance:{}\n".format(valid_metrics))
        print("Test Dataset Metrics performance:{}\n".format(test_metrics))
        print("Learning rate :{}\n\n".format(train_optimizer.param_groups[0]['lr']))

        logger.info(msg=f"\n===== Epoch {step} =====\ntrain loss:{train_loss}, "
                                    f"valid loss:{val_loss},test loss:{test_loss}\n"
                                    f"valid metrics:{valid_metrics}\n"
                                    f"test metrics:{test_metrics}\n"
                                    f"learning rate:{train_optimizer.param_groups[0]['lr']}\n")

        if step <= 10: # Warm up phase
            continue

        if step % 5 == 0:
            model_param = copy.deepcopy(model.state_dict())
            torch.save(model_param,f'{model_save_path}/{args.model_name}_model_params_epoch_{step}_seed_{args.seed}.pth')
            test_metrics = {k: float(v) for k, v in test_metrics.items()}
            valid_metrics = {k: float(v) for k, v in valid_metrics.items()}

            results = {
                'metrics': test_metrics,
                'preds': preds,
                'labels': labels,
                'pred_date': dl_test.output_dates,
                'last_date': dl_test.input_dates,
                'tickers': dl_test.tickers_to_date
            }

            with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
                pickle.dump(results, f)

            with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(test_metrics, f, indent=4)

            with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
                json.dump(valid_metrics, f, indent=4)

    print("SAVING LAST EPOCH RESULT AS THE TEST RESULT!")


    # torch.save(model.state_dict(),f'{model_save_path}/TEST_{args.model_name}_model_params_seed_{args.seed}.pth')

    print("\n" + "==" * 10 + " Training Over " + "==" * 10)
    writer.close()


if __name__=='__main__':

    args = ArgumentParser()
    # Dataset Parameters
    args.add_argument('--model_name', type=str, default='MATCC')
    args.add_argument('--universe', type=str, default='sxxp')
    args.add_argument('--data_path', type=str, default='./data')
    args.add_argument('--data_preprocessing', action='store_true', default=False)
    args.add_argument('--nation', type=str, default='us')

    # Prediction Parameters
    args.add_argument('--seq_len', type=int, default=8)
    args.add_argument('--pred_len', type=int, default=1)
    args.add_argument('--start_date', type=str, default='2019-01-01')
    args.add_argument('--end_train_date', type=str, default='2022-12-31')
    args.add_argument('--start_valid_date', type=str, default='2023-01-01')
    args.add_argument('--end_valid_date', type=str, default='2023-12-31')
    args.add_argument('--start_test_date', type=str, default='2010-03-01')
    args.add_argument('--end_date', type=str, default='2010-03-31')

    # Model Hyperparameters
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_epoch', type=int, default=70)
    args.add_argument('--warmUp_epoch', type=int, default=10)
    args.add_argument('--lr', type=float, default=3e-4)
    args.add_argument('--gamma', type=float, default=1.0)
    args.add_argument('--coef', type=float, default=1.0)
    args.add_argument('--cosine_period', type=float, default=4)
    args.add_argument('--T_0', type=int, default=15)
    args.add_argument('--T_mult', type=float, default=1)
    args.add_argument('--eta_min', type=float, default=2e-5)
    args.add_argument('--weight_decay', type=float, default=1e-3)
    args.add_argument('--d_model', type=int, default=256)
    args.add_argument('--n_head', type=int, default=4)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--gate_input_start_index', type=int, default=157)
    args.add_argument('--gate_input_end_index', type=int, default=220)
    args.add_argument('--train_stop_loss_thred', type=float, default=0.95)
    args.add_argument('--num_workers', type=int, default=4)

    args = args.parse_args()

    print(args)

    torch.cuda.manual_seed(args.seed)
    model_save_path, metrics_path, log_dir = create_saving_path(args)
    logger, writer = create_logger_and_writer(args, log_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = MATCC(
        d_model=args.d_model,
        d_feat=args.gate_input_start_index,
        seq_len=args.seq_len,
        t_nhead=args.n_head,
        S_dropout_rate=args.dropout,
        gate_input_start_index=args.gate_input_start_index,
        gate_input_end_index=args.gate_input_end_index
    ).to(device)

    train_optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    lr_scheduler = ChainedScheduler(train_optimizer, T_0=args.T_0, T_mul=args.T_mult, eta_min=args.eta_min,
                                    last_epoch=-1, max_lr=args.lr, warmup_steps=args.warmUp_epoch,
                                    gamma=args.gamma, coef=args.coef, step_size=3, cosine_period=args.cosine_period)

    logging_info(logger, writer, args, train_optimizer, lr_scheduler)

    train(args, model_save_path, metrics_path, train_optimizer, lr_scheduler)







