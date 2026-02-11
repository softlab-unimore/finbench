"""Train and evaluate the Dynamic Graph Diffusion Neural Network."""

from __future__ import annotations

import json
import os
import pickle
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_gen import DatasetConfig, CustomDataset
from model.dgdnn import DGDNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ExperimentConfig:
    """All hyper-parameters required for training the model."""

    root: Path
    destination: Path
    market: str
    train_range: Tuple[str, str]
    val_range: Tuple[str, str]
    test_range: Tuple[str, str]
    window: int = 19
    pred_len: int = 1
    fast_approximation: bool = False
    layers: int = 6
    expansion_step: int = 7
    num_heads: int = 2
    embedding_hidden: int = 1024
    embedding_output: int = 256
    raw_feature_size: int = 64
    classes: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1.5e-5
    epochs: int = 1200


def set_seed(seed):
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


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


def select_valid_ticker(df, start_date, end_train, start_test, end_date, seq_len=20):
    df_train = df[(df['date'] >= start_date) & (df['date'] <= end_train)]
    df_valid = df[(df['date'] > end_train) & (df['date'] < start_test)]
    df_test = df[(df['date'] >= start_test) & (df['date'] <= end_date)]
    df_test = df_test.groupby('instrument').filter(lambda x: len(x) > seq_len)
    df_valid = df_valid.groupby('instrument').filter(lambda x: len(x) > seq_len)
    df_train = df_train.groupby('instrument').filter(lambda x: len(x) > seq_len)

    tickers_train = set(df_train['instrument'].unique())
    tickers_test = set(df_test['instrument'].unique())
    tickers_valid = set(df_valid['instrument'].unique())
    valid_tickers = tickers_train.intersection(tickers_test)
    valid_tickers = tickers_valid.intersection(valid_tickers)

    df = df[df['instrument'].isin(valid_tickers)]
    return df, valid_tickers


def preprocess_df(df):
    ticker_dict = {ticker: sub_df.set_index('date').sort_index() for ticker, sub_df in df.groupby('instrument')}
    ticker_dict = {k: v.drop(columns=['instrument']) for k, v in ticker_dict.items()}
    return ticker_dict


def build_dataset(config: ExperimentConfig, mode: str, date_range: Tuple[str, str], df) -> CustomDataset:
    dataset_config = DatasetConfig(
        root=str(config.root),
        dest=str(config.destination),
        market=config.market,
        tickers=df.keys(),
        start=date_range[0],
        end=date_range[1],
        window=config.window,
        pred_len=config.pred_len,
        mode=mode,
        fast_approx=config.fast_approximation,
    )
    return CustomDataset(df, dataset_config)


def build_model(config: ExperimentConfig, num_nodes: int, timestamp: int) -> DGDNN:
    diffusion_size = [timestamp * 5, 64, 128, 256, 256, 256, 128]
    embedding_size = [64 + 64, 128 + 256, 256 + 256, 256 + 256, 256 + 256, 128 + 256]

    emb_output = config.embedding_output
    raw_feature_size = config.raw_feature_size
    if config.num_heads != 2:
        scale = config.num_heads / 2.0
        emb_output = int(round(emb_output * scale))
        raw_feature_size = int(round(raw_feature_size * scale))
        diffusion_size = [diffusion_size[0]] + [int(round(x * scale)) for x in diffusion_size[1:]]
        embedding_size = [int(round(x * scale)) for x in embedding_size]

    model = DGDNN(
        diffusion_size,
        embedding_size,
        config.embedding_hidden,
        emb_output,
        raw_feature_size,
        config.classes,
        config.layers,
        num_nodes,
        config.expansion_step,
        config.num_heads,
        active=[True] * config.layers,
    )
    return model.to(device)


def train_epoch(model: DGDNN, dataset: CustomDataset, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sample in dataset:
        features = sample["X"].to(device)
        adjacency = sample["A"].to(device)
        target = sample["Y"].to(device).long()

        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == target).sum())
        total += target.size(0)

    accuracy = correct / total if total else 0.0
    return total_loss, accuracy


def get_metrics(preds, labels):
    preds = np.argmax(preds, axis=2).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    return {
        'F1': f1_score(labels, preds, average="macro"),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall': recall_score(labels, preds),
        'MCC': matthews_corrcoef(labels, preds)
    }


def evaluate(model: DGDNN, dataset: CustomDataset) -> Tuple[dict, list, list]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for sample in dataset:
            features = sample["X"].to(device)
            adjacency = sample["A"].to(device)
            target = sample["Y"].to(device).long()

            logits = model(features, adjacency)

            # predictions.append(np.expand_dims(logits.argmax(dim=1).cpu().numpy(), axis=1))
            predictions.append(logits.cpu().numpy())
            targets.append(np.expand_dims(target.cpu().numpy(), axis=1))

    metrics = get_metrics(predictions, targets)
    return metrics, predictions, targets


def main() -> None:
    """Entry point for training and evaluation."""

    args = ArgumentParser()

    args.add_argument('--data_path', type=str, default='./data')
    args.add_argument('--universe', type=str, default='sp500')
    args.add_argument('--model_name', type=str, default='DGDNN')

    args.add_argument('--start_date', type=str, default='2015-01-01')
    args.add_argument('--end_train_date', type=str, default='2018-12-31')
    args.add_argument('--start_valid_date', type=str, default='2019-01-01')
    args.add_argument('--end_valid_date', type=str, default='2019-12-31')
    args.add_argument('--start_test_date', type=str, default='2020-01-01')
    args.add_argument('--end_date', type=str, default='2020-12-31')

    args.add_argument('--seq_len', type=int, default=5)
    args.add_argument('--pred_len', type=int, default=1)

    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_epochs', type=int, default=1200)
    args.add_argument('--expansion_step', type=int, default=7)
    args.add_argument('--job_id', type=int, default=0)

    args = args.parse_args()

    set_seed(args.seed)
    metrics_path = f'./results/{args.universe}/{args.model_name}/{args.seed}/y{args.start_test_date.split("-")[0]}'
    os.makedirs(metrics_path, exist_ok=True)

    df = pd.read_csv(f'{args.data_path}/{args.universe}/{args.universe}.csv')
    const = pd.read_csv(f'{args.data_path}/constituents/eodhd/{args.universe}.csv')
    tickers = filter_constituents_by_date(const, args.start_test_date)
    df = df[df['instrument'].isin(tickers['EODHD'].tolist())]

    df, valid_tickers = select_valid_ticker(df, args.start_date, args.end_train_date, args.start_test_date, args.end_date, args.seq_len)
    df = df[['open', 'close', 'high', 'low', 'volume', 'instrument', 'date']]

    ticker_dict = preprocess_df(df)

    experiment = ExperimentConfig(
        root=Path(f"{args.data_path}/{args.universe}"),
        destination=Path(f"./preprocessed/DGDNN_{args.job_id}/{args.universe}"),
        window=args.seq_len,
        pred_len=args.pred_len,
        market=args.universe,
        train_range=(args.start_date, args.end_train_date),
        val_range=(args.start_valid_date, args.end_valid_date),
        test_range=(args.start_test_date, args.end_date),
        epochs=args.n_epochs,
        expansion_step=args.expansion_step
    )

    train_dataset = build_dataset(experiment, "train", experiment.train_range, ticker_dict)
    val_dataset = build_dataset(experiment, "validation", experiment.val_range, ticker_dict)
    test_dataset = build_dataset(experiment, "test", experiment.test_range, ticker_dict)

    model = build_model(experiment, num_nodes=len(valid_tickers), timestamp=experiment.window)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay
    )

    for epoch in range(1, experiment.epochs + 1):
        loss, acc = train_epoch(model, train_dataset, optimizer)
        print(f"Epoch {epoch:04d}: loss={loss:.4f}, acc={acc:.4f}")

    val_metrics, _, _ = evaluate(model, val_dataset)
    test_metrics, preds, labels = evaluate(model, test_dataset)

    print(f"Validation -- Acc: {val_metrics['Accuracy']}, F1: {val_metrics['F1']}, MCC: {val_metrics['MCC']}")
    print(f"Test -- Acc: {test_metrics['Accuracy']}, F1: {test_metrics['F1']}, MCC: {test_metrics['MCC']}")

    last_dates = [d.strftime("%Y-%m-%d") for d in test_dataset.dates[args.seq_len - 1: -args.pred_len]]
    pred_dates = [d.strftime("%Y-%m-%d") for d in test_dataset.dates[args.seq_len + args.pred_len - 1 :]]

    results = {
        'metrics': test_metrics,
        'preds': preds,
        'labels': labels,
        'last_date': last_dates,
        'pred_date': pred_dates,
        'tickers': [list(test_dataset.tickers)]*(len(test_dataset)+1),
    }

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(val_metrics, f, indent=4)

    with open(f'{metrics_path}/metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)

    with open(f'{metrics_path}/results_sl{args.seq_len}_pl{args.pred_len}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('Test metrics: ', test_metrics)


if __name__ == "__main__":
    main()