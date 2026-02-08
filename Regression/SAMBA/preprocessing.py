import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, seq_len, pred_len, start_date, end_date, period='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.idx = 1
        all_dates = X['date'].drop_duplicates().sort_values().reset_index(drop=True)

        if period == 'valid' or period == 'test':
            all_dates_dt, start_date_dt = pd.to_datetime(all_dates), pd.to_datetime(start_date)
            self.idx = (all_dates_dt >= start_date_dt).idxmax() - seq_len - pred_len + 1
            start_date = all_dates.iloc[self.idx]

        if self.idx > 0:
            X = X[(X['date'] >= start_date) & (X['date'] <= end_date)].copy()
            all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

            num_windows = X.shape[0] - seq_len - pred_len + 1
            windows = [X[i:i+self.seq_len] for i in range(num_windows)]
            self.dates = all_dates[self.seq_len + self.pred_len - 1:]
            self.last_dates = [windows[i].iloc[-1]['date'] for i in range(num_windows)]

            self.windows = np.stack([windows[i].iloc[:, 1:] for i in range(num_windows)])
            self.y = self.windows[:, -1, -1]

            mask_valid = np.isnan(self.windows).any(axis=(1, 2))
            self.windows = self.windows[~mask_valid]
            self.y = self.y[~mask_valid]
            self.dates = self.dates[~mask_valid].to_list()
            self.last_dates = list(np.array(self.last_dates)[~mask_valid])

    def __getitem__(self, index):
        window = torch.tensor(self.windows[index], dtype=torch.float)
        target = torch.tensor(self.y[index], dtype=torch.float).unsqueeze(-1)
        return window, target

    def __len__(self):
        return len(self.windows)

