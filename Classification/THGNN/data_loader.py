import numpy as np
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, correlation_matrix, seq_len, pred_len, start_date, end_date, period='train'):
        self.df = dataset
        self.corr_matr = correlation_matrix
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start_date = start_date
        self.end_date = end_date
        self.period = period

        self.dates = self.df['date'].drop_duplicates().reset_index(drop=True)

        start_date_idx = self.dates.index.get_loc(self.dates[self.dates >= start_date].index[0])
        end_date_idx = self.dates.index.get_loc(self.dates[self.dates <= end_date].index[-1]) + 1

        if period == 'test' or period == 'valid':
            start_date_idx = start_date_idx - self.seq_len - self.pred_len + 1
            self.start_date = self.dates[start_date_idx]

        self.dates = self.dates[start_date_idx:end_date_idx].tolist()
        self.last_seq_dates = self.dates[self.seq_len - 1: -self.pred_len]
        self.pred_dates = self.dates[self.seq_len - 1 + self.pred_len:]

        self.df = self.df[(self.df['date'] >= self.start_date) & (self.df['date'] <= self.end_date)].reset_index(drop=True)

        self.filter_corr_matrix()
        self.extract_windows_for_features()
        self.extract_mask()

    def filter_corr_matrix(self):
        # Forza tipi coerenti
        corr_dates = np.array(self.corr_matr['dates'], dtype='datetime64[ns]')
        subset_dates = np.array(self.dates[self.seq_len - 1:-self.pred_len], dtype='datetime64[ns]')

        mask = np.isin(corr_dates, subset_dates)
        self.corr_matr['adj_matrix'] = self.corr_matr['adj_matrix'][mask]
        self.corr_matr['dates'] = subset_dates

    def extract_windows_for_features(self):
        feature_cols = [col for col in self.df.columns if col not in ['date', 'instrument']]
        self.df = (
            self.df.reset_index()
            .pivot(index='date', columns='instrument')[feature_cols]
            .reorder_levels([1, 0], axis=1).sort_index(axis=1)
            .to_numpy()
            .reshape(
                len(self.df['date'].unique()),
                len(self.df['instrument'].unique()),
                len(feature_cols)
            )
        )

        windows = []
        for i in range(self.df.shape[0] - self.seq_len - self.pred_len + 1):
            window = self.df[i:i+self.seq_len]
            windows.append(window)
        self.df = np.array(windows)
        self.df = self.df.transpose(0, 2, 1, 3)

    def extract_mask(self):
        t, tick, _, _ = self.df.shape
        values = self.df[:, :, -1, 0]
        self.mask = np.zeros((t, tick), dtype=bool)

        k = min(100, tick)

        for i in range(t):
            top_k = np.argpartition(-values[i], k-1)[:k]
            bottom_k = np.argpartition(values[i], k-1)[:k]
            selected = np.concatenate([top_k, bottom_k])
            self.mask[i, selected] = True

    def __getitem__(self, idx):
        data = {
            'pos_adj': torch.tensor(self.corr_matr['adj_matrix'][idx, 0], dtype=torch.float32),
            'neg_adj': torch.tensor(self.corr_matr['adj_matrix'][idx, 1], dtype=torch.float32),
            'features': torch.tensor(self.df[idx, :, :, 1:], dtype=torch.float32),
            'labels': torch.tensor((self.df[idx, :, -1, 0] > 0).astype(int), dtype=torch.float32),
            'mask': self.mask[idx].tolist()
        }
        return data

    def __len__(self):
        return len(self.df)