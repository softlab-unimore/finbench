import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


class RobustZScoreNormalization:
    def __init__(self, df, eps=1e-12):
        self.eps = eps
        self.z_score_cols = df.columns[2:-1]  # Exclude 'date', 'instrument' and 'Label' columns
        self.median = df.median(numeric_only=True)
        abs_dev = df[self.z_score_cols].transform(lambda x: (x - x.median()).abs())
        self.mad = abs_dev.median(numeric_only=True) + self.eps

    def robust_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        mad = self.mad * 1.4826  # Scale MAD to match standard deviation
        df[self.z_score_cols] = (df[self.z_score_cols] - self.median[self.z_score_cols]) / mad[self.z_score_cols]
        df[self.z_score_cols] = np.clip(df[self.z_score_cols], -3, 3)
        return df


class CSVDataset(Dataset):
    def __init__(self, df_alpha, seq_len, pred_len, start_date, end_date, z_score, period='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len

        all_dates = df_alpha['date'].drop_duplicates().sort_values().reset_index(drop=True)

        if period == 'valid' or period == 'test':
            all_dates_dt, start_date_dt = pd.to_datetime(all_dates), pd.to_datetime(start_date)
            idx = (all_dates_dt >= start_date_dt).idxmax() - seq_len - pred_len + 1
            start_date = all_dates[idx]

        df_alpha = df_alpha[(df_alpha['date'] >= start_date) & (df_alpha['date'] <= end_date)].copy()
        df_alpha = df_alpha.groupby('instrument', group_keys=False).apply(lambda g: g.iloc[:-self.pred_len])
        df_alpha = df_alpha.sort_values(['instrument', 'date']).reset_index(drop=True)

        print(f'Applying z-score normalization for {period} period...')
        df_alpha = z_score.robust_zscore(df_alpha)
        df_alpha = df_alpha.fillna(0)

        self.input_dates, self.output_dates, all_dates = self.extract_dates(all_dates, start_date, end_date)

        self.feature_cols = [col for col in df_alpha.columns if col not in ['date', 'instrument']]

        self.ticker_dfs = {
            ticker: group.reset_index(drop=True)
            for ticker, group in df_alpha.groupby('instrument', group_keys=False)
            if len(group) >= seq_len
        }

        print(f'Extracting windows aligned by date for {period} period...')
        self.extract_windows_aligned_by_date(all_dates)

        print(f'Dataset loaded {period}...')

    def extract_windows_aligned_by_date(self, dates):
        self.date_to_data = {}
        self.date_to_idx = []
        self.tickers_to_date = []

        for current_date in tqdm(dates):
            sequences = []
            tickers = []
            for ticker, ticker_df in self.ticker_dfs.items():
                sub_df = ticker_df[ticker_df['date'] <= current_date]
                if len(sub_df) >= self.seq_len:
                    window = sub_df.iloc[-self.seq_len:]
                    if window['date'].iloc[-1] == current_date:
                        sequences.append(window[self.feature_cols].to_numpy())
                        tickers.append(ticker)

            if len(sequences) > 0:
                self.date_to_data[current_date] = np.stack(sequences, axis=0)  # [n_ticker, seq_len, feature_dim]
                self.date_to_idx.append(current_date)
                self.tickers_to_date.append(tickers)

    def extract_dates(self, dates, start_date, end_date):
        dates = dates[(dates >= start_date) & (dates <= end_date)].to_list()
        output_dates = dates[self.seq_len + self.pred_len - 1:]
        input_dates = dates[self.seq_len - 1: -self.pred_len]
        return input_dates, output_dates, dates

    def __getitem__(self, idx):
        date = self.date_to_idx[idx]
        data = self.date_to_data[date]  # [n_ticker, seq_len, feature_dim]
        return torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.date_to_idx)


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__()
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = np.ones(len(data_source), dtype=int)  # ogni indice corrisponde a 1 "giorno"
        self.daily_index = np.arange(len(data_source))  # già ordinato

    def __iter__(self):
        indices = np.arange(len(self.data_source))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield i

    def __len__(self):
        return len(self.data_source)
