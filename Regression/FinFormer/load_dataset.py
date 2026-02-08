import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def cszscore(self, x):
        return (x - x.mean()).div(x.std())


class CustomDataset(Dataset):
    def __init__(self, df_alpha, d_feat, pred_len, start_date, end_date, z_score, period='train'):
        self.d_feat = d_feat # input_size nei parametri originali
        self.pred_len = pred_len
        self.z_score = z_score

        all_dates = df_alpha['date'].drop_duplicates().sort_values().reset_index(drop=True)

        if period == 'valid' or period == 'test':
            all_dates_dt, start_date_dt = pd.to_datetime(all_dates), pd.to_datetime(start_date)
            idx = (all_dates_dt >= start_date_dt).idxmax() - pred_len
            start_date = all_dates[idx]

        df_alpha = df_alpha[(df_alpha['date'] >= start_date) & (df_alpha['date'] <= end_date)].copy()

        print(f'Applying z-score normalization for {period} period...')
        self.df_alpha = self.z_score.robust_zscore(df_alpha)
        self.df_alpha = self.df_alpha.fillna(0)

        self.df_alpha = self.df_alpha.sort_values(['date', 'instrument'])

        self.valid_dates = self.df_alpha['date'].drop_duplicates().sort_values().tolist()

        print(f'Dataset loaded {period}...')

    def restore_daily_index(self, daily_index):
        return pd.Index(self.valid_dates.loc[daily_index])

    def __getitem__(self, idx):
        date = self.valid_dates[idx]
        day_data = self.df_alpha[self.df_alpha['date'] == date].copy()

        feature_cols = day_data.columns.difference(['date', 'instrument', 'Label']).tolist()
        features = day_data[feature_cols].values.astype(np.float32)
        features = features.reshape(features.shape[0], -1, self.d_feat)

        instruments = day_data['instrument'].unique().tolist()

        label = day_data['Label'].values.astype(np.float32)
        label = self.z_score.cszscore(pd.Series(label)).values

        return {
            'data': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'instruments': instruments,
            'daily_index': idx,
            'daily_date': len(self.valid_dates)
        }

    def __len__(self):
        return len(self.valid_dates) - self.pred_len