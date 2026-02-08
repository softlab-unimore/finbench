
import numpy as np
import pandas as pd
import torch

# operation
from . import tools
from .tools import filter_constituents_by_date


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        arg: List of all the args passed in argument parser
        flag: 'train' or 'test'
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
    """

    def __init__(self,
                 arg,
                 flag='train',
                 random_choose=False,
                 random_move=False,
                 window_size=-1):

        self.arg = arg

        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.features = ['ma5', 'ma10','ma20', 'close']

        self.choose_start_end_date(flag)
        self.load_data(flag)

    def choose_start_end_date(self, flag):
        if flag == 'train':
            self.start_date = self.arg.start_date
            self.end_date = self.arg.end_train_date
        elif flag == 'test':
            self.start_date = self.arg.start_test_date
            self.end_date = self.arg.end_date
        else:
            self.start_date = self.arg.start_valid_date
            self.end_date = self.arg.end_valid_date

    def extract_idx(self, flag):
        self.dates = self.dates.reset_index(drop=True)
        if flag == 'train':
            start_idx = 0
            end_idx = pd.to_datetime(self.dates[self.dates > self.end_date]).idxmin() - self.arg.seq_len - self.arg.pred_len + 1
        else:
            start_idx = pd.to_datetime(self.dates[self.dates > self.start_date]).idxmin() - self.arg.seq_len - self.arg.pred_len + 1
            end_idx = self.dates.index.max() - self.arg.seq_len - self.arg.pred_len + 2

        return start_idx, end_idx

    def extract_ma(self, df):
        windows = [5, 10, 20]
        for w in windows:
            df[f'ma{w}'] = df.groupby('instrument')['close'].transform(
                lambda x: x.rolling(window=w, min_periods=w).mean()
            )
        df['ma_nan'] = df[['ma5', 'ma10', 'ma20']].isna().any(axis=1)
        nan_matrix = df.pivot(index='date', columns='instrument', values='ma_nan').sort_index()
        nan_mask = ~nan_matrix.any(axis=1).values
        df = df.dropna(subset=['ma5', 'ma10', 'ma20'])
        return df, nan_mask

    def extract_labels(self, df):
        df['label'] = df.groupby('instrument', group_keys=False).apply(
            lambda g: (g['close'].shift(-self.arg.pred_len) - g['close']) / g['close'])
        labels = df[['instrument', 'date', 'label']]
        labels_matr = labels.pivot(index='date', columns='instrument', values='label').sort_index()
        self.label = [row.values.astype(np.float32) for _, row in labels_matr.iterrows()]

    def normalize_data(self):
        close_idx = self.C - 1
        p_T = self.data[:, close_idx, -1, :]
        self.data = self.data / p_T[:, None, None, :]

    def load_data(self, flag):
        dataset = pd.read_csv(f'{self.arg.data_path}/{self.arg.universe}/{self.arg.universe}.csv')
        constituents = pd.read_csv(f'{self.arg.data_path}/{self.arg.universe}/{self.arg.universe}_constituents.csv')
        tickers = filter_constituents_by_date(constituents, self.arg.start_test_date)['EODHD'].tolist()
        dataset = dataset[dataset['instrument'].isin(tickers)]
        dataset = dataset[['instrument', 'date', 'close']]
        dataset = dataset.sort_values(by=['instrument', 'date'])

        # Filter by date
        all_dates = dataset['date'].drop_duplicates().sort_values()
        self.dates = all_dates[(all_dates >= self.arg.start_date) & (all_dates <= self.arg.end_date)]

        dates_df = pd.DataFrame({'date': self.dates})

        all_data = []
        for inst, df_inst in dataset.groupby('instrument'):
            merged = pd.merge(dates_df, df_inst, on='date', how='left').ffill().bfill().infer_objects()
            merged['instrument'] = inst
            all_data.append(merged)

        dataset = pd.concat(all_data, ignore_index=True)

        # Extract labels and features
        dataset, nan_mask = self.extract_ma(dataset)
        self.extract_labels(dataset)
        panel = dataset.set_index(['instrument', 'date'])[self.features].unstack('date')
        self.tickers = panel.index.to_list()
        N, F, T = len(panel.index), len(self.features), len(panel.columns.levels[1])
        data = panel.to_numpy().reshape(N, F, T)

        # Extract windows
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=self.arg.seq_len, axis=2)
        windows = np.transpose(windows, (2, 1, 3, 0))

        self.dates = self.dates[nan_mask]
        self.start_idx, self.end_idx = self.extract_idx(flag)
        self.data = windows[self.start_idx:self.end_idx]
        self.label = self.label[self.start_idx:self.end_idx]

        self.dates_gt = self.dates[self.start_idx + self.arg.seq_len + self.arg.pred_len -1 : self.end_idx + self.arg.seq_len + self.arg.pred_len -1]
        self.last_seq_date = self.dates[self.start_idx + self.arg.seq_len - 1:self.end_idx + self.arg.seq_len - 1]

        self.N, self.C, self.T, self.V = self.data.shape

        # TODO: controllare se deve essere normalizzato o meno.
        #  Da capire se eventualmente dividere tutto il per max della sequenza di train.
        self.closing_price = self.data[:, -1, -1, :]
        self.normalize_data()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = np.array(self.label[index])
        closing_price = np.array(self.closing_price[index])
        
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, closing_price, label
