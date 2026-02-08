import datetime
from torch.utils.data import Sampler
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from loader import *


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


def build_config_from_args(args):
    data_config = {
        'START_TIME': args.start_date.replace("-", ""),
        'END_TIME': args.end_date.replace("-", ""),
        'FIT_START': args.start_date.replace("-", ""),
        'FIT_END': args.end_train_date.replace("-", ""),
        'VALID_START': args.start_valid_date.replace("-", ""),
        'VALID_END': args.end_valid_date.replace("-", ""),
        'TEST_START': args.start_test_date.replace("-", ""),
        'TEST_END': args.end_date.replace("-", ""),
    }

    model_config = {
        'd_feat': args.d_feat,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'temporal_dropout': args.temporal_dropout,
        'snum_head': args.snum_head
    }

    return data_config, model_config



class DailyBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)



def edgeIndexTransform(adj):
    tmp_coo = sp.coo_matrix(adj)
    indices = np.vstack((tmp_coo.row, tmp_coo.col))
    edge_index = torch.LongTensor(indices)
    edge_index = to_undirected(edge_index)
    return edge_index


def pprint(*args):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)
