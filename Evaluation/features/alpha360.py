import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.validation import validate_prices_decorator

from .utils import ref


@validate_prices_decorator
def calculate_alpha360(prices: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Calculate alpha360 indicators for the given DataFrame.

    # https://qlib.readthedocs.io/en/v0.6.2/component/data.html#converting-csv-format-into-qlib-format
    Note:  columns names of dataset provided by Qlib should
    include open, close, high, low, volume and factor (all adjusted)

    Args:
        prices: DataFrame with OHLCV data and a MultiIndex (security, date)
        verbose: If True, display progress bar during calculation

    Returns:
        DataFrame with all calculated indicators
    """

    # Create dictionary to store columns (avoid PerformanceWarning)
    cols = {}

    # qlib OHLC must be adjusted
    cols['close'] = prices['adj_close']
    cols['open'] = prices['adj_open']
    cols['low'] = prices['adj_low']
    cols['high'] = prices['adj_high']
    cols['volume'] = prices['volume']

    # VWAP approximation
    vwap = (cols['high'] + cols['low'] + cols['close']) / 3

    # 0 window features
    cols['CLOSE0'] = cols['close'] / cols['close']
    cols['OPEN0'] = cols['open'] / cols['close']
    cols['HIGH0'] = cols['high'] / cols['close']
    cols['LOW0'] = cols['low'] / cols['close']
    cols[f'VOLUME0'] = cols['volume'] / (cols['volume'] + 1e-12)
    cols['VWAP0'] = vwap / cols['close']

    for w in tqdm(np.arange(1, 60), desc='Calculating indicators for different windows', disable=not verbose):
        cols[f'CLOSE{w}'] = ref(cols['close'], w) / cols['close']
        cols[f'OPEN{w}'] = ref(cols['open'], w) / cols['close']
        cols[f'HIGH{w}'] = ref(cols['high'], w) / cols['close']
        cols[f'LOW{w}'] = ref(cols['low'], w) / cols['close']
        cols[f'VOLUME{w}'] = ref(cols['volume'], w) / (cols['volume'] + 1e-12)
        cols[f'VWAP{w}'] = ref(vwap, w) / cols['close']

    # Convert to DataFrame
    res_df = pd.DataFrame(cols, index=prices.index)

    # Drop OHLC+V columns
    res_df = res_df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    return res_df
