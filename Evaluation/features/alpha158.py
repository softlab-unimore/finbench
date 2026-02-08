import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import maximum, minimum

from utils.validation import validate_prices_decorator

from .utils import ref, mean, std, slope, rsquare, resi, max_, min_, quantile, rank, idx_max, idx_min, corr, sum_


@validate_prices_decorator
def calculate_alpha158(prices: pd.DataFrame, windows: list = None, verbose: bool = False) -> pd.DataFrame:
    """ Calculate alpha158 indicators for the given DataFrame.

    # https://qlib.readthedocs.io/en/v0.6.2/component/data.html#converting-csv-format-into-qlib-format
    Note:  columns names of dataset provided by Qlib should
    include open, close, high, low, volume and factor (all adjusted)

    Args:
        prices: DataFrame with OHLCV data and a MultiIndex (security, date)
        windows: List of lookback periods to use (default Alpha158: [5, 10, 20, 30, 60])
        verbose: If True, display progress bar during calculation

    Returns:
        DataFrame with all calculated indicators
    """
    if windows is None:
        windows = [5, 10, 20, 30, 60]

    # Create dictionary to store columns (avoid PerformanceWarning)
    cols = {}

    # qlib OHLC must be adjusted
    cols['close'] = prices['adj_close']
    cols['open'] = prices['adj_open']
    cols['low'] = prices['adj_low']
    cols['high'] = prices['adj_high']
    cols['volume'] = prices['volume']

    # Calculate basic price indicators that don't depend on lookback windows
    cols['KMID'] = (cols['close'] - cols['open']) / cols['open']
    cols['KLEN'] = (cols['high'] - cols['low']) / cols['open']
    cols['KMID2'] = (cols['close'] - cols['open']) / (cols['high'] - cols['low'] + 1e-12)
    cols['KUP'] = (cols['high'] - maximum(cols['open'], cols['close'])) / cols['open']
    cols['KUP2'] = (cols['high'] - maximum(cols['open'], cols['close'])) / (cols['high'] - cols['low'] + 1e-12)
    cols['KLOW'] = (minimum(cols['open'], cols['close']) - cols['low']) / cols['open']
    cols['KLOW2'] = (minimum(cols['open'], cols['close']) - cols['low']) / (cols['high'] - cols['low'] + 1e-12)
    cols['KSFT'] = (2 * cols['close'] - cols['high'] - cols['low']) / cols['open']
    cols['KSFT2'] = (2 * cols['close'] - cols['high'] - cols['low']) / (cols['high'] - cols['low'] + 1e-12)
    cols['OPEN0'] = cols['open'] / cols['close']
    cols['HIGH0'] = cols['high'] / cols['close']
    cols['LOW0'] = cols['low'] / cols['close']

    # Pre-calculate common series that will be reused
    close_diff = cols['close'] - ref(cols['close'], 1)
    close_diff_abs = abs(close_diff)
    close_up = maximum(close_diff, 0)
    close_down = maximum(-close_diff, 0)

    volume_diff = cols['volume'] - ref(cols['volume'], 1)
    volume_diff_abs = abs(volume_diff)
    volume_up = maximum(volume_diff, 0)
    volume_down = maximum(-volume_diff, 0)

    log_volume = np.log(cols['volume'] + 1)

    # For indicators that need relative price changes
    rel_price_change = cols['close'] / ref(cols['close'], 1) - 1
    weighted_volume = abs(rel_price_change) * cols['volume']
    log_rel_volume_change = np.log(cols['volume'] / ref(cols['volume'], 1) + 1)

    # Calculate indicators for each window
    for w in tqdm(windows, desc='Calculating indicators for different windows', disable=not verbose):
        # Price-based indicators
        cols[f'ROC{w}'] = ref(cols['close'], w) / cols['close']
        cols[f'MA{w}'] = mean(cols['close'], w) / cols['close']
        cols[f'STD{w}'] = std(cols['close'], w) / cols['close']
        cols[f'BETA{w}'] = slope(cols['close'], w) / cols['close']
        cols[f'RSQR{w}'] = rsquare(cols['close'], w)
        cols[f'RESI{w}'] = resi(cols['close'], w) / cols['close']
        cols[f'MAX{w}'] = max_(cols['high'], w) / cols['close']
        cols[f'MIN{w}'] = min_(cols['low'], w) / cols['close']
        cols[f'QTLU{w}'] = quantile(cols['close'], w, 0.8) / cols['close']
        cols[f'QTLD{w}'] = quantile(cols['close'], w, 0.2) / cols['close']
        cols[f'RANK{w}'] = rank(cols['close'], w)

        # Relative strength indicators
        min_low = min_(cols['low'], w)
        max_high = max_(cols['high'], w)
        cols[f'RSV{w}'] = (cols['close'] - min_low) / (max_high - min_low + 1e-12)

        # Index-based indicators
        cols[f'IMAX{w}'] = idx_max(cols['high'], w) / w
        cols[f'IMIN{w}'] = idx_min(cols['low'], w) / w
        cols[f'IMXD{w}'] = cols[f'IMAX{w}'] - cols[f'IMIN{w}']

        # Correlation indicators
        cols[f'CORR{w}'] = corr(cols['close'], log_volume, w)
        cols[f'CORD{w}'] = corr(cols['close'] / ref(cols['close'], 1), log_rel_volume_change, w)

        # Count-based indicators
        cols[f'CNTP{w}'] = mean(cols['close'] > ref(cols['close'], 1), w)
        cols[f'CNTN{w}'] = mean(cols['close'] < ref(cols['close'], 1), w)
        cols[f'CNTD{w}'] = cols[f'CNTP{w}'] - cols[f'CNTN{w}']

        # Sum-based indicators
        sum_abs_changes = sum_(close_diff_abs, w)
        cols[f'SUMP{w}'] = sum_(close_up, w) / (sum_abs_changes + 1e-12)
        cols[f'SUMN{w}'] = sum_(close_down, w) / (sum_abs_changes + 1e-12)
        cols[f'SUMD{w}'] = (sum_(close_up, w) - sum_(close_down, w)) / (sum_abs_changes + 1e-12)

        # Volume-based indicators
        cols[f'VMA{w}'] = mean(cols['volume'], w) / (cols['volume'] + 1e-12)
        cols[f'VSTD{w}'] = std(cols['volume'], w) / (cols['volume'] + 1e-12)
        cols[f'WVMA{w}'] = std(weighted_volume, w) / (mean(weighted_volume, w) + 1e-12)

        sum_abs_vol_changes = sum_(volume_diff_abs, w)
        cols[f'VSUMP{w}'] = sum_(volume_up, w) / (sum_abs_vol_changes + 1e-12)
        cols[f'VSUMN{w}'] = sum_(volume_down, w) / (sum_abs_vol_changes + 1e-12)
        cols[f'VSUMD{w}'] = (sum_(volume_up, w) - sum_(volume_down, w)) / (sum_abs_vol_changes + 1e-12)

    # Convert to DataFrame
    res_df = pd.DataFrame(cols, index=prices.index)

    # Drop OHLC+V columns
    res_df = res_df.drop(columns=['open', 'high', 'low', 'close', 'volume'])

    return res_df
