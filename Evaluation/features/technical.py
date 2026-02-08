import pandas as pd

from utils.validation import validate_prices_decorator

from .utils import ref, mean, std, ema, sum_


@validate_prices_decorator
def calculate_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators based on OHLC price data.

    Computes normalized versions of price fields and moving averages.

    Args:
        prices: DataFrame containing OHLC price data with columns:
            'open', 'high', 'low', 'close', 'adj_close'

    Returns:
        DataFrame with original data and calculated technical indicators.
    """

    # Make a copy to avoid modifying the original
    res_df = prices.copy()

    # Calculate normalized price values relative to close
    res_df['c_open'] = res_df['open'] / res_df['close'] - 1
    res_df['c_high'] = res_df['high'] / res_df['close'] - 1
    res_df['c_low'] = res_df['low'] / res_df['close'] - 1

    # Calculate returns (current price relative to previous price)
    for col in ['open', 'high', 'low', 'close']:
        res_df[f'{col}_price_change'] = res_df[col] / ref(res_df[col], 1) - 1
        res_df[f'adj_{col}_price_change'] = res_df[f'adj_{col}'] / ref(res_df[f'adj_{col}'], 1) - 1

    # Calculate ratios of open, high, and low to previous close
    for col in ['open', 'high', 'low', 'close']:
        res_df[f'{col}_close_ratio_d1'] = res_df[col] / ref(res_df['close'], 1)
        res_df[f'adj_{col}_close_ratio_d1'] = res_df[f'adj_{col}'] / ref(res_df[f'adj_close'], 1)

    # Calculate absolute returns of close prices
    res_df['adj_close_abs_return'] = res_df['adj_close'] - ref(res_df['adj_close'], 1)
    res_df['close_abs_return'] = res_df['close'] - ref(res_df['close'], 1)

    # Calculate moving average indicators for each window
    windows = [5, 10, 15, 20, 25, 30]
    for k in windows:
        res_df[f'z_d{k}'] = sum_(res_df['adj_close'], k) / (k * res_df['adj_close']) - 1
        res_df[f'f_d{k}'] = sum_(res_df['adj_close'], k) / (k * (res_df['adj_close'] - 1))

    # TA-lib indicators (CNNPred paper) https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas/
    res_df['vol'] = res_df[f'volume'] / ref(res_df[f'volume'], 1) - 1
    res_df['mom1'] = res_df[f'adj_close'] - ref(res_df['adj_close'], 1)
    res_df['mom2'] = res_df[f'adj_close'] - ref(res_df['adj_close'], 2)
    res_df['mom3'] = res_df[f'adj_close'] - ref(res_df['adj_close'], 3)

    res_df['roc5'] = 100 * (res_df[f'adj_close'] - ref(res_df['adj_close'], 5)) / ref(res_df['adj_close'], 5)
    res_df['roc10'] = 100 * (res_df[f'adj_close'] - ref(res_df['adj_close'], 10)) / ref(res_df['adj_close'], 10)
    res_df['roc15'] = 100 * (res_df[f'adj_close'] - ref(res_df['adj_close'], 15)) / ref(res_df['adj_close'], 15)
    res_df['roc20'] = 100 * (res_df[f'adj_close'] - ref(res_df['adj_close'], 20)) / ref(res_df['adj_close'], 20)

    res_df['ema10'] = ema(res_df['adj_close'], 10)
    res_df['ema20'] = ema(res_df['adj_close'], 20)
    res_df['ema50'] = ema(res_df['adj_close'], 50)
    res_df['ema200'] = ema(res_df['adj_close'], 200)

    # Drop original OHLC columns
    res_df = res_df.drop(columns=prices.columns)

    return res_df


@validate_prices_decorator
def calculate_market_information(prices: pd.DataFrame, instruments: list[str]) -> pd.DataFrame:
    """Calculate market information metrics for specified instruments.

    Computes price changes, rolling statistics for price changes and volume
    ratios across multiple time windows for the given instruments.

    Args:
        prices: Multi-index DataFrame with 'instrument' level containing OHLCV data. Expected columns:'adj_close', 'volume'
        instruments: List of instrument identifiers to process.

    Returns:
        DataFrame with calculated metrics unstacked by instrument. Columns are
        named as '{metric}_{instrument}' format.

    Raises:
        ValueError: If any instrument in instruments is not found in the DataFrame.
    """
    # Validate that all instruments exist in the DataFrame
    available_instruments = prices.index.get_level_values('instrument').unique()
    missing_instruments = [instrument for instrument in instruments if instrument not in available_instruments]
    if missing_instruments:
        raise ValueError(f'Instruments not found in DataFrame: {missing_instruments}')

    # Filter DataFrame to requested instruments
    res_df = prices.loc[instruments].copy()

    # Calculate daily price changes (returns)
    res_df['price_change'] = res_df['adj_close'] / ref(res_df['adj_close'], 1) - 1
    # Calculate rolling statistics for multiple time windows
    for w in [5, 10, 20, 30, 60]:
        # Rolling mean and standard deviation of price changes
        res_df[f'mean_price_change_{w}'] = mean(res_df['price_change'], w)
        res_df[f'std_price_change_{w}'] = std(res_df['price_change'], w)
        # Volume statistics relative to current volume
        res_df[f'mean_vol_{w}'] = mean(res_df['volume'], w) / res_df['volume']
        res_df[f'std_vol_{w}'] = std(res_df['volume'], w) / res_df['volume']

    # Remove original OHLCV columns, keeping only calculated metrics
    res_df = res_df.drop(columns=prices.columns)

    # Reshape data: instruments become columns
    res_df = res_df.unstack(level='instrument')
    res_df.columns = res_df.columns.map('_'.join)

    return res_df.sort_index(ascending=True)
