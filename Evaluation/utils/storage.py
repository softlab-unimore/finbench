import json

import pandas as pd

from .validation import validate_prices


def load_config(config_file: str = 'config.json') -> dict:
    """Loads configuration from config.json file.

    Returns:
        dict: Configuration parameters including API token.

    Raises:
        FileNotFoundError: If config.json is missing.
        json.JSONDecodeError: If config.json is invalid.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'eodhd' not in config:
                raise KeyError(f"api_token not found in {config_file}")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"{config_file} not found. Please copy config.template.json and fill in your API token")


def resample_stock_dataset(df: pd.DataFrame, freq: str = 'ME') -> pd.DataFrame:
    """Resample a stock dataset with a multi-index containing dates to a specified frequency

    This function extracts dates from the DataFrame's multi-index, resamples them to the
    specified frequency, and returns a subset of the original DataFrame containing only
    the data points that correspond to the resampled dates

    Args:
        df (pd.DataFrame): Input DataFrame with a multi-index that includes a 'date' level
        freq (str, optional): Pandas frequency string for resampling. Common values:
                              - 'D': Daily
                              - 'W': Weekly
                              - 'ME': Month End (default)
                              - 'MS': Month Start
                              - 'QE': Quarter End
                              - 'YE': Year End
    Returns:
        pd.DataFrame: Resampled DataFrame containing only rows with dates that match the resampling frequency.
            Maintains the original structure and all columns of the input DataFrame
    Raises:
        KeyError: If 'date' is not found in the DataFrame's index levels.
        ValueError: If the DataFrame is empty or if an invalid method is provided.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    if 'date' not in df.index.names:
        raise ValueError("Input DataFrame's index must contain a level named 'date'")

    # Extract dates from the multi-index
    dates = df.index.get_level_values('date')

    # Create resampler based on method and get the actual sampled dates
    resample_series = pd.Series(dates, index=dates).resample(freq).last()
    sampled_dates = resample_series.dropna().tolist()

    # Filter original DataFrame to include only sampled dates
    df_sampled = df[dates.isin(sampled_dates)].copy()
    return df_sampled


def read_stock_dataset(filepath: str) -> pd.DataFrame:
    """ Loads and processes a multi-instrument stock dataset from a CSV file

    Args:
        filepath: Path to the CSV file containing stock data for multiple instruments

    Returns:
        A DataFrame with instrument and date as multi-index and validated prices

    Raises:
        ValueError: If required columns are missing, date parsing fails, or
            price validation fails
    """
    stock_dataset = pd.read_csv(filepath)

    if 'date' not in stock_dataset.columns:
        raise ValueError('Required column "date" not found in dataset')
    if 'instrument' not in stock_dataset.columns:
        raise ValueError('Required column "instrument" not found in dataset')

    stock_dataset['date'] = pd.to_datetime(stock_dataset['date'], format='%Y-%m-%d')
    stock_dataset.set_index(['instrument', 'date'], drop=True, inplace=True)

    validate_prices(stock_dataset)
    return stock_dataset


def read_time_series(filepath: str) -> pd.DataFrame:
    """ Loads and processes multivariate time series data from a CSV file

    Args:
        filepath: Path to the CSV file containing time series data

    Returns:
        A DataFrame with date as index, sorted chronologically

    Raises:
        ValueError: If required column 'date' is missing or if dates are not
            monotonically increasing
    """
    time_series = pd.read_csv(filepath)

    if 'date' not in time_series.columns:
        raise ValueError('Required column "date" not found in dataset')

    time_series['date'] = pd.to_datetime(time_series['date'], format='%Y-%m-%d')
    time_series.set_index('date', inplace=True)

    if not time_series.index.is_monotonic_increasing:
        raise ValueError('Date index must be monotonically increasing')

    return time_series
