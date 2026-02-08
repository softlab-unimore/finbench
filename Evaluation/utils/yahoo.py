import pandas as pd
import yfinance as yf


def download_commodities(instruments: list[str]) -> pd.DataFrame:
    """Downloads daily closing prices for specified commodity futures.

    Args:
        instruments: List of commodity ticker symbols (e.g., ['HG=F', 'NG=F', 'GC=F', 'SI=F']).

    Returns:
        pd.DataFrame: Daily closing prices with datetime index and commodity symbols as columns.
                     Index is named 'date' and sorted in ascending chronological order.

    Raises:
        ValueError: If no valid data is returned for the specified instruments.
        ConnectionError: If there are network issues accessing Yahoo Finance.
        KeyError: If the 'Close' price data is not available for the instruments.
    """
    commodities = yf.download(instruments, auto_adjust=True, progress=False)
    commodities = commodities['Close']
    commodities.index.name = 'date'
    commodities.index = pd.to_datetime(commodities.index)
    commodities = commodities.sort_index(ascending=True)

    if commodities.empty:
        raise ValueError(f"No valid data returned for instruments: {instruments}")

    return commodities


def download_commodities_returns(instruments: list[str]) -> pd.DataFrame:
    """Downloads commodity prices and calculates daily percentage returns.

    Args:
        instruments: List of commodity ticker symbols (e.g., ['HG=F', 'NG=F', 'GC=F', 'SI=F']).

    Returns:
        pd.DataFrame: Daily percentage returns with datetime index and commodity symbols as columns.
                     First row will contain NaN values as returns cannot be calculated for the first date.

    Raises:
        ValueError: If no valid data is returned for the specified instruments.
        ConnectionError: If there are network issues accessing Yahoo Finance.
        KeyError: If the 'Close' price data is not available for the instruments.
    """
    commodities = download_commodities(instruments)
    return commodities.pct_change(fill_method=None)
