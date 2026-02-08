import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_portfolio_returns(returns: pd.DataFrame, weights: list) -> pd.Series:
    """ Computes daily returns for a rebalanced portfolio.

    Args:
        returns: DataFrame containing daily returns for each asset
        weights: List of portfolio weights for each asset

    Returns:
        Series of daily portfolio returns
    """
    if len(weights) != returns.shape[1]:
        raise ValueError("Number of weights must match number of assets")

    # Reshape weights for matrix multiplication
    weights = np.array(weights).reshape((1, -1))
    pf_rets = np.sum(returns.values * weights, axis=1)
    return pd.Series(pf_rets, index=returns.index)


def portfolio_daily_returns(
        portfolio_history: list[dict],
        asset_prices: dict[str, pd.DataFrame],
        verbose: bool = False
) -> pd.Series:
    """Calculates daily returns for a portfolio over multiple time periods

    Args:
        portfolio_history: List of portfolio snapshots containing tickers, weights and date ranges
        asset_prices: Dictionary mapping tickers to DataFrames
            with columns 'adj_close' and 'adj_open' and a datetime index
        verbose: If True, prints progress bar for processing portfolio daily returns

    Returns:
        Series of daily portfolio returns

    Raises:
        ValueError: If portfolio history is empty or if price data is missing for a required ticker

    Note:
        Portfolio is assumed to be rebalanced at each snapshot period
    """

    if not portfolio_history:
        raise ValueError("Portfolio history cannot be empty")

    # Filter asset prices to only include tickers in portfolio history
    pf_tickers = set([ticker for snp in portfolio_history for ticker in snp['tickers']])
    asset_prices = {k: df for k, df in asset_prices.items() if k in pf_tickers}

    # Precompute returns for all assets
    asset_returns = {}
    for ticker, price_df in asset_prices.items():
        assert price_df.index.is_monotonic_increasing, f'Price for {ticker} must have monotonically increasing index'
        returns_df = price_df[['adj_close', 'adj_open']].copy()
        returns_df['open_to_close'] = (returns_df['adj_close'] - returns_df['adj_open']) / returns_df['adj_open']
        returns_df['returns'] = returns_df['adj_close'].pct_change()
        returns_df = returns_df.dropna()
        asset_returns[ticker] = returns_df

    # Process each portfolio snapshot period
    period_returns = []
    for snapshot in tqdm(portfolio_history, disable=not verbose, desc='Processing Portfolio Snapshots'):
        # Extract period returns for each ticker
        period_data = []
        for ticker in snapshot['tickers']:
            if ticker not in asset_prices:
                raise ValueError(f'Missing price data for ticker {ticker}')

            # Extract returns for the snapshot period
            ticker_returns = asset_returns[ticker]
            mask = (
                    (ticker_returns.index >= snapshot['test_start']) &
                    (ticker_returns.index < snapshot['test_end'])
            )
            period_ticker_returns = ticker_returns.loc[mask, 'returns'].copy()

            # Use intraday return for first day to avoid bias
            if not period_ticker_returns.empty:
                first_date = period_ticker_returns.index[0]
                period_ticker_returns.iloc[0] = asset_returns[ticker].loc[first_date, 'open_to_close']

            period_data.append(period_ticker_returns)

        if period_data:
            # Combine returns for all tickers in the period
            combined_returns = pd.concat(period_data, axis=1)
            combined_returns.columns = snapshot['tickers']

            # Compute weighted portfolio returns
            period_returns.append(
                compute_portfolio_returns(
                    combined_returns.sort_index().fillna(0),
                    snapshot['weights']
                )
            )
        else:
            # Create a series of zeros with business day index for the period
            zero_returns = pd.Series(
                0,
                index=pd.date_range(
                    start=snapshot['test_start'],
                    end=snapshot['test_end'],
                    freq='B'
                )[:-1]  # Exclude test_end (non-inclusive)
            )
            period_returns.append(zero_returns)

    if not period_returns:
        raise ValueError('No valid returns data found in any period')

    # Combine all period returns
    all_returns = pd.concat(period_returns, axis=0).sort_index(ascending=True)
    return all_returns
