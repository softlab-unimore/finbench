import pandas as pd

from .optimization import PortfolioOptimizer, OptimizationMethod, RiskMethod, ReturnMethod, ObjectiveFunction


def find_nearest_min(asset_returns: dict[str, pd.Series]) -> pd.Timestamp:
    """ Find the most recent common date available across all assets """
    return max([rets.index.min() for rets in asset_returns.values()])


def get_backtest_iterator(start_date: str, end_date: str, freq: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate a list of time periods for backtesting using pandas date_range.

    Args:
        start_date: Start date in string format (e.g., '2023-01-01')
        end_date: End date in string format (e.g., '2023-12-31')
        freq: Frequency string (e.g., '1M', '1D', '5B', '1W')

    Returns:
        List of (period_start, period_end) timestamp tuples
    """
    # Convert string dates to timestamps
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Normalize start date based on frequency
    if freq.endswith('W'):
        # For weekly, align to start of week (Monday)
        start = start - pd.Timedelta(days=start.dayofweek)
    elif freq.endswith('M'):
        # For monthly, align to start of month
        start = start.replace(day=1)

    # Generate date range with the specified frequency
    # Replace 'BD' with 'B' for business days and 'M' with 'MS' for month-start
    pandas_freq = freq.replace('BD', 'B')
    pandas_freq = pandas_freq.replace('M', 'MS')

    dates = pd.date_range(start=start, end=end, freq=pandas_freq)

    # Create pairs of consecutive dates
    if len(dates) <= 1:
        return []

    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def run_portfolio_backtest(
        asset_returns: dict[str, pd.Series],
        start_date: str,
        end_date: str,
        freq: str,
        optimization_method: str = OptimizationMethod.MVO,
        risk_method: str = RiskMethod.LEDOIT_WOLF,
        return_method: str = ReturnMethod.MEAN_HISTORICAL,
        objective: str = ObjectiveFunction.SHARPE,
        l2_constraint: float = None,
        target_return: float = None,
        target_volatility: float = None,
        risk_free_rate: float = 0.0,
        drop_na: bool = False,
        align_date: str = None,
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Performs a portfolio optimization backtest across multiple time periods.

    Note:
        CAPM returns, which aims to be slightly more stable than the default mean historical return
        I think a better option is Ledoit-Wolf shrinkage, which reduces the extreme values in the covariance matrix

    Args:
        asset_returns: Dictionary mapping tickers to historical daily returns
        start_date: Start date for backtest in 'YYYY-MM-DD' format
        end_date: End date for backtest in 'YYYY-MM-DD' format
        freq: Frequency of portfolio rebalancing ('M', 'Q', 'Y')
        optimization_method: Method to use for optimization
        risk_method: Method for risk estimation
        return_method: Method for return estimation
        objective: Optimization objective
        l2_constraint: L2 regularization parameter (for MVO)
        target_return: Target portfolio return (annualized)
        target_volatility: Target portfolio volatility (annualized)
        risk_free_rate: Annualized risk-free rate
        drop_na: Whether to remove rows with missing values
        align_date: Optional date to align all return series
        verbose: Whether to print additional information and plots

    Returns:
        DataFrame with portfolio weights for each asset during each period
    """
    # Combine the returns series into a single DataFrame
    returns = pd.concat(asset_returns.values(), axis=1).dropna(how='all')
    returns.columns = asset_returns.keys()

    # Handle missing return values if specified
    if drop_na:
        returns = returns.dropna()

    # Apply start date alignment if specified
    if align_date:
        returns = returns[returns.index >= pd.to_datetime(align_date, format='%Y-%m-%d')]

    # Get the backtest periods
    periods = get_backtest_iterator(start_date, end_date, freq)

    # Initialize results storage
    results = []

    # Iterate through each backtest period
    for i, (period_start, period_end) in enumerate(periods):
        if verbose:
            print(f"Running backtest {i + 1}/{len(periods)}: {period_start.date()} to {period_end.date()}")

        # Get historical data up to this period
        past_returns = returns[returns.index < period_start]

        # Skip periods with insufficient data
        if past_returns.empty or past_returns.shape[0] < 30:
            if verbose:
                print(f"Skipping period {i + 1} due to insufficient historical data")
            continue

        # Create and run the optimizer
        optimizer = PortfolioOptimizer(
            returns=past_returns,
            optimization_method=optimization_method,
            risk_method=risk_method,
            return_method=return_method,
            objective=objective,
            l2_constraint=l2_constraint,
            target_return=target_return,
            target_volatility=target_volatility,
            risk_free_rate=risk_free_rate,
            verbose=verbose
        )

        weights = optimizer.optimize()

        # Format and store the weights with date as first element
        formatted_date = period_start.strftime('%Y-%m-%d')
        weight_results = [f"{ticker}, {round(weight * 100, 5)}%" for ticker, weight in weights.items()]
        results.append([formatted_date] + weight_results)

    # Create and return the results DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.columns = ['Date'] + [f'Weight_{i}' for i in range(results_df.shape[1] - 1)]

    return results_df
