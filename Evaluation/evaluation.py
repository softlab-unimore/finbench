import pandas as pd

from utils.storage import load_config, read_stock_dataset
from utils.eodhd import download_multi

from portfolio.transforms import create_long_short_portfolio_history
from portfolio.returns import portfolio_daily_returns

import quantstats as qs


def evaluate_benchmarks(benchmarks: list[str], start_date: str, end_date: str, api_token: str) -> pd.DataFrame:
    """ Evaluate performance metrics for multiple benchmarks over a specified period """

    mkeys = ['Cumulative Return', 'CAGR﹪', 'Sharpe', 'Sortino', 'Volatility (ann.)', 'Max Drawdown', 'Avg. Drawdown']

    # Download data
    asset_returns, errors = download_multi(benchmarks, api_token=api_token, data_type='returns', p=4)
    if errors:
        raise ValueError(f"Failed to download {len(errors)} benchmarks: {errors}")

    results = {}

    for benchmark, returns in asset_returns.items():
        # returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
        # Filter by date range
        returns = returns.loc[start_date:end_date]

        if returns.empty:
            print(f"Warning: No data for {benchmark} in specified date range")
            continue

        # Extract core metrics
        metrics = qs.reports.metrics(returns, mode='full', rf=0.0, compounded=True, display=False)['Strategy'].to_dict()

        benchmark_results = {metric: metrics.get(metric, None) for metric in mkeys}

        # Calculate end-of-year returns
        eoy_returns = returns.resample('YE').apply(qs.stats.comp)
        eoy_returns.index = eoy_returns.index.year
        benchmark_results.update(eoy_returns.to_dict())

        results[benchmark] = benchmark_results

    # Convert to DataFrame and transpose
    df = pd.DataFrame(results).T

    return df


def download_asset_prices(pf_history: list[dict], price_path: str = None) -> dict[str, pd.DataFrame]:
    userdata = load_config('config.json')
    asset_prices = {}

    # Try to load cached prices if path provided
    if price_path:
        prices = read_stock_dataset(price_path)
        max_date = prices.index.get_level_values('date').max()

        # Only use cached data if it covers the full date range
        if max_date > pf_history[-1]['test_end']:
            asset_prices = {k: df.droplevel(0) for k, df in prices.groupby('instrument')}
            print(f"Loaded {len(asset_prices)} assets from cache")

    # Identify missing tickers
    pf_tickers = {ticker for snp in pf_history for ticker in snp['tickers']}
    missing_tickers = list(pf_tickers - set(asset_prices.keys()))

    # Download missing prices
    if missing_tickers:
        print(f"Downloading {len(missing_tickers)} missing tickers...")
        missing_prices, errors = download_multi(
            missing_tickers,
            api_token=userdata.get('eodhd'),
            data_type='price',
            verbose=True,
            p=4
        )
        if errors:
            raise ValueError(f"{len(errors)} tickers failed to download: {errors}")

        asset_prices.update(missing_prices)

    return asset_prices


def demo_portfolio_evaluation():
    # Config params
    price_path = './data/dataset/sp500/sp500.csv'
    model_name, sl, pl = 'MASTER_sp500_alpha158', 60, 20
    years = range(2020, 2024 + 1)
    top_k = 10
    short_k = 0

    # Define prediction paths
    pred_paths = [f'./data/preds/master_sl{sl}_pl{pl}/{year}_results_sl{sl}_pl{pl}.pkl' for year in years]

    # Create portfolio history
    print("Creating portfolio history...")
    pf_history = create_long_short_portfolio_history(
        pred_paths,
        top_k=top_k,
        short_k=short_k,
        start_date='2020-01-01',
        end_date='2024-12-31',
        freq=f'{pl}B',
        verbose=True
    )

    # Download/load asset prices
    print("Loading asset prices...")
    asset_prices = download_asset_prices(pf_history, price_path)

    # Compute portfolio daily returns
    print("Computing portfolio returns...")
    pf_returns = portfolio_daily_returns(pf_history, asset_prices)
    pf_cum_rets = (1 + pf_returns).cumprod() * 100000

    # Compute strategy metrics
    metrics = qs.reports.metrics(pf_returns, mode='full', rf=0.0, compounded=True, display=False)

    # Compute EOY returns
    eoy_returns = pf_returns.resample("YE").apply(qs.stats.comp)

    print('Hello World!')


if __name__ == '__main__':
    demo_portfolio_evaluation()
    evaluate_benchmarks(
        benchmarks=['SXXP.INDX', 'GSPC.INDX', 'DJI.INDX', 'SX5E.INDX', 'NDX.INDX'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        api_token=load_config('config.json').get('eodhd')
    )
