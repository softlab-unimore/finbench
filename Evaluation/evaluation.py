import json
import os
import pickle
from argparse import ArgumentParser

import pandas as pd

from Evaluation.convert_predictions import convert_classification_preds, convert_daily_to_cumulative_returns
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


def filter_valid_dates(pred_paths: list[str], args: ArgumentParser):
    for elem in pred_paths:
        with open(elem, 'rb') as f:
            results = pickle.load(f)

        # case 1: string list -> each list element is a ticker
        tickers = results['tickers']
        if isinstance(tickers[0], str):
            common = set(tickers)

        # case 2: list of list -> each list element is a list of tickers for a date
        else:
            max_len = max(len(t) for t in tickers)
            k = max(1, int(0.1 * max_len))
            mask = [len(i) > k for i in tickers]
            results['preds'] = [x for x, m in zip(results['preds'], mask) if m]
            results['labels'] = [x for x, m in zip(results['labels'], mask) if m]
            results['last_date'] = [x for x, m in zip(results['last_date'], mask) if m]
            results['pred_date'] = [x for x, m in zip(results['pred_date'], mask) if m]
            results['tickers'] = [x for x, m in zip(results['tickers'], mask) if m]

            common = set(results['tickers'][0])
            for t in tickers[1:]:
               common &= set(t)

        common_len = len(common)

        if args.top_k != 0 and args.short_k != 0:
            limit = args.top_k + args.short_k
        elif args.top_k != 0 and args.short_k == 0:
            limit = args.top_k
        elif args.top_k == 0 and args.short_k != 0:
            limit = args.short_k

        if common_len < limit:
            raise ValueError(f"Not enough valid tickers in {elem}: {common_len} < {limit}")
        else:
            print(f"{elem}: {common_len} valid tickers")

        with open(elem, 'wb') as f:
            pickle.dump(results, f)


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


def demo_portfolio_evaluation(args):
    # Config params
    price_path = f'./data/{args.universe}/{args.universe}.csv'
    out_dir = f'./evaluation_result/{args.universe}/top{args.top_k}_short{args.short_k}/{args.type}/{args.model}'
    os.makedirs(out_dir, exist_ok=True)

    # Define prediction paths
    pred_paths = []
    for year, model_conf in args.configurations_by_year.items():
        pred_paths.append(
            f'../{args.type}/{args.model}/results/{args.universe}/{model_conf}/{args.seed}/y{year}/results_sl{args.sl}_pl{args.pl}.pkl'
        )

    print("Preds_paths: " + str(pred_paths))

    # Preprocessing: filter dates with enough tickers
    filter_valid_dates(pred_paths, args)

    # Preprocessing: convert classification predictions
    if args.type == 'Classification':
        pred_paths = convert_classification_preds(pred_paths, args.model)

    if args.type == 'Regression' and args.model == 'D-Va':
        pred_paths, args.pl = convert_daily_to_cumulative_returns(pred_paths, args.sl, args.pl)

    # Create portfolio history
    print("Creating portfolio history...")
    pf_history = create_long_short_portfolio_history(
        pred_paths,
        top_k=args.top_k,
        short_k=args.short_k,
        start_date='2020-01-01',
        end_date='2024-12-31',
        freq=f'{args.pl}B',
        verbose=True
    )

    with open(os.path.join(out_dir, f"pf_history_sl{args.sl}_pl{args.pl}_seed{args.seed}.pkl"), "wb") as f:
        pickle.dump(pf_history, f)

    # Download/load asset prices
    print("Loading asset prices...")
    asset_prices = download_asset_prices(pf_history, price_path)

    # Compute portfolio daily returns
    print("Computing portfolio returns...")
    pf_returns = portfolio_daily_returns(pf_history, asset_prices)
    pf_cum_rets = (1 + pf_returns).cumprod() * 100000

    # Compute strategy metrics
    metrics = qs.reports.metrics(pf_returns, mode='full', rf=0.0, compounded=True, display=False)
    metrics.to_csv(os.path.join(out_dir, f"metrics_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv"))

    # Compute EOY returns
    eoy_returns = pf_returns.resample("YE").apply(qs.stats.comp)
    eoy_returns.to_csv(os.path.join(out_dir, f"eoy_returns_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv"))

    print("Save results in:", out_dir)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--universe', type=str, default='sx5e', help='Universe')
    args.add_argument('--model', type=str, default='MASTER', help='Model name')
    args.add_argument('--configuration_by_year', type=json.loads, required=True, help="Dictionary year -> configuration (JSON format)")
    args.add_argument('--seed', type=int, default=0, help='Random seed')
    args.add_argument('--type', type=str, default='Regression', help='Type (Regression/Classification/Ranking')
    args.add_argument('--sl', type=int, default=5, help='sequence length')
    args.add_argument('--pl', type=int, default=1, help='pred length')
    args.add_argument('--top_k', type=int, default=10, help='top_k')
    args.add_argument('--short_k', type=int, default=0, help='short_k')

    args = args.parse_args()

    demo_portfolio_evaluation(args)

    # evaluate_benchmarks(
    #     benchmarks=['SXXP.INDX', 'GSPC.INDX', 'DJI.INDX', 'SX5E.INDX', 'NDX.INDX'],
    #     start_date='2020-01-01',
    #     end_date='2024-12-31',
    #     api_token=load_config('config.json').get('eodhd')
    # )
