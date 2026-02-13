import json
import os
import pickle
from argparse import ArgumentParser
from types import SimpleNamespace

from Evaluation.convert_predictions import convert_classification_preds, convert_daily_to_cumulative_returns
from Evaluation.utils.evaluation import download_asset_prices, load_best_results, extract_yearly_config

from portfolio.transforms import create_long_short_portfolio_history
from portfolio.returns import portfolio_daily_returns

import quantstats as qs


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


def build_runtime_args(args, year_to_config):
    return SimpleNamespace(
        universe=args.universe,
        model=args.model,
        type=args.type,
        seed=args.seed,
        sl=args.sl,
        pl=args.pl,
        top_k=args.top_k,
        short_k=args.short_k,
        start_year=args.initial_year,
        configurations_by_year=year_to_config
    )


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--universe', type=str, default='sx5e', help='Universe', required=True)
    args.add_argument('--model', type=str, default='MASTER', help='Model name', required=True)
    args.add_argument("--initial_year", type=int, default=2021, required=True)
    args.add_argument('--seed', type=int, default=0, help='Random seed')
    args.add_argument('--type', type=str, default='Regression', choices=['Regression', 'Classification', 'Ranking'], help='Prediction type')
    args.add_argument('--sl', type=int, default=5, help='sequence length')
    args.add_argument('--pl', type=int, default=1, help='pred length')
    args.add_argument('--top_k', type=int, default=10, help='top_k')
    args.add_argument('--short_k', type=int, default=0, help='short_k')
    args = args.parse_args()

    BASE_DIRS = f"../{args.type}"
    best_results = load_best_results(BASE_DIRS, args.model, args.universe)

    print(f'\n==== {args.model} ====')

    best_config = best_results[args.model][args.universe]

    for seed, year_to_config in extract_yearly_config(best_config, args):
        print(
            f"[RUN] {args.model} | {args.universe} | seed={seed} | sl={args.sl} pl={args.pl} | "
            f"years={list(year_to_config.keys())}"
        )

        runtime_args = build_runtime_args(args, year_to_config)
        demo_portfolio_evaluation(runtime_args)

    # evaluate_benchmarks(
    #     benchmarks=['SXXP.INDX', 'GSPC.INDX', 'DJI.INDX', 'SX5E.INDX', 'NDX.INDX'],
    #     start_date='2020-01-01',
    #     end_date='2024-12-31',
    #     api_token=load_config('config.json').get('eodhd')
    # )
