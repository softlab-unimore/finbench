import os
import pickle
from argparse import ArgumentParser
from types import SimpleNamespace

from Evaluation.convert_predictions import convert_classification_preds, convert_daily_to_cumulative_returns
from Evaluation.portfolio.returns import portfolio_daily_returns
from Evaluation.portfolio.transforms import create_quintile_portfolios_history
from Evaluation.utils.evaluation import download_asset_prices, load_best_results, extract_yearly_config


def demo_portfolio_evaluation_quintile(args, qs=None):
    # Config params
    price_path = f'./data/{args.universe}/{args.universe}.csv'
    out_dir = f'./evaluation_result_quintile/{args.universe}/{args.type}/{args.model_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Define prediction paths
    pred_paths = []
    for year, model_conf in args.configurations_by_year.items():
        pred_paths.append(
            f'../{args.type}/{args.model}/results/{args.universe}/{model_conf}/{args.seed}/y{year}/results_sl{args.sl}_pl{args.pl}.pkl'
        )

    print("Preds_paths: " + str(pred_paths))

    # Preprocessing: convert classification predictions
    if args.type == 'Classification':
        pred_paths = convert_classification_preds(pred_paths, args.model)

    if args.type == 'Regression' and args.model == 'D-Va':
        pred_paths, args.pl = convert_daily_to_cumulative_returns(pred_paths, args.sl, args.pl)

    # Create portfolio history
    print("Creating portfolio history...")
    quintile_histories = create_quintile_portfolios_history(
        pred_paths,
        start_date=f'{args.start_year}-01-01',
        end_date='2024-12-31',
        freq=f'{args.pl}B',
        verbose=True
    )

    for q, pf_history in quintile_histories.items():
        print(f"Processing quintile {q}...")

        with open(os.path.join(out_dir, f"pf_history_q{q}_sl{args.sl}_pl{args.pl}_seed{args.seed}.pkl"), "wb") as f:
            pickle.dump(pf_history, f)

        print("Loading asset prices...")
        asset_prices = download_asset_prices(pf_history, price_path)

        print("Computing portfolio returns...")
        pf_returns = portfolio_daily_returns(pf_history, asset_prices)

        metrics = qs.reports.metrics(
            pf_returns,
            mode='full',
            rf=0.0,
            compounded=True,
            display=False
        )
        metrics.to_csv(os.path.join(out_dir, f"metrics_q{q}_sl{args.sl}_pl{args.pl}_seed{args.seed}.csv"))

        eoy_returns = pf_returns.resample("YE").apply(qs.stats.comp)
        eoy_returns.to_csv(os.path.join(out_dir, f"eoy_returns_q{q}_sl{args.sl}_pl{args.pl}_seed{seed}.csv"))


def build_runtime_args(args, year_to_config):
    return SimpleNamespace(
        universe=args.universe,
        model=args.model,
        type=args.type,
        seed=args.seed,
        sl=args.sl,
        pl=args.pl,
        configurations_by_year=year_to_config
    )


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--universe', type=str, default='sx5e', help='Universe')
    args.add_argument('--model', type=str, default='MASTER', help='Model name')
    args.add_argument('--seed', type=int, default=0, help='Random seed')
    args.add_argument('--type', type=str, default='Regression', help='Type (Regression/Classification/Ranking')
    args.add_argument('--sl', type=int, default=5, help='sequence length')
    args.add_argument('--pl', type=int, default=1, help='pred length')
    args.add_argument('--start_year', type=int, default=2021, help='start year')

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
        demo_portfolio_evaluation_quintile(args)