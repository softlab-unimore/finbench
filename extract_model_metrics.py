import os
import re
import json
import csv
import argparse
from collections import defaultdict

TYPE_CONFIG = {
    "Classification": ("F1", True, ["Accuracy", "F1", "MCC", "Precision", "Recall"]),
    "Regression": ("MAE", False, ["MSE", "MAE", "RMSE", "R2"]),
    "Ranking": ("RankICIR", True, ["IC", "RankIC", "ICIR", "RankICIR"]),
}

VAL_RE = re.compile(r"val_.*_sl(\d+)_pl(\d+)\.json")
TEST_RE = re.compile(r".*_sl(\d+)_pl(\d+)\.json")
DEFAULT_SL_PL_ORDER = ["sl5_pl1", "sl20_pl5", "sl60_pl20"]


def find_best_for_model(base_path, model, type_name):
    """Scan results for a single model and write best_results.json in the model folder.

    base_path: path to the folder that contains models for the given type, e.g. ./Classification
    model: name of the model folder inside base_path
    type_name: one of Classification/Regression/Ranking
    """
    if type_name not in TYPE_CONFIG:
        raise RuntimeError("Unknown type: {}".format(type_name))

    metric_name, maximize, _ = TYPE_CONFIG[type_name]

    if 'CNNPred2D' in model or 'CNNPred3D' in model:
        model_path = os.path.join(base_path, 'CNNPred')
    else:
        model_path = os.path.join(base_path, model)

    if not os.path.isdir(model_path):
        raise RuntimeError("Model '{}' not found in {}".format(model, base_path))

    if 'CNNPred2D' in model or 'CNNPred3D' in model:
        res_folder = 'results2D' if 'CNNPred2D' in model else 'results3D'
        results_path = os.path.join(model_path, res_folder)
    else:
        results_path = os.path.join(model_path, "results")

    if not os.path.isdir(results_path):
        raise RuntimeError("No 'results' folder for model '{}'".format(model))

    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )
    )

    for universe in os.listdir(results_path):
        universe_path = os.path.join(results_path, universe)
        if not os.path.isdir(universe_path):
            continue

        # gather seeds present across configs
        seeds = set(
            s
            for cfg in os.listdir(universe_path)
            if os.path.isdir(os.path.join(universe_path, cfg))
            for s in os.listdir(os.path.join(universe_path, cfg))
        )

        for seed in seeds:
            years = set(
                y
                for cfg in os.listdir(universe_path)
                for s in os.listdir(os.path.join(universe_path, cfg))
                if s == seed
                for y in os.listdir(os.path.join(universe_path, cfg, s))
            )

            for year in years:
                best_per_slpl = {}

                for config in os.listdir(universe_path):
                    config_path = os.path.join(universe_path, config, seed, year)
                    if not os.path.isdir(config_path):
                        continue

                    for fname in os.listdir(config_path):
                        match = VAL_RE.match(fname)
                        if not match:
                            continue

                        sl, pl = match.groups()
                        key = "sl{}_pl{}".format(sl, pl)

                        with open(os.path.join(config_path, fname), 'r') as f:
                            try:
                                data = json.load(f)
                            except Exception:
                                continue

                        score = data.get(metric_name)
                        if score is None:
                            continue

                        if key not in best_per_slpl:
                            take_better = True
                        else:
                            prev = best_per_slpl[key]['metrics'].get(metric_name)
                            if prev is None:
                                take_better = True
                            else:
                                if maximize:
                                    take_better = score > prev
                                else:
                                    take_better = score < prev

                        if take_better:
                            name_test_res = fname.replace('val_', '')
                            test_metrics = {}
                            test_path = os.path.join(config_path, name_test_res)
                            if os.path.exists(test_path):
                                with open(test_path, 'r') as f:
                                    try:
                                        test_metrics = json.load(f)
                                    except Exception:
                                        test_metrics = {}

                            best_per_slpl[key] = {
                                'config': config,
                                'metrics': test_metrics
                            }

                if best_per_slpl:
                    results[model][universe][seed][year] = best_per_slpl

    out_file = os.path.join(base_path, model, 'best_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("[{}] results saved to {}".format(model, out_file))


# ---------------- CSV extraction ----------------

def extract_metrics_recursive(d, sl_pl_key=None, metrics_order=None):
    """Return dict {sl_pl: {metric: value}} for leaf nodes that contain 'metrics'."""
    result = {}
    if isinstance(d, dict):
        if 'metrics' in d and isinstance(d['metrics'], dict):
            key = sl_pl_key or ''
            row = {}
            for m in (metrics_order or []):
                row[m] = d['metrics'].get(m, '')
            return {key: row}
        else:
            for k, v in d.items():
                new_key = k if 'sl' in k and 'pl' in k else sl_pl_key
                res = extract_metrics_recursive(v, new_key, metrics_order)
                result.update(res)
    return result


def process_best_results_file(json_path, sl_pl_order=None, metrics_order=None):
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            print('Invalid JSON:', json_path)
            return

    sl_pl_order = sl_pl_order or DEFAULT_SL_PL_ORDER
    metrics_order = metrics_order or []

    # Build rows keyed by (year, seed, universe)
    all_rows = {}
    for top_key, top_val in data.items():
        for universe_key, universe_val in top_val.items():
            for seed_key, seed_val in universe_val.items():
                for year_key, year_val in seed_val.items():
                    metrics_dict = extract_metrics_recursive(year_val, None, metrics_order)
                    all_rows[(year_key, seed_key, universe_key)] = metrics_dict

    csv_path = os.path.join(os.path.dirname(json_path), 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        header = ['Year', 'Seed', 'Universe']
        for sl_pl in sl_pl_order:
            for m in metrics_order:
                header.append(sl_pl + '_' + m)
        writer.writerow(header)

        for (year, seed, universe), metrics_dict in sorted(all_rows.items(), key=lambda x: (x[0][2], x[0][0], int(x[0][1]) if str(x[0][1]).isdigit() else x[0][1])):
            row = [year, seed, universe]
            for sl_pl in sl_pl_order:
                if sl_pl in metrics_dict:
                    for m in metrics_order:
                        row.append(metrics_dict[sl_pl].get(m, ''))
                else:
                    row += [''] * len(metrics_order)
            writer.writerow(row)

    print('CSV created at', csv_path)


def find_and_process(root_dir, sl_pl_order=None, metrics_order=None):
    """Walk root_dir and process each best_results.json found."""
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file == 'best_results.json':
                process_best_results_file(os.path.join(dirpath, file), sl_pl_order, metrics_order)


# ----------------- CLI ------------------

def main():
    parser = argparse.ArgumentParser(description='finbench tools: find best results and extract CSVs')

    parser.add_argument('--type', choices=list(TYPE_CONFIG.keys()), required=True, help='Type: Classification, Regression or Ranking')
    parser.add_argument('--model', required=True, help='Model folder name (if omitted, run for all models under type)')
    parser.add_argument('--base', help='Base path for the type (default: ./<Type>)', default=None)

    args = parser.parse_args()

    type_name = args.type
    base = args.base if args.base else os.path.join(os.getcwd(), type_name)

    find_best_for_model(base, args.model, type_name)

    if args.model in ['HIST', 'DiscoverPLF', 'FinFormer']:
        DEFAULT_SL_PL_ORDER = ["sl1_pl1", "sl1_pl5", "sl1_pl20"]
    elif args.model == 'D-Va':
        DEFAULT_SL_PL_ORDER = ["sl2_pl2", "sl6_pl6", "sl20_pl20"]

    metrics_order = TYPE_CONFIG[args.type][2]
    find_and_process(args.root, DEFAULT_SL_PL_ORDER, metrics_order)


if __name__ == '__main__':
    main()

