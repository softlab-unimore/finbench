import argparse
import os
import re
import logging
import pandas as pd
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----------------------------
# Helper
# ----------------------------
def _parse_percent_or_none(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ----------------------------
# EOY returns aggregation
# ----------------------------
FILENAME_EOY_RE = re.compile(r"eoy_returns_sl(?P<sl>\d+)_pl(?P<pl>\d+)_seed(?P<seed>\d+)\.csv")

def extract_eoy_returns(base_dir: str = "evaluation_result",
                        top_k: int = 10,
                        short_k: int = 0,
                        output_prefix: str = "aggregated_eoy_returns") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    topk_shortx = f"top{top_k}_short{short_k}"

    for universo in os.listdir(base_dir):
        universo_path = os.path.join(base_dir, universo)
        if not os.path.isdir(universo_path):
            continue

        ts_path = os.path.join(universo_path, topk_shortx)
        if not os.path.isdir(ts_path):
            continue

        for tipologia in os.listdir(ts_path):
            tipologia_path = os.path.join(ts_path, tipologia)
            if not os.path.isdir(tipologia_path):
                continue

            for modello in os.listdir(tipologia_path):
                modello_path = os.path.join(tipologia_path, modello)
                if not os.path.isdir(modello_path):
                    continue

                for fname in os.listdir(modello_path):
                    match = FILENAME_EOY_RE.match(fname)
                    if not match:
                        continue

                    sl = int(match.group("sl"))
                    pl = int(match.group("pl"))
                    seed = int(match.group("seed"))

                    fpath = os.path.join(modello_path, fname)
                    try:
                        df = _safe_read_csv(fpath)
                    except Exception as e:
                        logging.warning(f"Errore lettura {fpath}: {e}")
                        continue

                    if "date" not in df.columns or len(df.columns) < 2:
                        logging.warning(f"Formato CSV inatteso: {fpath}")
                        continue

                    df["anno"] = pd.to_datetime(df["date"]).dt.year
                    value_col = df.columns[1]

                    for _, r in df.iterrows():
                        rows.append({
                            "universo": universo,
                            "modello": modello,
                            "tipologia": tipologia,
                            "topk": top_k,
                            "shortx": short_k,
                            "sl": sl,
                            "pl": pl,
                            "seed": seed,
                            "anno": int(r["anno"]),
                            "valore": r[value_col]
                        })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        logging.info("Nessun dato EOY trovato.")
        return df_all

    df_agg = (
        df_all
        .groupby([
            "universo", "modello", "tipologia",
            "topk", "shortx", "sl", "pl", "anno"
        ], as_index=False)
        .agg(valore=("valore", "mean"))
    )

    df_final = (
        df_agg
        .pivot_table(
            index=[
                "universo", "modello", "tipologia",
                "topk", "shortx", "sl", "pl"
            ],
            columns="anno",
            values="valore"
        )
        .reset_index()
    )

    out_csv = f"{output_prefix}_{top_k}_{short_k}.csv"
    df_final.to_csv(out_csv, index=False)
    logging.info(f"Creato file: {out_csv} ({len(df_final)} righe)")
    return df_final

# ----------------------------
# Metrics extraction
# ----------------------------
TARGET_METRICS = [
    "Cumulative Return",
    "CAGR﹪",
    "Sharpe",
    "Max Drawdown",
    "Volatility (ann.)",
    "Sortino/√2",
    "Avg. Drawdown"
]
METRICS_FILENAME_RE = re.compile(r"metrics_sl(\d+)_pl(\d+)_seed(\d+)\.csv")

def extract_metrics(root_dir: str = "evaluation_result",
                    mean_output: str = "summary_metrics_mean_over_seeds.csv") -> pd.DataFrame:
    rows = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            match = METRICS_FILENAME_RE.match(file)
            if not match:
                continue

            sl, pl, seed = match.groups()
            file_path = os.path.join(root, file)

            parts = os.path.normpath(file_path).split(os.sep)
            try:
                idx = parts.index(os.path.basename(root_dir))
                universo = parts[idx + 1]
                top_short = parts[idx + 2]
                task_type = parts[idx + 3]
                modello = parts[idx + 4]
                m = re.match(r"top(\d+)_short(\d+)", top_short)
                topk, shortk = m.groups() if m else (None, None)
            except Exception:
                logging.warning(f"Skipping path non conforme: {file_path}")
                continue

            try:
                df = _safe_read_csv(file_path)
                metrics_dict = {}
                for metric in TARGET_METRICS:
                    val = df.loc[df.iloc[:, 0] == metric, df.columns[1]]
                    raw = val.values[0] if len(val) > 0 else None
                    metrics_dict[metric] = _parse_percent_or_none(raw)
            except Exception as e:
                logging.warning(f"Errore lettura {file_path}: {e}")
                continue

            row = {
                "universo": universo,
                "modello": modello,
                "task_type": task_type,
                "topk": topk,
                "shortk": shortk,
                "seed": int(seed),
                "sl": int(sl),
                "pl": int(pl),
                **metrics_dict
            }
            rows.append(row)

    final_df = pd.DataFrame(rows)
    if final_df.empty:
        logging.info("Nessun file metrics trovato.")
        return final_df

    int_cols = ["seed", "sl", "pl"]
    for c in ["topk", "shortk"]:
        if c in final_df.columns:
            try:
                final_df[c] = final_df[c].astype(float).astype('Int64')
            except Exception:
                pass
    final_df[int_cols] = final_df[int_cols].astype(int)

    # mean over seeds
    GROUP_COLS = [
        "universo",
        "modello",
        "task_type",
        "topk",
        "shortk",
        "sl",
        "pl"
    ]
    metric_cols = TARGET_METRICS
    final_df[metric_cols] = final_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    grouped_df = (
        final_df
        .groupby(GROUP_COLS, as_index=False)[metric_cols]
        .mean()
    )
    grouped_df.to_csv(mean_output, index=False)
    logging.info(f"Creato {mean_output} con {len(grouped_df)} righe")
    return final_df

# ----------------------------
# Quintile metrics extraction
# ----------------------------
TARGET_METRICS_QUINT = ["CAGR﹪"]
QUINT_FILENAME_RE = re.compile(r"metrics_q(\d+)_sl(\d+)_pl(\d+)_seed(\d+)\.csv")

def extract_metrics_quintile(root_dir: str = "evaluation_result_quintile",
                             mean_output: str = "summary_metrics_quintile_mean_over_seeds.csv") -> pd.DataFrame:
    rows = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            match = QUINT_FILENAME_RE.match(file)
            if not match:
                continue
            quintile, sl, pl, seed = match.groups()
            file_path = os.path.join(root, file)

            parts = os.path.normpath(file_path).split(os.sep)
            try:
                idx = parts.index(os.path.basename(root_dir))
                universo = parts[idx + 1]
                task_type = parts[idx + 2]
                modello = parts[idx + 3]
            except Exception:
                logging.warning(f"Skipping path non conforme: {file_path}")
                continue

            try:
                df = _safe_read_csv(file_path)
                metrics_dict = {}
                for metric in TARGET_METRICS_QUINT:
                    val = df.loc[df.iloc[:, 0] == metric, df.columns[1]]
                    raw = val.values[0] if len(val) > 0 else None
                    metrics_dict[metric] = _parse_percent_or_none(raw)
            except Exception as e:
                logging.warning(f"Errore lettura {file_path}: {e}")
                continue

            row = {
                "universo": universo,
                "modello": modello,
                "task_type": task_type,
                "sl": int(sl),
                "pl": int(pl),
                "seed": int(seed),
                "quintile": int(quintile),
                **metrics_dict
            }
            rows.append(row)

    final_df = pd.DataFrame(rows)
    if final_df.empty:
        logging.info("Nessun file quintile trovato.")
        return final_df

    GROUP_COLS = ["universo", "modello", "task_type", "sl", "pl"]
    final_df[TARGET_METRICS_QUINT] = final_df[TARGET_METRICS_QUINT].apply(pd.to_numeric, errors="coerce")
    mean_df = final_df.groupby(GROUP_COLS + ["quintile"], as_index=False)[TARGET_METRICS_QUINT].mean()

    pivot_df = mean_df.pivot_table(
        index=GROUP_COLS,
        columns="quintile",
        values=TARGET_METRICS_QUINT[0]
    ).reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns=lambda x: f"q{x}" if str(x).isdigit() else x)
    pivot_df.to_csv(mean_output, index=False)
    logging.info(f"Creato {mean_output} con {len(pivot_df)} righe")
    return final_df

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(prog="metrics_extraction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eoy = sub.add_parser("eoy", help="Aggrega eoy returns")
    p_eoy.add_argument("--base_dir", default="evaluation_result")
    p_eoy.add_argument("--top_k", type=int, required=True)
    p_eoy.add_argument("--short_k", type=int, required=True)
    p_eoy.add_argument("--output_prefix", default="aggregated_eoy_returns")

    p_metrics = sub.add_parser("metrics", help="Estrai metriche standard")
    p_metrics.add_argument("--root_dir", default="evaluation_result")
    p_metrics.add_argument("--mean_output", default="summary_metrics_mean_over_seeds.csv")

    p_quint = sub.add_parser("quintile", help="Estrai metriche quintile")
    p_quint.add_argument("--root_dir", default="evaluation_result_quintile")
    p_quint.add_argument("--mean_output", default="summary_metrics_quintile_mean_over_seeds.csv")

    p_all = sub.add_parser("all", help="Esegui tutte le pipeline")
    p_all.add_argument("--top_k", type=int, required=True)
    p_all.add_argument("--short_k", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "eoy":
        extract_eoy_returns(args.base_dir, args.top_k, args.short_k, args.output_prefix)
    elif args.cmd == "metrics":
        extract_metrics(args.root_dir, args.output, args.mean_output)
    elif args.cmd == "quintile":
        extract_metrics_quintile(args.root_dir, args.output, args.mean_output)

if __name__ == "__main__":
    main()

