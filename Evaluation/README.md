# Evaluation module
This module provides a unified command-line interface (CLI) to extract, aggregate, and summarize:

- Standard performance metrics  
- End-of-Year (EOY) returns
- Quintile-based metrics  

---
## Requirements
- Python **≥ 3.10**
- Dependencies:
```bash
conda create -n Evaluation python=3.10
pip install -r requirements.txt
```
## Extract metrics

The script to extract metrics exposes a CLI with subcommands:

```bash
python metrics_extraction.py <command> [options]
```
A command is mandatory! The available commands are:
- `eoy`: Extract EOY returns from the specified directory.
- `metrics`: Extract standard performance metrics from the specified directory.
- `quintile`: Extract quintile-based metrics from the specified directory.

Options:
- `--root_dir`: Directory containing metrics return files (default: `evaluation_result`).
- `--mean_output`: Output CSV file for aggregated metrics returns (default: `summary_metrics_mean_over_seeds.csv`).

Options (for `eoy` command):
- `--base_dir`: Directory containing EOY return files (default: `evaluation_result`).
- `--top_k`: Filter by top-k (default: None - required).
- `--short_x`: Filter by short-x (default: None - required).
- `--output_prefix`: Output CSV file for aggregated EOY returns (default: `aggregated_eoy_returns.csv`).

The output is a CSV file with aggregated EOY returns based on the specified filters. `aggregated_eoy_returns_<top_k>_<short_k>.csv`

Example:
```bash
python metrics_extraction.py eoy --top_k 10 --short_k 10
```

The results extracted by this module follow this specific directory structure for both standard and quintile evaluations. 

#### Standard Evaluation Results
```bash
Evaluation
  └──  evaluation_result/
    └── <universe>/
        └── top<k>_short<x>/
            └── <task_type>/
                └── <model>/
                    ├── metrics_sl<sl>_pl<pl>_seed<seed>.csv
                    └── eoy_returns_sl<sl>_pl<pl>_seed<seed>.csv
```
#### Quintile Evaluation Results
```bash
Evaluation
  └── evaluation_result_quintile/
    └── <universe>/
        └── <task_type>/
            └── <model>/
                └── metrics_q<q>_sl<sl>_pl<pl>_seed<seed>.csv

```



The output is a CSV file with aggregated metrics returns. 

1. `metrics`: `summary_metrics_mean_over_seeds.csv` contains the mean of the extracted metrics across different seeds for each model and configuration. Extracted metrics:
    - `Cumulative Return`
    - `CAGR﹪`
    - `Sharpe`
    - `Max Drawdown`
    - `Volatility (ann.)`
    - `Sortino/√2`
    - `Avg. Drawdown`


2. `eoy`: `eoy_returns_sl<sl>_pl<pl>_seed<seed>.csv` contains the aggregated EOY returns based on the specified filters (top-k and short-x).


3. `quintile`: `summary_metrics_quintile_mean_over_seeds.csv` contains the mean of the extracted quintile-based metrics across different seeds for each model and configuration. 