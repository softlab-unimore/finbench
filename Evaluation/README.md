# Evaluation module
This script provides a unified command-line interface (CLI) to extract, aggregate, and summarize:

- End-of-Year (EOY) returns  
- Standard performance metrics  
- Quintile-based metrics  

It is designed for structured quantitative evaluation pipelines, not for interactive usage.

---
## Requirements
- Python **≥ 3.9**
- Dependencies:
```bash
pip install pandas
```
## Expected Directory Structure
The script relies on a strict directory layout. Files that do not match this structure will be skipped.
### Standard Evaluation Results
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
### Quintile Evaluation Results
```bash
Evaluation
  └── 
    evaluation_result_quintile/
    └── <universe>/
        └── <task_type>/
            └── <model>/
                └── metrics_q<q>_sl<sl>_pl<pl>_seed<seed>.csv
```
## Command Line Usage
The script exposes a CLI with subcommands:
```bash
python metrics_extraction.py <command> [options]
```
A command is mandatory! The available commands are:
- `eoy`: Extract EOY returns from the specified directory.
- `metrics`: Extract standard performance metrics from the specified directory.
- `quintile`: Extract quintile-based metrics from the specified directory.

### EOY
Options:
- `--base_dir`: Directory containing EOY return files (default: `evaluation_result`).
- `--top_k`: Filter by top-k (default: None - required).
- `--short_x`: Filter by short-x (default: None - required).
- `--output_prefix`: Output CSV file for aggregated EOY returns (default: `aggregated_eoy_returns.csv`).

The output is a CSV file with aggregated EOY returns based on the specified filters. `aggregated_eoy_returns_<top_k>_<short_k>.csv`

Example:
```bash
python metrics_extraction.py eoy --top_k 10 --short_k 10
```
### METRICS
Options:
- `--root_dir`: Directory containing metrics return files (default: `evaluation_result`).
- `--mean_output`: Output CSV file for aggregated metrics returns (default: `summary_metrics_mean_over_seeds.csv`).

Extracted metrics:
- `Cumulative Return`
- `CAGR﹪`
- `Sharpe`
- `Max Drawdown`
- `Volatility (ann.)`
- `Sortino/√2`
- `Avg. Drawdown`

The output is a CSV file with aggregated metrics returns. `summary_metrics_mean_over_seeds.csv`

Example:
```bash
python metrics_extraction.py metrics
```

### QUINTILE
Options:
- `--root_dir`: Directory containing quintile metrics return files (default: `evaluation_result_quintile`).
- `--mean_output`: Output CSV file for aggregated quintile metrics returns (default: `summary_metrics_quintile_mean_over_seeds.csv`).

The output is a CSV file with aggregated quintile metrics returns. `summary_metrics_quintile_mean_over_seeds.csv`

Example:
```bash
python metrics_extraction.py quintile
```
