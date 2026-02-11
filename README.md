# FinBench

FinBench is a collection of tools, datasets and example implementations to evaluate and experiment with models and algorithms in the financial domain (time-series forecasting, ranking, portfolio simulation, factor modeling, etc.). The repository aims to provide a reproducible foundation for research, benchmarking and rapid prototyping in quantitative finance and financial machine learning.

## Key features

- Structured datasets and data loaders for common financial tasks.
- Preprocessing and feature engineering utilities (technical indicators, rolling statistics, factor calculation).
- Baseline model implementations across multiple tasks (classification, regression, ranking, portfolio optimization).
- Evaluation and backtesting tools for reproducible experiment comparison.
- Per-model example training scripts and requirements to reproduce results.

## Repository layout (high level)

- `Classification/` — Multiple classification model implementations and training scripts (e.g., Adv-ALSTM, CNNPred, DGDNN, HGTAN, MAN-SF, THGNN).
- `Ranking/` — Ranking models and related training pipelines.
- `Regression/` — Regression and forecasting models (FinFormer, FactorVAE, HIST and more).
- `Evaluation/` — Evaluation and backtesting utilities, evaluation scripts and configuration templates.

Note: Each model implementation includes their own `requirements.txt` and example training scripts.

## Quick start

1. Clone the repository:

```
   git clone https://github.com/softlab-unimore/finbench.git
   cd finbench
```

2. Create and activate a Python virtual environment for each model and Evaluation package.


3. Install dependencies.

   - Global evaluation tools (used by `Evaluation/`):

   ```
      pip install -r Evaluation/requirements.txt
   ```

   - Per-model dependencies: each model folder (for example `Classification/Adv-ALSTM/`) contains a `requirements.txt` with the packages needed for training and evaluation of that model. Follow the instructions in each model folder.


## Running evaluation and examples

- Evaluation: `Evaluation/main.py` and `Evaluation/evaluation.py` provide mechanisms to extract data and compute metrics on model predictions. 

    
- Model training: all the models provide a `train.py` (or `train_2D.py` / `train_3D.py`) script inside their folder. Typical usage (adjust per-model arguments):

Check the docs or the training script in the model folder for model-specific flags and data requirements. 


Almost all models were tested with **Python 3.10**; however, some exceptions (e.g., Adv-ASLTM) required different Python versions due to library compatibility issues.


## License

This repository includes a `LICENSE` file at the project root. Review it for terms and conditions before using the code in production.

