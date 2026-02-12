# FinFormer model

This folder contains an implementation of the FinFormer model for financial time-series classification.

Requirements
- Python 3.10 (recommended)

Conda-based installation (Windows PowerShell)
1. Create and activate a conda environment with Python 3.10:
```powershell
conda create -n FinFormer python=3.10-y
conda activate FinFormer
```
2. Upgrade pip and install dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

Training example
Run training with default-ish parameters (PowerShell):
```powershell
python train.py --universe <universe> --train --test
```
Main command-line arguments (see `train.py` for full list):
- `--data_path`: path to data folder
- `--universe`: dataset universe
- `--seq_len`: lookback window length used as model input
- `--pred_len`: number of future steps to predict
- `--start_date`, `--end_train_date`, `--start_valid_date`, `--end_valid_date`, `--start_test_date`, `--end_date`: date boundaries defining the full dataset and the train/validation/test splits
- `--seed`: random seed for reproducibility
- `--hidden_size`: hidden size of the model
- `--snum_head`: number of dynamic spatial attention heads
- `--lr`: learning rate
- `--batch_size`: batch size for training
- `--num_layers`: number of layers in the model


Outputs
- Validation/test metrics are saved under: `results/<universe>/<model>/<seed>/y<year>`.
- Example output files: `metrics_sl{seq}_pl{pred_len}.json` and `results_sl{seq}_pl{pred_len}.pkl` saved in the metrics path.

