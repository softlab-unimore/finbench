# Adv-ALSTM model

This folder contains an implementation of the Adv-ALSTM model for financial time-series classification.

Requirements
- Python 3.8 (recommended)

Conda-based installation (Windows PowerShell)
1. Create and activate a conda environment with Python 3.8:
```powershell
conda create -n adv-alstm python=3.8 -y
conda activate adv-alstm
```
2. Upgrade pip and install dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```
Note on TensorFlow: `requirements.txt` currently pins `tensorflow==1.8.0`.

Training example
Run training with default-ish parameters (PowerShell):
```powershell
python train.py --data_path ../../Evaluation/data --universe <universe> 
```
Main command-line arguments (see `train.py` for full list):
- `--data_path`: path to data folder (default `./data`)
- `--universe`: dataset universe (default `sp500`)
- `--start_date`, `--end_train_date`, `--end_valid_date`, `--end_date`: train/val/test date ranges
- `--seq`: history sequence length 
- `--pred_len`: prediction length
- `--alpha_l2`: L2 regularization weight
- `--beta_adv`: adversarial loss weight 
- `--epsilon_adv`: adversarial perturbation magnitude
- `--unit`: LSTM hidden units 
- `--seed`: random seed for reproducibility


Outputs
- Validation/test metrics are saved under: `results2D/<universe>/<model>/<seed>/y<year>` or `results3D/<universe>/<model>/<seed>/y<year>`.
- Example output files: `metrics_sl{seq}_pl{pred_len}.json` and `results_sl{seq}_pl{pred_len}.pkl` saved in the metrics path.

