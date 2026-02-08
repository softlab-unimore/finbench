import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def filter_constituents_by_date(constituents: pd.DataFrame, test_start_date: str) -> pd.DataFrame:
    """
    Filters a DataFrame of constituents to include only those active on a given test start date

    Args:
        constituents: DataFrame with 'StartDate' and 'EndDate' columns
        test_start_date: The date to check for active constituents (YYYY-MM-DD format)

    Returns:
        A new DataFrame containing only the active constituents.
    """
    if not all(col in constituents.columns for col in ['StartDate', 'EndDate']):
        raise ValueError('The "constituents" DataFrame must contain StartDate and EndDate columns')

    start_dates = pd.to_datetime(constituents['StartDate'])
    end_dates = pd.to_datetime(constituents['EndDate'])
    test_start = pd.to_datetime(test_start_date)

    # Fill missing dates with logical defaults: very early for start dates, a future date for end dates
    start_dates = start_dates.fillna(pd.Timestamp.min)
    end_dates = end_dates.fillna(pd.Timestamp.max)

    is_active = (start_dates < test_start) & (end_dates >= test_start)

    return constituents[is_active].copy()


def generate_fake_data(num_batches=5, batch_size=20, seed=42):
    """
    Generate fake data matching your structure.

    Args:
        num_batches: Number of batches (length of main lists)
        batch_size: Size of each numpy array (should be 20)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys: 'tickers', 'preds', 'labels', 'dates', 'last_seq_date'
    """
    np.random.seed(seed)
    random.seed(seed)

    # Common stock tickers for realistic data
    ticker_pool = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'LYFT',
        'SNAP', 'TWTR', 'PINS', 'ROKU', 'ZM', 'DOCU', 'SHOP', 'SQ',
        'SNOW', 'PLTR', 'COIN', 'RBLX', 'ABNB', 'DASH', 'DDOG', 'CRWD',
        'OKTA', 'VEEV', 'WDAY', 'NOW', 'SPLK', 'TEAM', 'ATLR', 'MDB',
        'NET', 'FSLY', 'ESTC', 'BILL', 'SMAR', 'PLAN', 'GTLB', 'FROG',
        'DOCN', 'LIDR', 'OPEN', 'RKLB', 'SPCE', 'NKLA', 'LCID', 'RIVN'
    ]

    tickers = []
    preds = []
    labels = []
    preds_dates = []
    last_dates = []

    # Base date for generating sequences
    base_date = datetime(2024, 1, 1)

    for batch_idx in range(num_batches):
        # Generate batch of 20 tickers
        batch_tickers = np.array(random.sample(ticker_pool, batch_size))

        # Generate predictions (random values between -1 and 1)
        batch_preds = np.random.uniform(-1, 1, (batch_size, 1))

        # Generate labels (somewhat correlated with predictions + noise)
        batch_labels = batch_preds + np.random.normal(0, 0.3, (batch_size, 1))

        # Generate prediction dates (sequential dates)
        batch_start_date = base_date + timedelta(days=batch_idx * 30)
        batch_pred_dates = batch_start_date.strftime('%Y-%m-%d')

        # Generate last sequence dates (a few days before prediction dates)
        batch_last_dates = (batch_start_date + timedelta(days=-3)).strftime('%Y-%m-%d')

        tickers.append(batch_tickers)
        preds.append(batch_preds)
        labels.append(batch_labels)
        preds_dates.append(batch_pred_dates)
        last_dates.append(batch_last_dates)

    return {
        'tickers': tickers,
        'preds': preds,
        'labels': labels,
        'preds_dates': preds_dates,
        'last_dates': last_dates
    }
