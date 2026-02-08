import pickle
import numpy as np
import pandas as pd

from .backtest import get_backtest_iterator


def _validate_ml_results(d: dict) -> None:
    """ Validate the structure and content of prediction results dictionary """
    # Check required keys
    required_keys = ['labels', 'preds', 'tickers', 'last_date', 'pred_date']
    missing_keys = [key for key in required_keys if key not in d]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Validate that tickers, last_date, and pred_date are lists
    if not isinstance(d['tickers'], list):
        raise ValueError("'tickers' must be a list")

    if not isinstance(d['last_date'], list):
        raise ValueError("'last_date' must be a list")

    if not isinstance(d['pred_date'], list):
        raise ValueError("'pred_date' must be a list")

    # Validate predictions and labels structure
    if isinstance(d['preds'], list):
        # Both should be lists when preds is a list
        if not isinstance(d['labels'], list):
            raise ValueError("'labels' must be a list when 'preds' is a list")

        # Check all elements are numpy arrays
        non_array_preds = [i for i, arr in enumerate(d['preds']) if not isinstance(arr, np.ndarray) or arr.ndim > 2]
        if non_array_preds:
            raise ValueError(f"'preds' contains non-numpy arrays at indices: {non_array_preds}")

        non_array_labels = [i for i, arr in enumerate(d['labels']) if not isinstance(arr, np.ndarray) or arr.ndim > 2]
        if non_array_labels:
            raise ValueError(f"'labels' contains non-numpy arrays at indices: {non_array_labels}")

        # Check all arrays have the same number of dimensions
        pred_dims = [arr.ndim for arr in d['preds']]
        if len(set(pred_dims)) > 1:
            raise ValueError(f"'preds' contains arrays with inconsistent dimensions: {pred_dims}")

        # Check shapes match between corresponding arrays
        shape_mismatches = [
            i for i, (pred_arr, label_arr) in enumerate(zip(d['preds'], d['labels']))
            if pred_arr.shape != label_arr.shape
        ]
        if shape_mismatches:
            raise ValueError(f"Shape mismatch between preds and labels at indices: {shape_mismatches}")

    elif isinstance(d['preds'], np.ndarray):
        # Both should be numpy arrays when preds is an array
        if not isinstance(d['labels'], np.ndarray):
            raise ValueError("'labels' must be a numpy array when 'preds' is a numpy array")

        # Validate that all tickers are strings
        if d['tickers'] and any(not isinstance(ticker, str) for ticker in d['tickers']):
            raise ValueError("All tickers must be strings")

        # Check shapes match
        if d['preds'].shape != d['labels'].shape:
            raise ValueError(f"Shape mismatch: preds {d['preds'].shape} != labels {d['labels'].shape}")

        # Ensure predictions array is 2-dimensional
        if d['preds'].ndim != 2:
            raise ValueError("'preds' array must be 2-dimensional")

    else:
        raise ValueError(f"'preds' must be a list of numpy arrays or a numpy array, got {type(d['preds'])}")

    # Check that all collections have the same length
    collections = ['preds', 'labels', 'tickers', 'last_date', 'pred_date']
    lengths = {key: len(d[key]) for key in collections}
    if len(set(lengths.values())) > 1:
        length_info = ", ".join(f"{key}: {length}" for key, length in lengths.items())
        raise ValueError(f"Inconsistent collection lengths - {length_info}")

    # For list-based predictions, check array lengths match across corresponding elements
    if isinstance(d['preds'], list):
        length_mismatches = [
            i for i, (pred_arr, ticker_arr) in enumerate(zip(d['preds'], d['tickers']))
            if not isinstance(ticker_arr, list) or len(ticker_arr) != len(pred_arr)
        ]
        if length_mismatches:
            raise ValueError(f"Array length mismatches at: {length_mismatches}")

    # Validate date fields
    try:
        last_date_series = pd.to_datetime(d['last_date'])
        assert isinstance(last_date_series, pd.DatetimeIndex)
    except Exception as e:
        raise ValueError(f"Error parsing 'last_date' as dates: {e}")

    try:
        pred_date_series = pd.to_datetime(d['pred_date'])
        assert isinstance(pred_date_series, pd.DatetimeIndex)
    except Exception as e:
        raise ValueError(f"Error parsing 'pred_date' as dates: {e}")

    # Check that all last_date entries are before pred_date entries
    invalid_dates = last_date_series >= pred_date_series
    if invalid_dates.any():
        # invalid_indices = invalid_dates[invalid_dates].index.tolist()
        raise ValueError("'last_date' must be before 'pred_date' for all entrie")


def _sort_ml_results(
        preds_list: list[np.ndarray],
        tickers_list: list[list],
        last_dates: pd.DatetimeIndex,
        preds_dates: pd.DatetimeIndex
) -> tuple[list[np.ndarray], list[list], pd.DatetimeIndex, pd.DatetimeIndex]:
    """Sort all results by last_dates."""
    if not last_dates.is_monotonic_increasing:
        sort_indices = np.argsort(last_dates)
        preds_list = [preds_list[i] for i in sort_indices]
        tickers_list = [tickers_list[i] for i in sort_indices]
        last_dates = last_dates[sort_indices]
        preds_dates = preds_dates[sort_indices]

    return preds_list, tickers_list, last_dates, preds_dates


def _prepare_ml_results(
        prediction_path: str,
        sort_by_date: bool = True
) -> tuple[list[np.ndarray], list[list], pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Prepare and validate prediction results from a pickle file, grouping by last_date and sorting chronologically.

    Args:
        prediction_path: Path to pickle file containing prediction results dictionary with keys:
           - 'labels': List of numpy arrays or single numpy array
           - 'preds': List of numpy arrays or single numpy array
           - 'tickers': List of ticker symbols
           - 'last_date': List of last observation dates
           - 'pred_date': List of prediction dates

        sort_by_date: Whether to sort results by last_dates in ascending order. Defaults to True.

    Returns:
        Tuple of (preds_list, tickers_list, last_dates, pred_dates) where:
        - preds_list: List of numpy arrays grouped by last_date
        - tickers_list: List of ticker lists corresponding to each prediction group
        - last_dates: DatetimeIndex of unique last dates (sorted)
        - pred_dates: DatetimeIndex of prediction dates (sorted)

    Raises:
        ValueError: If validation fails
    """
    # Load the pickle file
    with open(prediction_path, 'rb') as f:
        results = pickle.load(f)

    # Validate input data structure
    _validate_ml_results(results)

    if isinstance(results['preds'], np.ndarray):
        # Handle single array case: group predictions by last_date
        grouped_df = pd.DataFrame({
            'preds': results['preds'].tolist(),
            'tickers': results['tickers'],
            'last_dates': results['last_date'],
            'preds_dates': results['pred_date']
        }).groupby('last_dates').agg(list)

        # Convert each group back to a list
        preds_list = grouped_df['preds'].apply(np.array).tolist()
        tickers_list = grouped_df['tickers'].tolist()

        # Extract last date index
        last_dates = pd.to_datetime(grouped_df.index)

        # Extract first prediction date for each group
        preds_dates = pd.to_datetime(grouped_df['preds_dates'].apply(lambda x: x[0]).tolist())

    else:
        # Handle list case: extract data directly
        preds_list = results['preds']
        tickers_list = results['tickers']
        last_dates = pd.to_datetime(results['last_date'])
        preds_dates = pd.to_datetime(results['pred_date'])

    # Sort all data by last_dates if not already sorted
    if sort_by_date:
        preds_list, tickers_list, last_dates, preds_dates = _sort_ml_results(
            preds_list, tickers_list, last_dates, preds_dates
        )

    return preds_list, tickers_list, last_dates, preds_dates


def combine_ml_results(
        prediction_paths: list[str],
        sort_by_date: bool = True
) -> tuple[list[np.ndarray], list[list], pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Load and combine prediction results from multiple pickle files

    Args:
        prediction_paths: List of paths to pickle files containing prediction results
        sort_by_date: Whether to sort combined results by last_dates. Defaults to True

    Returns:
        Tuple of combined (preds_list, tickers_list, last_dates, pred_dates)
    """
    all_preds = []
    all_tickers = []
    all_last_dates = []
    all_pred_dates = []

    # Load and collect results from each path
    for path in prediction_paths:
        preds, tickers, last_dates, pred_dates = _prepare_ml_results(path, sort_by_date=False)

        all_preds.extend(preds)
        all_tickers.extend(tickers)
        all_last_dates.extend(last_dates.tolist())
        all_pred_dates.extend(pred_dates.tolist())

    # Convert back to appropriate types
    all_last_dates = pd.DatetimeIndex(all_last_dates)
    all_pred_dates = pd.DatetimeIndex(all_pred_dates)

    # Sort all data by last_dates if requested
    if sort_by_date:
        all_preds, all_tickers, all_last_dates, all_pred_dates = _sort_ml_results(
            all_preds, all_tickers, all_last_dates, all_pred_dates
        )

    return all_preds, all_tickers, all_last_dates, all_pred_dates


def _find_relevant_prediction_index(last_dates, current_date):
    """ Find the index of the most recent prediction before the current date """
    valid_indices = np.where(last_dates < current_date)[0]
    if len(valid_indices) == 0:
        raise ValueError(f'No prediction found for the period starting {current_date}')

    latest_index = valid_indices[-1]
    time_discrepancy = (current_date - last_dates[latest_index]).days
    if time_discrepancy > 7:
        raise ValueError(
            f'Discrepancy of {time_discrepancy} days between backtest start date {current_date} '
            f'and prediction date {last_dates[latest_index]}'
        )

    return latest_index


def create_long_short_portfolio_history(
        prediction_paths: list[str],
        top_k: int = 10,
        short_k: int = 0,
        exclude_tickers: list[str] = None,
        start_date: str = None,
        end_date: str = None,
        freq: str = None,
        verbose: bool = False
) -> list[dict]:
    """
    Generates a portfolio history based on a top-k/short-k stock selection strategy from model predictions

    Args:
        prediction_paths: List of paths to pickle files containing model predictions
        top_k: The number of top-performing stocks to go long on
        short_k: The number of bottom-performing stocks to go short on (default: 0)
        exclude_tickers: List of tickers to exclude from portfolio
        start_date: The start date for the backtest in 'YYYY-MM-DD' format
        end_date: The end date for the backtest in 'YYYY-MM-DD' format
        freq: The frequency for rebalancing the portfolio (e.g., 'M' for monthly)
        verbose: If True, prints detailed start/end date deviations and window duration comparisons. Default is False

    Returns:
        A list of dictionaries, where each dictionary represents the portfolio for a specific period
        Weights are positive for long positions and negative for short positions

    Raises:
        FileNotFoundError: If the file at `predictions_path` does not exist
        ValueError: If the data lists within the file have inconsistent lengths or if
                    a prediction set has an unexpected format
        ValueError: If the data is invalid or no suitable predictions are found
    """

    # Extract predictions
    preds_list, tickers_list, last_dates, preds_dates = combine_ml_results(prediction_paths)

    if not start_date and not end_date and not freq:
        freq = np.median([len(pd.bdate_range(start, end)) for start, end in zip(last_dates, preds_dates)])
        freq = f'{int(freq)}B'
        start_date, end_date = preds_dates[0].strftime('%Y-%m-%d'), preds_dates[-1].strftime('%Y-%m-%d')
        if verbose:
            print(f"Auto-detected: {start_date} to {end_date} with frequency {freq}")
    elif start_date and end_date and freq:
        pass
    else:
        raise ValueError(
            "Either provide ALL of (start_date, end_date, freq) or NONE of them for auto-detection. "
            f"Got: start_date={start_date}, end_date={end_date}, freq={freq}"
        )

    # Generate portfolio history
    portfolio_history = []
    time_steps = get_backtest_iterator(start_date, end_date, freq)
    for period_start, period_end in time_steps:
        # Find the most recent prediction before the period start
        pred_index = _find_relevant_prediction_index(last_dates, period_start)
        pred_set = preds_list[pred_index]
        ticker_set = tickers_list[pred_index]

        # Filter out excluded tickers
        if exclude_tickers:
            ticker_set = np.array(ticker_set)
            valid_mask = np.isin(ticker_set, exclude_tickers, invert=True)
            pred_set = pred_set[valid_mask]
            ticker_set = ticker_set[valid_mask].tolist()

        # Get top-k (long positions) and bottom-k (short positions)
        scores = pred_set if pred_set.ndim == 1 else pred_set[:, -1]
        top_k_indices = np.argsort(scores)[-top_k:] if top_k > 0 else []
        short_k_indices = np.argsort(scores)[:short_k] if short_k > 0 else []

        # Combine tickers
        long_tickers = [ticker_set[j] for j in top_k_indices]
        short_tickers = [ticker_set[j] for j in short_k_indices]
        portfolio_tickers = long_tickers + short_tickers

        if not portfolio_tickers:
            if verbose:
                print(f"Warning: All tickers excluded for period {period_start}")

            portfolio_history.append({
                'tickers': [],
                'weights': [],
                'test_start': period_start,
                'test_end': period_end
            })
            continue

        # Assign equal weights to the selected tickers.
        long_weights = [1.0 / len(portfolio_tickers)] * len(long_tickers)
        short_weights = [-1.0 / len(portfolio_tickers)] * len(short_tickers)
        weights = long_weights + short_weights

        portfolio_history.append({
            'tickers': portfolio_tickers,
            'weights': weights,
            'test_start': period_start,
            'test_end': period_end,
            'pred_start': last_dates[pred_index] + pd.tseries.offsets.BDay(1),
            'pred_end': preds_dates[pred_index] + pd.tseries.offsets.BDay(1),
        })

    if verbose:
        # Calculate timing deviations and window duration metrics
        df = pd.DataFrame([step for step in portfolio_history if step['tickers']])
        df['start_date_error'] = (df['test_start'] - df['pred_start']).dt.days
        df['end_date_error'] = (df['pred_end'] - df['test_end']).dt.days
        df['backtest_window'] = (df['test_end'] - df['test_start']).dt.days
        df['forecast_window'] = (df['pred_end'] - df['pred_start']).dt.days

        stats = df[['start_date_error', 'end_date_error', 'backtest_window', 'forecast_window']].describe().iloc[1:]
        print(stats.round(2))

    return portfolio_history


def create_quintile_portfolios_history(
        prediction_paths: list[str],
        start_date: str,
        end_date: str,
        freq: str,
        exclude_tickers: list[str] = None,
        verbose: bool = False
) -> dict[int, list[dict]]:
    """
    Generates 5 portfolio histories based on quintile ranking from model predictions.

    Each quintile contains approximately 20% of stocks, ranked by prediction scores:
    - Quintile 5: Top 20% (top predictions)
    - Quintile 4: Next 20%
    - Quintile 3: Middle 20%
    - Quintile 2: Next 20%
    - Quintile 1: Bottom 20% (worst predictions)

    Args:
        prediction_paths: List of paths to pickle files containing model predictions
        start_date: The start date for the backtest in 'YYYY-MM-DD' format
        end_date: The end date for the backtest in 'YYYY-MM-DD' format
        freq: The frequency for rebalancing the portfolio (e.g., 'M' for monthly)
        exclude_tickers: List of tickers to exclude from portfolios
        verbose: If True, prints detailed information about each quintile

    Returns:
        Dictionary with keys 1-5 (quintiles), where each value is a list of
        portfolio dictionaries for that quintile across all time periods

    Raises:
        FileNotFoundError: If prediction files don't exist
        ValueError: If data is invalid or parameters are inconsistent
    """

    # Extract predictions
    preds_list, tickers_list, last_dates, preds_dates = combine_ml_results(prediction_paths)

    # Initialize portfolio histories for each quintile
    quintile_histories = {i: [] for i in range(1, 6)}

    # Generate portfolio history
    time_steps = get_backtest_iterator(start_date, end_date, freq)

    for period_start, period_end in time_steps:
        # Find the most recent prediction before the period start
        pred_index = _find_relevant_prediction_index(last_dates, period_start)
        pred_set = preds_list[pred_index]
        ticker_set = tickers_list[pred_index]

        # Filter out excluded tickers
        if exclude_tickers:
            ticker_set = np.array(ticker_set)
            valid_mask = np.isin(ticker_set, exclude_tickers, invert=True)
            pred_set = pred_set[valid_mask]
            ticker_set = ticker_set[valid_mask].tolist()

        # Get scores and sorted indices
        scores = pred_set if pred_set.ndim == 1 else pred_set[:, -1]
        sorted_indices = np.argsort(scores)

        # Split into 5 quintiles and assign stocks to each one
        quintile_splits = np.array_split(sorted_indices, 5)

        for q, quintile_indices in enumerate(quintile_splits):
            portfolio_tickers = [ticker_set[j] for j in quintile_indices]

            if not portfolio_tickers:
                if verbose:
                    print(f"Warning: Quintile {q + 1} is empty for period {period_start}")

                quintile_histories[q + 1].append({
                    'tickers': [],
                    'weights': [],
                    'test_start': period_start,
                    'test_end': period_end,
                    'quintile': q + 1,
                    'n_stocks': 0
                })
                continue

            # Equal weights within each quintile
            weights = [1.0 / len(portfolio_tickers)] * len(portfolio_tickers)

            quintile_histories[q + 1].append({
                'tickers': portfolio_tickers,
                'weights': weights,
                'test_start': period_start,
                'test_end': period_end,
                'pred_start': last_dates[pred_index] + pd.tseries.offsets.BDay(1),
                'pred_end': preds_dates[pred_index] + pd.tseries.offsets.BDay(1),
                'quintile': q + 1,
                'n_stocks': len(portfolio_tickers)
            })

    return quintile_histories
