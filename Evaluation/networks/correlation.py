import numpy as np
import pandas as pd

from utils.validation import validate_prices_decorator


def _compute_feature_correlations(
        prices: pd.DataFrame,
        features: list[str],
        ticker_i: str,
        ticker_j: str,
        window_size: int
) -> list[pd.Series]:
    """Compute rolling correlations between two instruments for each feature.

    Args:
        prices: Price data with MultiIndex.
        features: List of feature column names.
        ticker_i: First instrument identifier.
        ticker_j: Second instrument identifier.
        window_size: Rolling window size.

    Returns:
        List of correlation Series, one for each feature.
    """
    correlations = []

    for feature in features:
        # Extract feature data for both instruments
        series_i = prices.loc[ticker_i, feature]
        series_j = prices.loc[ticker_j, feature]

        # Compute rolling correlation
        correlation = series_i.rolling(window=window_size).corr(series_j)
        correlations.append(correlation)

    return correlations


def _average_correlations_across_features(
        correlations: list[pd.Series],
        dates: list
) -> np.ndarray:
    """Average correlations across features and align with date index.

    Args:
        correlations: List of correlation Series to average.
        dates: Target date index for alignment.

    Returns:
        Numpy array of averaged correlations aligned with dates.
    """
    # Concatenate correlations and compute mean across features
    correlation_df = pd.concat(correlations, axis=1)
    avg_correlation = correlation_df.mean(axis=1)

    # Align with target dates (fills missing dates with NaN)
    avg_correlation_aligned = avg_correlation.reindex(dates)

    return avg_correlation_aligned.values


def build_correlation_adjacency_matrix(
        prices: pd.DataFrame,
        features: list[str],
        window_size: int = 20
) -> tuple[np.ndarray, list, list[str]]:
    """Build a temporal adjacency matrix based on rolling correlations.

    Computes rolling correlations between instruments for each feature, then
    averages across features to create a symmetric adjacency matrix for each
    time step.

    Args:
        prices: MultiIndex DataFrame with levels ['instrument', 'date']
            and columns representing different features.
        features: List of column names to compute correlations for.
        window_size: Number of periods for rolling correlation calculation (default 20).

    Returns:
        A tuple containing:
        - temporal_adj_matrix: 3D numpy array of shape (n_dates, n_tickers, n_tickers)
          containing correlation-based adjacency matrices over time.
        - dates: Sorted list of unique dates from the data.
        - tickers: List of unique instrument names.

    Raises:
        ValueError: If features are not present in price_data columns.
        IndexError: If price_data doesn't have required MultiIndex structure.
    """

    missing_features = set(features) - set(prices.columns)
    if missing_features:
        raise ValueError(f"Features not found in price_data: {missing_features}")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    # Extract unique instruments and dates from MultiIndex
    tickers = prices.index.get_level_values('instrument').unique().tolist()
    dates = sorted(prices.index.get_level_values('date').unique().tolist())

    # Initialize 3D adjacency matrix: [time, instrument_i, instrument_j]
    n_dates, n_tickers = len(dates), len(tickers)
    temporal_adj_matrix = np.full(
        (n_dates, n_tickers, n_tickers),
        np.nan,
        dtype=np.float64
    )

    # Compute pairwise correlations for each instrument pair
    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            # Calculate rolling correlations across all features
            feature_correlations = _compute_feature_correlations(
                prices, features, tickers[i], tickers[j], window_size
            )

            # Average correlations across features for each time step
            avg_correlation = _average_correlations_across_features(
                feature_correlations, dates
            )

            # Fill symmetric positions in adjacency matrix
            temporal_adj_matrix[:, i, j] = avg_correlation
            temporal_adj_matrix[:, j, i] = avg_correlation  # Ensure symmetry

    return temporal_adj_matrix, dates, tickers


def _set_diagonal_zero_inplace(adj_matrix):
    """Modify the original matrix in-place."""
    n_temporal, n_stocks = adj_matrix.shape[0], adj_matrix.shape[1]
    time_indices = np.arange(n_temporal)[:, None]
    diag_indices = np.arange(n_stocks)
    adj_matrix[time_indices, diag_indices, diag_indices] = 0


@validate_prices_decorator
def build_pos_neg_correlation_adjacency_matrix(
        prices: pd.DataFrame,
        features: list[str],
        window_size: int = 20,
        th: float = 0.6,
) -> tuple[np.ndarray, list, list[str], list[str]]:
    """ Build positive and negative correlation adjacency matrices


    Computes rolling correlations between instruments for each feature, then
    averages across features to create a symmetric adjacency matrix for each
    time step. Next converts  into binary adjacency matrices by
    thresholding correlations. Creates separate matrices for positive and
    negative correlations above the absolute threshold value.

    Args:
        prices: MultiIndex DataFrame with levels ['instrument', 'date'] and columns representing different features.
        features: List of column names to compute correlations for
        window_size: Number of periods for rolling correlation calculation (default 20)
        th: Correlation threshold for creating binary adjacency matrices. Must be between 0 and 1

    Returns:
        A tuple containing:
        - pos_neg_adj_matrix: Binary adjacency matrix of shape (T, 2, S, S)
          where the second dimension represents [positive, negative] correlations.
        - dates: Sorted list of unique dates from the data (T).
        - feature_names: List of feature names ['pos', 'neg'] (2).
        - tickers: List of unique instrument names (S).

    Raises:
        ValueError: If threshold is not in the valid range [0, 1].
    """

    if not 0 <= th <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {th}")

    temporal_adj_matrix, dates, tickers = build_correlation_adjacency_matrix(prices, features, window_size)

    # Create binary adjacency matrices based on correlation thresholds
    pos_adj = (temporal_adj_matrix > th).astype(np.int8)
    neg_adj = (temporal_adj_matrix < -th).astype(np.int8)

    # Remove self-loops from both matrices
    _set_diagonal_zero_inplace(adj_matrix=pos_adj)
    _set_diagonal_zero_inplace(adj_matrix=neg_adj)

    # Combine positive and negative adjacency matrices
    pos_neg_adj_matrix = np.stack([pos_adj, neg_adj], axis=1)
    return pos_neg_adj_matrix, dates, ['pos', 'neg'], tickers
