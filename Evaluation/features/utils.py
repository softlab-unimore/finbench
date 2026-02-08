import pandas as pd
from .rolling import rolling_slope, rolling_rsquare, rolling_resi


def ref(series: pd.Series, n: int) -> pd.Series:
    """Shifts values in a time series by n periods, handling MultiIndex data.

    For a Series with MultiIndex (instrument, date), this function shifts the values
    within each instrument group independently.

    Args:
        series: A pandas Series with values to be shifted. Must have a MultiIndex with
            the first level as the instrument identifier and the second level as the date.
        n: The number of periods to shift. Positive values shift to past data
            (lag) and negative values shift to future data (lead). Zero is not
            allowed and will raise a ValueError.

    Returns:
        A pandas Series with the same index as the input, containing the shifted values.
        For n > 0, the first n values in each group will be NaN.
        For n < 0, the last |n| values in each group will be NaN.
    """
    if n == 0:
        raise ValueError("The shift parameter n cannot be zero. Use positive values "
                         "for past data or negative values for future data.")

    # Apply shifting within each instrument group
    return series.groupby(level=0).shift(n)


def mean(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling mean of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling mean values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).mean()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-2 * (n - 1))  # Truncate initial NaN values
    return result


def ema(series: pd.Series, n: int) -> pd.Series:
    """Calculate the exponential moving average of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the exponential moving period

    Returns:
        A pandas Series with the calculated rolling mean values
    """
    result = series.groupby(level=0).ewm(span=n, adjust=False).mean()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.ewm with MultiIndex
    return result


def std(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling standard deviation of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling standard deviation values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).std()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-2 * (n - 1))  # Truncate initial NaN values
    return result


def sum_(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling sum of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling sum values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).sum()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-2 * (n - 1))  # Truncate initial NaN values
    return result


def max_(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling maximum of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling maximum values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).max()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def min_(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling minimum of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling minimum values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).min()
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def quantile(series: pd.Series, n: int, q: float) -> pd.Series:
    """Calculate the rolling quantile of a Series within each security group.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window
        q: Quantile to compute (0 <= q <= 1)

    Returns:
        A pandas Series with the calculated rolling quantile values
    """
    result = series.groupby(level=0).rolling(n, min_periods=1).quantile(q)
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def rank(series: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling rank (percentile) of the last value in the window.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated rolling rank values (0 to 1)
    """

    result = series.groupby(level=0).rolling(n, min_periods=1).rank(pct=True)
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def idx_max(series: pd.Series, n: int) -> pd.Series:
    """Find the position (1-based) of the maximum value in each rolling window.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the position of the maximum value in each window
    """

    result = series.groupby(level=0).rolling(n, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def idx_min(series: pd.Series, n: int) -> pd.Series:
    """Find the position (1-based) of the minimum value in each rolling window.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the position of the minimum value in each window
    """

    result = series.groupby(level=0).rolling(n, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
    result = result.reset_index(level=0, drop=True)  # Needed for groupby.rolling with MultiIndex
    result = result.groupby(level=0).tail(-(n - 1))  # Truncate initial NaN values
    return result


def slope(series: pd.Series, n: int) -> pd.Series:
    """Calculate the slope coefficient of linear regression for rolling windows.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated slope values
    """
    result = pd.Series(index=series.index, dtype=float)

    for security, group in series.groupby(level=0):
        if len(group) >= n:
            # Convert to numpy array for faster processing
            values = group.values
            slope_values = rolling_slope(values, n)
            # Align with the original index
            result.loc[group.index[n:]] = slope_values[n:]

    return result


def rsquare(series: pd.Series, n: int) -> pd.Series:
    """Calculate the R-squared value of linear regression for rolling windows.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated R-squared values
    """
    result = pd.Series(index=series.index, dtype=float)

    for security, group in series.groupby(level=0):
        if len(group) >= n:
            # Convert to numpy array for faster processing
            values = group.values
            rsquare_values = rolling_rsquare(values, n)
            # Align with the original index
            result.loc[group.index[n:]] = rsquare_values[n:]

    return result


def resi(series: pd.Series, n: int) -> pd.Series:
    """Calculate the residual of linear regression for rolling windows.

    Args:
        series: A pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated residual values
    """
    result = pd.Series(index=series.index, dtype=float)

    for security, group in series.groupby(level=0):
        if len(group) >= n:
            # Convert to numpy array for faster processing
            values = group.values
            resi_values = rolling_resi(values, n)
            # Align with the original index
            result.loc[group.index[n:]] = resi_values[n:]

    return result


def corr(series: pd.Series, series2: pd.Series, n: int) -> pd.Series:
    """Calculate the rolling correlation between two series within each security group.

    Args:
        series: First pandas Series with a MultiIndex (security, date)
        series2: Second pandas Series with a MultiIndex (security, date)
        n: The size of the rolling window

    Returns:
        A pandas Series with the calculated correlation values
    """
    if not series.index.equals(series2.index):
        raise ValueError("Both series must have the same index structure")

    result = pd.Series(index=series.index, dtype=float)

    for security, group1 in series.groupby(level=0):
        if len(group1) >= n:
            # Use pandas built-in rolling correlation
            corr_values = group1.rolling(n).corr(series2[[security]]).values
            # Align with the original index
            result.loc[group1.index] = corr_values

    return result
