from functools import wraps

import pandas as pd


def validate_prices(prices):
    """
    Validate that prices DataFrame has correct MultiIndex structure and is properly sorted

    Args:
        prices: DataFrame with MultiIndex ['instrument', 'date']

    Raises:
        ValueError: If validation fails
    """

    # Check if it's a MultiIndex
    if not isinstance(prices.index, pd.MultiIndex):
        raise ValueError('DataFrame must have a MultiIndex with instrument in first level and date in second level')

    # Validate the index level count
    if prices.index.nlevels < 2:
        raise ValueError('DataFrame index must have at least two levels')

    # Check level names if they exist
    if prices.index.names[0] is not None and prices.index.names[0] != 'instrument':
        raise ValueError(f"First index level must be named 'instrument', got '{prices.index.names[0]}'")

    if prices.index.names[1] is not None and prices.index.names[1] != 'date':
        raise ValueError(f"Second index level must be named 'date', got '{prices.index.names[1]}'")

    # Validate date level format - can be datetime or string
    date_level = prices.index.levels[1]
    if isinstance(date_level, pd.DatetimeIndex):
        # Already datetime - good to go
        pass
    elif all(isinstance(d, str) for d in date_level):
        # Validate string format
        try:
            pd.to_datetime(date_level, format='%Y-%m-%d')
        except ValueError:
            raise ValueError('String dates in date level must be in YYYY-MM-DD format')
    else:
        raise ValueError('Date level must contain either datetime objects or strings in YYYY-MM-DD format')

    # Check if each ticker is sorted by date
    for instrument in prices.index.get_level_values(0).unique():
        instrument_data = prices.loc[instrument]
        if not instrument_data.index.is_monotonic_increasing:
            raise ValueError(f'Dates for instrument "{instrument}" are not sorted in ascending order')


def validate_prices_decorator(func):
    """
    Decorator to validate prices DataFrame before function execution

    Assumes the first argument of the decorated function is the prices DataFrame
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            prices = args[0]
            validate_prices(prices)
        elif 'prices' in kwargs:
            validate_prices(kwargs['prices'])
        else:
            raise ValueError("No prices DataFrame found in function arguments")

        return func(*args, **kwargs)

    return wrapper
