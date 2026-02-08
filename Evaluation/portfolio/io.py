import numpy as np
import pandas as pd


def _parse_weight_percentage(weight: str) -> float:
    """ Extract and validate portfolio weight from string format """
    try:
        # Validate weight format and convert to decimal
        assert weight.endswith('%'), f'Ticker weight must end with %'
        weight_value = float(weight.rstrip('%'))
        assert 0 <= weight_value <= 100, f'Ticker weight must be between 0% and 100%'
        return weight_value / 100
    except (AssertionError, ValueError) as e:
        raise ValueError(f'Invalid portfolio weight: {str(e)}')


def read_portfolio_history(portfolio_path: str, end_date: str = None) -> list[dict]:
    """
    Read portfolio data from CSV file and convert to standardized format.

    Args:
        portfolio_path: Path to CSV file containing portfolio data
        end_date: End date for the last portfolio period. If None, uses today's date

    Returns:
        List of dictionaries containing portfolio snapshots with:
        - test_start: Start date of portfolio period
        - test_end: End date of portfolio period
        - tickers: List of ticker symbols
        - weights: List of corresponding weights

    Raises:
        ValueError: If date format is invalid, weights don't sum to 100%, or end_date is before the last portfolio date
    """

    # Read and validate date format in portfolio file
    pf = pd.read_csv(portfolio_path, header=None, index_col=0)
    try:
        pf.index = pd.to_datetime(pf.index, format='%Y-%m-%d')
        pf.sort_index(inplace=True, ascending=True)
    except ValueError:
        raise ValueError('Invalid date format in portfolio file. The index must contain dates in YYYY-MM-DD format')

    # Parse end_date and validate it's after the last portfolio date
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d') if end_date else pd.to_datetime('today').normalize()
    if end_date <= pf.index.max():
        raise ValueError('End date must be after the last portfolio date')

    history = []
    for i, (dt, row) in enumerate(pf.iterrows()):
        # Clean data and extract tickers and weights
        row = row.dropna()
        tickers = [val.split(',')[0].strip() for val in row]
        weights = [_parse_weight_percentage(val.split(',')[1]) for val in row]

        # Validate total portfolio weight
        assert 0 <= sum(weights) <= 1.0 + 1e-3, 'Portfolio weights must be between 0% and 100%'

        # Set end date as next row's date or end_date for last row
        test_end = pf.index[i + 1] if i < len(pf) - 1 else end_date

        history.append({
            'test_start': dt,
            'test_end': test_end,
            'tickers': tickers,
            'weights': weights
        })

    return history


def _extract_portfolio(df_all: pd.DataFrame, table_id: int = 2):
    """Extract a specific portfolio table from the Excel report.

    Args:
        df_all: DataFrame containing multiple tables from the Excel sheet
        table_id: Index of the table to extract (default: 2)

    Returns:
        DataFrame containing only the requested table

    Raises:
        ValueError: If tables aren't properly formatted or table_id is out of bounds
    """
    # Identify rows that separate tables (all values are null)
    cond = df_all.isnull().all(axis=1).values

    # Detect start positions of tables (transition from null to data)
    table_starts = list(np.where(~cond[1:] & cond[:-1])[0] + 2)
    table_starts.insert(0, 0)

    # Detect end positions of tables (transition from data to null)
    table_ends = list(np.where(cond[1:] & ~cond[:-1])[0] + 1)
    table_ends.append(len(df_all))

    if len(table_starts) != len(table_ends):
        raise ValueError('Portfolio is not formatted correctly')

    if table_id >= len(table_starts):
        raise ValueError(f'Table {table_id} is out of bound')

    a = table_starts[table_id]
    b = table_ends[table_id]
    return df_all.iloc[a:b].copy()


def read_portfolio_report(portfolio_path: str, sheet_name: str) -> list[dict]:
    """Read and parse portfolio report from Excel file.

    Extracts portfolio data from the specified Excel sheet and converts it into a
    structured format with ticker symbols and weights.

    Args:
        portfolio_path: Path to the Excel portfolio report file
        sheet_name: Name of the sheet containing portfolio data

    Returns:
        List of dictionaries, each containing test period dates, tickers and weights

    Raises:
        ValueError: If the file is not an Excel file
        KeyError: If expected columns are missing in the Excel file
        FileNotFoundError: If the file doesn't exist
    """
    if not portfolio_path.endswith('.xlsx'):
        raise ValueError(f'Invalid portfolio report path: {portfolio_path}')

    try:
        reports = pd.read_excel(portfolio_path, sheet_name=sheet_name, index_col=0, header=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")
    except ValueError as e:
        raise ValueError(f"Error reading Excel file: {e}")

    # Extract a specific portfolio table from the Excel report
    pf = _extract_portfolio(reports, table_id=2)

    # Remove summary rows which are not part of the time series
    pf.drop(['AVG', 'YRET'], axis=0, inplace=True)
    # Convert index to datetime and ensure chronological order
    pf.index = pd.to_datetime(pf.index)
    pf.sort_index(inplace=True, ascending=True)

    try:
        # Last column contains portfolio composition data as string representation of list
        backtest = pf.iloc[:, -1].apply(eval).tolist()
    except (SyntaxError, TypeError):
        raise ValueError("Portfolio data format is incorrect - cannot evaluate entries")

    history = []
    for i, row in enumerate(backtest):
        # Extract ticker symbols (removing first character)
        tickers = [val['name'][1:] for val in row]

        # Equal weight allocation for all tickers
        weights = [1 / len(tickers)] * len(tickers)

        # Determine test period end date (next period start or estimated)
        test_end = pf.index[i + 1] if i < len(pf) - 1 else pf.index[i] + (pf.index[1] - pf.index[0])

        history.append({
            'test_start': pf.index[i],
            'test_end': test_end,
            'tickers': tickers,
            'weights': weights
        })

    return history
