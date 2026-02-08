import concurrent.futures
import os
import threading
import time

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException


def download_prices(ticker: str, api_token: str) -> pd.DataFrame:
    """Downloads daily EOD data for a given stock ticker

    Args:
        ticker: Stock symbol to download data for
        api_token: EOD Historical Data API authentication token

    Returns:
        pd.DataFrame: Daily EOD with adjusted prices

    Raises:
        requests.HTTPError: If the API request fails
        ValueError: If the price data contains duplicate or null dates
    """

    endpoint_url = f'https://eodhistoricaldata.com/api/eod/{ticker}?api_token={api_token}&fmt=json'

    r = requests.get(endpoint_url)
    r.raise_for_status()

    data = r.json()
    price = pd.DataFrame(data)

    if price.empty:
        raise ValueError(f'No ticker data found for {ticker}')

    # Ensure data quality by checking for problematic dates
    if price['date'].duplicated().any() or price['date'].isnull().any():
        raise ValueError(f'Price data for {ticker} contains duplicate or null dates')

    # Convert date column to datetime and set as index
    price['date'] = pd.to_datetime(price['date'])
    price = price.set_index('date', drop=True).sort_index(ascending=True)

    # Remove timezone info for consistency with other data
    price = price.tz_localize(None)

    # Adjust the price columns based on the adjusted close values
    price = price.rename(columns={'adjusted_close': 'adj_close'})
    adjustment_factor = price['adj_close'] / price['close']
    price['adj_open'] = price['open'] * adjustment_factor
    price['adj_high'] = price['high'] * adjustment_factor
    price['adj_low'] = price['low'] * adjustment_factor
    price['factor'] = adjustment_factor

    return price


def download_returns(ticker: str, api_token: str) -> pd.Series:
    """Downloads and calculates daily returns for a given stock ticker

    Args:
        ticker: Stock symbol to download data for
        api_token: EOD Historical Data API authentication token

    Returns:
        pd.Series: Daily returns calculated from adjusted closing prices

    Raises:
        requests.HTTPError: If the API request fails
        ValueError: If the price data contains duplicate or null dates
    """
    price_data = download_prices(ticker, api_token)
    return price_data['adj_close'].pct_change()


def _extract_fundamentals(data: dict) -> pd.DataFrame:
    """ Extract and combine quarterly financial data from API response """
    fin_cols = ['BS_commonStockSharesOutstanding', 'BS_totalStockholderEquity', 'BS_totalAssets', 'BS_totalLiab',
                'CF_dividendsPaid', 'CF_netIncome', 'IS_totalRevenue', 'IS_grossProfit', ]

    # Safely extract the nested quarterly data
    bs_data = data.get('Financials', {}).get('Balance_Sheet', {}).get('quarterly', {})
    cf_data = data.get('Financials', {}).get('Cash_Flow', {}).get('quarterly', {})
    is_data = data.get('Financials', {}).get('Income_Statement', {}).get('quarterly', {})

    # Validate data exists
    if not bs_data or not cf_data or not is_data:
        raise ValueError(f'No quarterly financial data found in API response')

    # Validate periods match across all statements
    if not (set(bs_data.keys()).issubset(cf_data.keys()) or set(cf_data.keys()).issubset(bs_data.keys())):
        raise ValueError('Quarterly periods mismatch between BS and CF')
    if not (set(bs_data.keys()).issubset(is_data.keys()) or set(is_data.keys()).issubset(bs_data.keys())):
        raise ValueError('Quarterly periods mismatch between BS and IS')
    if not (set(cf_data.keys()).issubset(is_data.keys()) or set(is_data.keys()).issubset(cf_data.keys())):
        raise ValueError('Quarterly periods mismatch between CF and IS')

    # Convert to DataFrames with datetime index
    df_bs = pd.DataFrame.from_dict(bs_data, orient='index')
    df_cf = pd.DataFrame.from_dict(cf_data, orient='index')
    df_is = pd.DataFrame.from_dict(is_data, orient='index')

    df_bs.index = pd.to_datetime(df_bs.index)
    df_cf.index = pd.to_datetime(df_cf.index)
    df_is.index = pd.to_datetime(df_is.index)

    # Add prefixes to column names to avoid conflicts
    df_bs = df_bs.add_prefix('BS_')
    df_cf = df_cf.add_prefix('CF_')
    df_is = df_is.add_prefix('IS_')

    # Merge all DataFrames on their index (datetime)
    # Note: Using the Balance Sheet as the primary source of truth for quarterly periods
    # combined_df = pd.concat([df_bs, df_cf, df_is], axis=1, join='outer', sort=True)
    combined_df = df_bs.join(df_cf, how='left').join(df_is, how='left')

    # Validate required metadata columns exist
    required_meta_cols = ['BS_date', 'BS_filing_date']
    missing_meta = [col for col in required_meta_cols if col not in combined_df.columns]
    if missing_meta:
        raise ValueError(f'Missing required metadata columns: {missing_meta}')

    # Add date columns from Balance Sheet
    combined_df['date'] = combined_df['BS_date']
    combined_df['filing_date'] = combined_df['BS_filing_date']

    # Validate required financial columns exist
    missing_fin_cols = [col for col in fin_cols if col not in combined_df.columns]
    if missing_fin_cols:
        raise ValueError(f'Missing required financial columns: {missing_fin_cols}')

    # Data integrity checks
    if combined_df['date'].duplicated().any():
        raise ValueError('Found duplicate report dates')

    if combined_df['date'].isnull().any():
        raise ValueError('Found null report dates')

    return combined_df[['date', 'filing_date'] + fin_cols].reset_index(drop=True)


def _clean_fundamentals(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """ Cleans and processes a DataFrame of fundamental data """
    # Convert date columns
    df['date'] = pd.to_datetime(df['date'])
    df['filing_date'] = pd.to_datetime(df['filing_date'])

    # Convert float columns
    feature_cols = df.columns.drop((['date', 'filing_date']))
    df[feature_cols] = df[feature_cols].astype(float)

    # Filter by start date and sort the DataFrame
    df = df[df['date'] > pd.to_datetime(start_date)]
    df = df.sort_values(by=['date', 'filing_date'], ascending=True)
    df = df.loc[df['filing_date'].first_valid_index():].reset_index(drop=True)

    # Data integrity checks
    if df.empty:
        raise ValueError(f'No quarterly data available after {start_date}')

    if df['filing_date'].isnull().all():
        raise ValueError('No valid filing dates found')

    # Make duplicated filing dates NaT ??
    # df.loc[df.duplicated(subset=['filing_date'], keep='last'), 'filing_date'] = pd.NaT

    # Fix filing date NaT rows using report date
    nan_filing_date_mask = df['filing_date'].isnull()
    if nan_filing_date_mask.any():
        df.loc[nan_filing_date_mask, feature_cols] = np.nan
        df.loc[nan_filing_date_mask, 'filing_date'] = df.loc[nan_filing_date_mask, 'date']

    return df


def download_market_cap(ticker: str, api_token: str, start_date: str = '2005-01-01') -> pd.DataFrame:
    """
    Downloads historical prices and fundamental data to calculate historical market cap and 15 factor zoo

    Args:
        ticker (str): The stock ticker symbol
        api_token (str): Your EOD Historical Data API token
        start_date (str, optional): The earliest date for the analysis. Defaults to '2005-01-01'

    Returns:
        pd.DataFrame: A DataFrame with a 'market_cap' column, indexed by date

    Raises:
        DataNotFoundError: If essential data cannot be found in the API response
        ConnectionError: If the API request fails
        ValueError: If data is inconsistent or cannot be parsed
    """
    endpoint_url = f'https://eodhistoricaldata.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json'

    try:
        response = requests.get(endpoint_url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f'API request failed for {ticker}: {e}') from e
    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f'Failed to parse JSON response for {ticker}: {e}') from e

    # Extract fundamentals data
    try:
        raw_df = _extract_fundamentals(data)
        fin_df = _clean_fundamentals(raw_df.copy(), start_date)
    except ValueError as e:
        raise ValueError(f'Data extraction failed for {ticker}: {str(e)}') from e

    # Fetch price data and filter bases on financials
    price_df = download_prices(ticker, api_token)
    price_df = price_df.loc[price_df.index >= fin_df['filing_date'].min(), ['adj_close', 'volume']]

    # Compute 15 Factor Zoo

    # ttm features
    fin_df['ttm_dividendsPaid'] = fin_df['CF_dividendsPaid'].fillna(0).abs().rolling(4).sum()
    fin_df['ttm_netIncome'] = fin_df['CF_netIncome'].rolling(4).sum()
    fin_df['ttm_totalRevenue'] = fin_df['IS_totalRevenue'].rolling(4).sum()
    fin_df['ttm_grossProfit'] = fin_df['IS_grossProfit'].rolling(4).sum()

    # Growth factors
    fin_df['sales_growth'] = fin_df['IS_totalRevenue'] / fin_df['IS_totalRevenue'].shift(4) - 1
    fin_df['earnings_growth'] = fin_df['CF_netIncome'] / fin_df['CF_netIncome'].shift(4) - 1

    # Profitability factors
    fin_df['gross_profit_to_assets'] = fin_df['ttm_grossProfit'] / fin_df['BS_totalAssets']
    fin_df['roe'] = fin_df['ttm_netIncome'] / fin_df['BS_totalStockholderEquity']
    fin_df['roa'] = fin_df['ttm_netIncome'] / fin_df['BS_totalAssets']

    # Debt Issuance factor
    fin_df['market_leverage'] = fin_df['BS_totalLiab'] / fin_df['BS_totalAssets']

    # Merge financials and prices
    factor_df = pd.merge_asof(
        left=price_df,
        right=fin_df,
        left_index=True,
        right_on='filing_date'
    )

    # Size factor
    factor_df['market_cap'] = factor_df['adj_close'] * factor_df['BS_commonStockSharesOutstanding']

    # Value factors
    factor_df['dividend_yield'] = factor_df['ttm_dividendsPaid'] / factor_df['market_cap']
    factor_df['earnings_to_price'] = factor_df['ttm_netIncome'] / factor_df['market_cap']
    factor_df['sales_to_price'] = factor_df['ttm_totalRevenue'] / factor_df['market_cap']
    factor_df['book_value_to_price'] = factor_df['BS_totalStockholderEquity'] / factor_df['market_cap']

    # Other factors and low risk factors
    factor_df['volatility'] = factor_df['adj_close'].pct_change().rolling(252).std()
    factor_df['momentum'] = factor_df['adj_close'].shift(21) / factor_df['adj_close'].shift(252) - 1
    factor_df['rev'] = factor_df['adj_close'] / factor_df['adj_close'].shift(21) - 1
    factor_df['share_turnover'] = factor_df['volume'].rolling(21).sum() / factor_df['BS_commonStockSharesOutstanding']

    factor_cols = [
        'dividend_yield', 'earnings_to_price', 'sales_to_price', 'book_value_to_price',  # Value Factors
        'sales_growth', 'earnings_growth',  # Growth Factors
        'gross_profit_to_assets', 'roe', 'roa',  # Profitability factors
        'market_leverage', 'momentum', 'rev', 'volatility', 'share_turnover',  # Other and low risk factors
        'market_cap'  # Size factor
    ]

    return factor_df[factor_cols]


def download_info(ticker: str, api_token: str) -> pd.DataFrame:
    """Downloads fundamental information for a given stock ticker

    Args:
        ticker: Stock symbol to download information for
        api_token: EOD Historical Data API authentication token

    Returns:
        pd.DataFrame: Fundamental company information including sector, industry and name data

    Raises:
        requests.HTTPError: If the API request fails
        KeyError: If the API response is missing expected fields
        ValueError: If the ticker fundamental data is incomplete or malformed
    """
    endpoint_url = f'https://eodhistoricaldata.com/api/fundamentals/{ticker}?api_token={api_token}&fmt=json'

    r = requests.get(endpoint_url)
    r.raise_for_status()

    data = r.json()

    # Check if the response contains the necessary data
    if 'General' not in data:
        raise ValueError(f'Fundamental data for {ticker} is incomplete or malformed')

    # Define the fields we want to extract
    fields = ['Code', 'PrimaryTicker', 'Name', 'GicSector', 'GicGroup', 'GicIndustry', 'GicSubIndustry']

    # Extract the information, with checks
    info = {}
    for k in fields:
        try:
            info[k] = data['General'][k]
        except KeyError:
            raise ValueError(f'Required field {k} not found in fundamental data for {ticker}')

    df = pd.DataFrame([info], index=[ticker])
    df.index.name = 'instrument'

    if df.empty:
        raise ValueError(f'Failed to extract fundamental data for {ticker}')

    return df


def search_eodhd_ticker(text: str, api_token: str) -> list[dict]:
    """
    Search for ticker symbols using EODHD API

    Args:
        text: Search query string
        api_token: EODHD API authentication token

    Returns:
        List of dictionaries containing ticker information

    Raises:
        ValueError: When no data is found or invalid response
        requests.RequestException: When API request fails
    """
    if not text.strip():
        raise ValueError('Search text cannot be empty')

    endpoint_url = f'https://eodhistoricaldata.com/api/search/{text.strip()}?api_token={api_token}&fmt=json'
    try:
        response = requests.get(endpoint_url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f'API request failed: {e}')

    # Validate response format
    if not isinstance(data, list):
        raise ValueError(f'Unexpected response format for {text}')

    if not data:
        raise ValueError(f'No ticker data found for {text}')

    return data


def download_news(ticker: str, api_token: str) -> list[dict]:
    """
    Download historical news data for a given ticker from EODHD API
    Fetches news from 2017-01-01 to today, month by month to handle API limits

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        api_token: EODHD API token for authentication

    Returns:
        List of news articles as dictionaries

    Raises:
        requests.RequestException: If API request fails
        ValueError: If no news found or invalid response format
    """

    # Generate month ranges from 2005 to today
    month_starts = pd.date_range(start='2017-01-01', end=pd.to_datetime('today'), freq='MS').strftime('%Y-%m-%d')
    month_ends = pd.date_range(start='2017-01-01', end=pd.to_datetime('today'), freq='ME').strftime('%Y-%m-%d')
    month_ranges = [(start_month, end_month) for start_month, end_month in zip(month_starts, month_ends)]

    endpoint_url = 'https://eodhd.com/api/news'
    results = []
    for start_month, end_month in month_ranges:
        params = {'s': ticker, 'limit': 1000, 'from': start_month, 'to': end_month, 'api_token': api_token,
                  'fmt': 'json'}
        max_retries = 3
        data = []
        for attempt in range(max_retries):
            try:
                response = requests.get(endpoint_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    error_msg = f'Failed to fetch news for {ticker} from {start_month} to {end_month}: {str(e)}'
                    raise requests.RequestException(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff

        # Validate response format
        if not isinstance(data, list):
            raise ValueError(f'Expected list response for {ticker} from {start_month} to {end_month}')

        # Add retrieved news
        if data:
            results.extend(data)

    if not results:
        raise ValueError(f'No news found for {ticker}')

    return results


def download_multi(
        tickers: list,
        api_token: str,
        data_type: str,
        verbose: bool = False,
        p: int = 2
) -> tuple[dict, list[dict]]:
    """
    Fetch financial data series for multiple tickers using parallel processing

    This function uses a ThreadPoolExecutor to fetch data for multiple tickers
    concurrently, which can significantly speed up the data retrieval process

    Args:
        tickers (list[str]): list of ticker symbols to fetch data for
        api_token (str): API token for authentication
        data_type (str): Type of data to download. Options: 'price', 'info'
        verbose (bool, optional): If True, print detailed progress. Defaults to False
        p (int, optional): Number of concurrent threads to use. Defaults to 2

    Returns:
        tuple[dict, list]: A tuple containing:
            - A dictionary of successfully fetched data
            - A list of dictionaries for tickers that failed, including error information

    Raises:
        Any exceptions raised by the underlying 'fetch_series' function.
    """

    # Define mapping of data types to download functions
    download_functions = {
        'price': download_prices,
        'returns': download_returns,
        'market_cap': download_market_cap,
        'info': download_info,
        'news': download_news,
        'search': search_eodhd_ticker
    }

    # Validate requested data type
    if data_type not in download_functions:
        raise ValueError(f'Unknown data type: {data_type}. Available types: {list(download_functions.keys())}')

    # Get the appropriate download function
    download_fn = download_functions[data_type]

    # Determine the number of threads to use
    p = min(len(set(tickers)) + 1, p if p != -1 else os.cpu_count())

    # Create a lock for thread-safe printing
    print_lock = threading.Lock()

    def process_ticker(args):
        """Process a single ticker and handle any exceptions """
        ticker = args[0]  # Extract ticker from args
        start_time_ = time.time()

        if verbose:
            with print_lock:
                print(f'Starting work on ticker: {ticker}')

        try:
            # Attempt to fetch data for the ticker
            data = download_fn(*args)
            result = {ticker: data}, []
        except Exception as e:
            if verbose:
                with print_lock:
                    print(f'Error processing ticker {ticker}: {str(e)}')
            result = {}, [{'Ticker': ticker, 'Error': str(e)}]

        if verbose:
            with print_lock:
                print(f'Finished {ticker}. Time: {time.time() - start_time_:.2f}s')

        return result

    # Prepare arguments for each ticker
    start_time = time.time()
    list_arguments = [(ticker, api_token) for ticker in set(tickers)]

    # Use ThreadPoolExecutor to process tickers concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=p) as executor:
        if verbose:
            print(f'Starting ThreadPoolExecutor with {p} workers')
        results = list(executor.map(process_ticker, list_arguments))

    # Aggregate results
    data_series = {}
    failed_tickers = []
    for res in results:
        data_series.update(res[0])
        failed_tickers.extend(res[1])

    if verbose:
        print(f'Completed in {time.time() - start_time:.2f}s. '
              f'Total data: {len(data_series)}, '
              f'Failed tickers: {len(failed_tickers)}')

    return data_series, failed_tickers


def download_constituents(index_symbol: str, api_token: str, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads current and historical constituent data for a given index from EOD Historical Data API

    Supported constituent
    https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds#Current_and_Historical_Index_Constituents_API

    Args:
        index_symbol  (str): The index symbol (e.g., 'GSPC.INDX')
        api_token (str): EOD Historical Data API token
        verbose (bool, optional): Whether to print status messages. Defaults to True

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - current: DataFrame with current index constituents
            - historical: DataFrame with historical index constituents

    Raises:
        RequestException: If there's an HTTP error during the API request
        ValueError: If the API response is missing required data or has inconsistencies
    """
    # Build the API endpoint URL
    endpoint_url = f'https://eodhistoricaldata.com/api/fundamentals/{index_symbol}?api_token={api_token}&fmt=json'

    try:
        # Make the API request
        response = requests.get(endpoint_url)
        response.raise_for_status()
        data = response.json()
    except RequestException as e:
        raise RequestException(f'Failed to retrieve data for index {index_symbol}: {str(e)}')

    # Validate required keys in the response
    required_keys = ['General', 'Components', 'HistoricalTickerComponents']
    missing_keys = [key for key in required_keys if key not in data]

    if missing_keys:
        raise ValueError(f"API response missing required data: {', '.join(missing_keys)}")

    # Extract current and historical constituent data
    current = pd.DataFrame(data['Components']).T
    historical = pd.DataFrame(data['HistoricalTickerComponents']).T

    # Validate that all current constituents exist in historical data
    miss_in_historical = set(current['Code']) - set(historical['Code'])
    if miss_in_historical:
        raise ValueError(f"Some current constituents are missing from historical data: {', '.join(miss_in_historical)}")

    # Get subset of historical data for current constituents
    hist_current = historical[historical['Code'].isin(current['Code'])].copy()

    # Fix inconsistencies between current and historical data
    inconsistent_entries = (hist_current['IsActiveNow'] == 0) | (hist_current['EndDate'].notnull())
    for idx, row in hist_current[inconsistent_entries].iterrows():
        # Case 1: Missing start date but should be active
        if pd.isnull(row['StartDate']):
            historical.loc[idx, 'IsActiveNow'] = 1
            if verbose:
                print(f"Warning: Missing StartDate for active constituent {row['Code']}. Find the correct start date!")

        # Case 2: Has end date but should be active
        elif pd.notnull(row['EndDate']) and row['IsActiveNow'] == 0:
            historical.loc[idx, 'IsActiveNow'] = 1
            historical.loc[idx, 'EndDate'] = None
            if verbose:
                print(f"Fixed: Removed EndDate for active constituent {row['Code']}")

        # Case 3: Other inconsistency
        else:
            if verbose:
                print(f"Warning: Unhandled inconsistency for constituent {row['Code']}. Manual fix required.")

    # Final validation: all current constituents should be marked as active in historical data
    inactive_currents = historical[
        (historical['Code'].isin(current['Code'])) &
        (historical['IsActiveNow'] == 0)
        ]

    if not inactive_currents.empty:
        inactive_codes = inactive_currents['Code'].tolist()
        raise ValueError(f'After fixes, some constituents still marked inactive in historical data: {inactive_codes}')

    return current, historical


def validate_eod_constituents(
        eod_constituents: pd.DataFrame,
        api_token: str,
        add_us: bool = True,
        verbose: bool = True,
        p: int = 1
) -> dict[str, str]:
    """Validates EOD constituents by checking for available historical data

    Attempts to find valid symbols with pricing data available within the date range
    specified in the 'EndDate' column. Tries both regular and '_old' symbol variants

    Args:
        eod_constituents: DataFrame containing constituent symbols with at least
            'Code' and 'EndDate' columns
        api_token: Authentication token for the EOD data API
        add_us: If True, append '.US' suffix to symbols without exchange designation
        verbose: If True, print status messages for unmatched symbols
        p: Number of parallel processes to use for downloading data

    Returns:
        dictionary mapping original symbol codes to validated EOD symbol formats
        Only valid symbols with available data are included

    Raises:
        ValueError: If required columns are missing or API token is empty
        TypeError: If eod_constituents is not a pandas DataFrame
    """

    def _get_eod_symbol(symbol_: str) -> str:
        """ Format symbol with exchange suffix if needed """
        if '.' in symbol_:  # Symbol already has exchange suffix
            return symbol_
        return f'{symbol_}.US' if add_us else symbol_

    def _get_eod_old_symbol(symbol_: str) -> str:
        """ Create alternative '_old' version of the symbol """
        eod_symbol = _get_eod_symbol(symbol_)
        # Handle edge case where symbol might not have an exchange suffix
        if '.' in eod_symbol:
            tk, exg = eod_symbol.rsplit('.', 1)
            return f'{tk}_old.{exg}'
        return f'{eod_symbol}_old'  # Fallback case

    # Input validation
    if not isinstance(eod_constituents, pd.DataFrame):
        raise TypeError('eod_constituents must be a pandas DataFrame')

    required_columns = ['Code', 'EndDate']
    missing_columns = [col for col in required_columns if col not in eod_constituents.columns]
    if missing_columns:
        raise ValueError(f'DataFrame is missing required columns: {missing_columns}')

    # Create extended symbol list with both regular and '_old' variants
    symbols = eod_constituents['Code'].unique().tolist()
    if verbose:
        print(f'Processing {len(symbols)} unique symbols')
    extended_eod_symbols = [_get_eod_symbol(symbol) for symbol in symbols]
    extended_eod_symbols += [_get_eod_old_symbol(symbol) for symbol in symbols]

    # Download price data for all symbols at once
    prices, _ = download_multi(extended_eod_symbols, api_token, data_type='price', verbose=False, p=p)

    # Process each constituent and find valid symbols
    validated_symbols = {}
    missing_data_count = 0
    for _, row in eod_constituents.iterrows():
        original_code = row['Code']
        symbol = _get_eod_symbol(original_code)
        old_symbol = _get_eod_old_symbol(original_code)

        # Convert EndDate to datetime or set to None if missing
        end_date = pd.to_datetime(row['EndDate']) if not pd.isnull(row['EndDate']) else None

        # Case 1: No end date specified, use symbol with data
        if end_date is None:
            if symbol in prices:
                validated_symbols[original_code] = symbol
                continue

        # Case 2: End date specified, check if data covers the date
        else:
            # Try regular symbol first
            if symbol in prices and not prices[symbol].empty:
                price_range = prices[symbol].index
                if price_range.min() <= end_date <= price_range.max():
                    validated_symbols[original_code] = symbol
                    continue

            # Try old symbol if regular one doesn't work
            if old_symbol in prices and not prices[old_symbol].empty:
                price_range = prices[old_symbol].index
                if price_range.min() <= end_date <= price_range.max():
                    validated_symbols[original_code] = old_symbol
                    continue

        # No valid data found for this symbol
        missing_data_count += 1
        if verbose:
            print(f'No valid data found for {original_code} at date {end_date}')

    # Summary report
    if verbose:
        valid_count = len(validated_symbols)
        total_count = len(eod_constituents)
        print(f'Validation complete: {valid_count}/{total_count} symbols validated ({valid_count / total_count:.1%})')

        if missing_data_count > 0:
            print(f'{missing_data_count} symbols could not be validated')

    return validated_symbols
