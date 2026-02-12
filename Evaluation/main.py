import json
import os

import numpy as np
import pandas as pd

import quantstats as qs
from features.alpha158 import calculate_alpha158
from features.alpha360 import calculate_alpha360
from features.technical import calculate_technicals, calculate_market_information
from networks.correlation import build_pos_neg_correlation_adjacency_matrix
from networks.incidence_matrix import align_and_concat_matrices
from networks.incidence_matrix import create_incidence_from_sector_industry, create_incidence_from_wikidata
from networks.knowledge_graph import build_wikidata_adjacency_matrix, find_ticker_wikidata_mapping
from networks.knowledge_graph import get_first_order_relation_indices
from networks.sector import build_sector_industry_adjacency_matrix
from portfolio.optimization import OptimizationMethod, RiskMethod, ReturnMethod, ObjectiveFunction
from portfolio.backtest import run_portfolio_backtest, find_nearest_min
from portfolio.returns import portfolio_daily_returns
from portfolio.io import read_portfolio_history, read_portfolio_report
from quantstats import plots as _plots
from utils import eodhd, fred, yahoo
from utils.storage import load_config, read_stock_dataset


def demo_portfolio_daily_return():
    userdata = load_config('config.json')

    # OPTION 1 : from user defined portfolio history
    pf_history = read_portfolio_history('./data/pf/portfolio-sample.csv', end_date='2023-12-31')

    # OPTION 2: from flow library
    pf_history = read_portfolio_report('./data/pf/portfolio_report_.xlsx', 'l10')

    # Download prices
    pf_tickers = list({ticker for snp in pf_history for ticker in snp['tickers']})
    asset_prices, errors = eodhd.download_multi(
        pf_tickers,
        api_token=userdata.get('eodhd'),
        data_type='price',
        verbose=True,
        p=4
    )
    assert len(errors) == 0

    # Compute portfolio daily returns
    pf_returns = portfolio_daily_returns(pf_history, asset_prices)
    return pf_returns


def demo_quantstats(pf_returns: pd.Series):
    # Define benchmark
    benchmark = eodhd.download_returns('SPY', load_config('config.json').get('eodhd'))
    benchmark.name = 'Benchmark'

    # Show sharpe ratio
    print(f'{qs.stats.sharpe(pf_returns):.3f}')

    # Show strategy metrics
    # rf (float): Risk-free rate expressed as a yearly (annualized) return
    # mode (str): 'basic | full"
    qs.reports.metrics(pf_returns, benchmark, mode='basic', rf=0.0, compounded=True, display=True)

    # # Worst 5 Drawdown
    # returns = pf_returns.copy()
    # dd = qs.stats.to_drawdown_series(returns)
    # dd_info = qs.stats.drawdown_details(dd).sort_values('max drawdown')
    # print(dd_info.head(4))

    _plots.returns(
        pf_returns,
        benchmark=benchmark,
        grayscale=False,
        figsize=(8, 4),
        show=True,
        ylabel='',
        prepare_returns=False,
        fontname=None
    )

    _plots.yearly_returns(
        pf_returns,
        benchmark=benchmark,
        grayscale=False,
        figsize=(8, 4),
        show=True,
        ylabel='',
        prepare_returns=False,
        fontname=None
    )


def demo_portfolio_optimization():
    userdata = load_config('config.json')

    # Define and download ticker returns
    tickers = ["GOOG", "AAPL", "META", "BABA", "AMZN", "GE", "AMD"]
    asset_returns, errors = eodhd.download_multi(tickers, api_token=userdata.get('eodhd'), data_type='returns', p=4)
    assert len(errors) == 0

    dt = find_nearest_min(asset_returns)
    print(f"\nCommon starting date for {len(asset_returns)} assets: {dt.strftime('%Y-%m-%d')}")

    optimized_pf = run_portfolio_backtest(
        asset_returns,
        '2019-01-01',
        '2019-06-01',
        '3M',
        OptimizationMethod.MVO,
        RiskMethod.LEDOIT_WOLF,
        ReturnMethod.MEAN_HISTORICAL,
        ObjectiveFunction.SHARPE,
        0.2,
        None,
        None,
        0.0,
        True,
        None,
        verbose=True
    )
    optimized_pf.to_csv('./data/pf/portfolio-sample-optimized.csv', index=False, header=False)


def demo_search_constituents():
    userdata = load_config('config.json')

    # Load constituents data and search for each ISIN using EODHD API
    constituents = pd.read_csv('./data/constituents/raw/sxxp.csv')
    results, err = eodhd.download_multi(constituents['ISIN'].tolist(), userdata.get('eodhd'), 'search', True, 6)

    # Extract first search result for each ISIN (assumes single best match)
    processed_results = {k: l[0] for k, l in results.items()}

    # Convert search results to DataFrame and merge with original constituents data
    results_df = pd.json_normalize(constituents['ISIN'].map(processed_results))
    results_df = results_df.add_suffix('_', axis=1)
    results_df = pd.concat([constituents, results], axis=1)

    # Mapping dictionaries for exchange conversions
    # Note: EODHD does not have IT exchange data, use LSE instead.
    exchange_isin = {'AS': 'NL', 'HE': 'FI', 'PA': 'FR', 'MC': 'ES', 'XETRA': 'DE', 'F': 'DE', 'LSE': 'IE', 'STU': 'DE',
                     'WAR': 'Pl', 'OL': 'NO', 'BR': 'BE', 'CO': 'DK', 'ST': 'SE', 'SW': 'CH'}
    isin_exchange = {'NL': 'AS', 'FI': 'HE', 'FR': 'PA', 'ES': 'MC', 'DE': 'XETRA', 'IE': 'LSE', 'IT': 'LSE',
                     'PL': 'WAR', 'NO': 'OL', 'BE': 'BR', 'DK': 'CO', 'SE': 'ST', 'CH': 'SW'}

    print('Here!')


def demo_validate_constituents():
    userdata = load_config('config.json')

    # Load historical constituents data
    historical = pd.read_csv('./data/constituents/eodhd/sxxp.csv')

    # Check for missing EODHD symbols
    missing_symbols = historical.loc[historical['EODHD'].isnull(), 'Name'].unique().tolist()
    if missing_symbols:
        print(f"Missing EODHD codes for: {missing_symbols}")

    # Get unique symbols that have EODHD codes
    symbols = historical.loc[historical['EODHD'].notnull(), 'EODHD'].unique().tolist()
    prices_dict, errors = eodhd.download_multi(symbols, userdata.get('eodhd'), data_type='price', verbose=False, p=6)

    # Display any download errors
    if errors:
        print(f"Download errors for: {[err['Ticker'] for err in errors]}")

    # Validate date coverage for each constituent
    for i, row in historical.iterrows():
        start_date = pd.to_datetime(row['StartDate']) if pd.notnull(row['StartDate']) else None
        end_date = pd.to_datetime(row['EndDate']) if pd.notnull(row['EndDate']) else None
        code = row['EODHD']

        # Skip if no code or price data not available
        if pd.isnull(code) or code not in prices_dict:
            continue

        price_start = prices_dict[code].index.min()
        price_end = prices_dict[code].index.max()

        # Validate start date coverage (with threshold)
        if start_date and (price_start - start_date).days > 15:
            print(f'Error: Price data starts {price_start} but constituent started {start_date} for {code}')

        # Validate end date coverage (with threshold)
        if end_date and (end_date - price_end).days > 15:
            print(f'Error: Price data ends {price_end} but constituent ended {end_date} for {code}')

        # Additional validation: Check if we have any data for the constituent period (with threshold)
        if start_date and end_date:
            if price_end < start_date or price_start > end_date:
                print(f'Error: No price data overlap for constituent period {start_date} to {end_date} for {code}')


def _format_dataset(data_series: dict[str, pd.DataFrame], start_date: str = '2000-01-01') -> pd.DataFrame:
    """Combines multiple DataFrames into a single DataFrame with MultiIndex.

    Takes a dictionary of DataFrames, adds the dictionary key as an 'instrument' column,
    filters for dates after the specified start_date, and sets a MultiIndex of ['instrument', 'date'].

    Args:
        data_series: Dictionary mapping instrument names to their respective DataFrames.
            Each DataFrame is expected to have a datetime index.
        start_date: String date in format 'YYYY-MM-DD' to filter data from.
            Defaults to '2000-01-01'.

    Returns:
        A combined DataFrame with a MultiIndex of ['instrument', 'date'], sorted by this index.

    Raises:
        ValueError: If data_series is empty or None.
    """
    if not data_series:
        raise ValueError('data_series cannot be empty or None')

    complete_dataset = []
    for instrument_name, df in data_series.items():
        # Make a copy to avoid modifying the original DataFrame
        instrument_df = df.copy()
        instrument_df['instrument'] = instrument_name
        instrument_df = instrument_df[instrument_df.index > start_date]
        instrument_df.index.name = 'date'
        instrument_df = instrument_df.reset_index()
        instrument_df = instrument_df.set_index(['instrument', 'date'])
        complete_dataset.append(instrument_df)

    if not complete_dataset:
        raise ValueError(f'No valid data found after filtering for dates > {start_date}')

    # Concatenate all DataFrames and sort by MultiIndex
    return pd.concat(complete_dataset, axis=0).sort_index()


def _format_universal_features(
        data_series: dict[str, pd.DataFrame],
        feature: str,
        start_date: str = '2000-01-01'
) -> pd.DataFrame:
    """Extracts a specific feature from multiple instruments and reshapes to wide format.

    Args:
        data_series: Dictionary mapping instrument names to their respective DataFrames.
            Each DataFrame is expected to have a datetime index and contain the specified feature.
        feature: Name of the column/feature to extract from each DataFrame.
        start_date: String date in format 'YYYY-MM-DD' to filter data from.
            Defaults to '2000-01-01'.

    Returns:
        DataFrame with dates as index and instruments as columns, containing the specified feature values.

    Raises:
        ValueError: If data_series is empty or None, or if no valid data found after filtering.
        KeyError: If the specified feature is not found in the DataFrames.
    """
    # Get formatted dataset with MultiIndex
    df = _format_dataset(data_series, start_date)
    # Extract the specific feature
    df = df[feature]
    # Reshape data: instruments become columns, dates remain as index
    res_df = df.unstack(level='instrument')
    return res_df


def demo_download_dataset(universe: str):
    userdata = load_config('config.json')
    out_dir = os.path.join('./data/dataset', universe)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Read historical constituents
    historical = pd.read_csv(f'./data/constituents/eodhd/{universe}.csv')

    # Step 2: Check null symbols
    missing_symbols = historical.loc[historical['EODHD'].isnull(), 'Name'].unique().tolist()
    if missing_symbols:
        print(f"Missing EODHD codes for: {missing_symbols}")

    symbols = historical.loc[historical['EODHD'].notnull(), 'EODHD'].unique().tolist()

    # Step 3: Download prices
    prices_dict, errors = eodhd.download_multi(symbols, userdata.get('eodhd'), data_type='price', verbose=True, p=4)
    prices = _format_dataset(prices_dict)
    prices.to_csv(os.path.join(out_dir, f'{universe}.csv'), index=True)

    # (Optional) Download general company details
    info, errors = eodhd.download_multi(symbols, userdata.get('eodhd'), data_type='info', verbose=True, p=4)
    info = pd.concat(info.values(), axis=0)
    info.to_csv(os.path.join(out_dir, f'{universe}_info.csv'), index=True)

    # (Optional) Download market cap series
    market_caps, errors = eodhd.download_multi(symbols, userdata.get('eodhd'), 'market_cap', verbose=True, p=4)
    _format_dataset(market_caps).to_csv(os.path.join(out_dir, f'{universe}_market_cap.csv'), index=True)

    print('Here!')


def demo_download_news():
    userdata = load_config('config.json')
    news_dir = './data/news/'
    os.makedirs(news_dir, exist_ok=True)

    # Collect all tickers
    universes = ['sp500', 'nasdaq100', 'dji', 'sx5e', 'sxxp']
    symbols = set()
    for universe in universes:
        df = pd.read_csv(f'./data/constituents/eodhd/{universe}.csv')
        symbols.update(df['EODHD'].dropna().unique())

    # Download only missing tickers
    existing_files = {file.strip('.json') for file in os.listdir(news_dir) if file.endswith('.json')}
    to_download = symbols - existing_files
    # to_download = to_download[0:200]  # download 200 at a time

    # Download news
    news, errors = eodhd.download_multi(list(to_download), userdata.get('eodhd'), data_type='news', verbose=True, p=1)

    # Handle errors
    if errors:
        print(f"Download errors for: {[err['Ticker'] for err in errors]}")

    # Save successful downloads
    if news:
        for ticker, ticker_news in news.items():
            output_file = os.path.join(news_dir, f'{ticker}.json')
            with open(output_file, 'w') as f:
                json.dump(ticker_news, f)

    print('Here!')


def demo_market_information():
    userdata = load_config('config.json')
    out_dir = os.path.join('./data/dataset')
    os.makedirs(out_dir, exist_ok=True)

    us_index_symbols = ['GSPC.INDX', 'DJI.INDX', 'NDX.INDX']
    eu_index_symbols = ['SXXP.INDX', 'SX5E.INDX']
    for region, symbols in zip(['us', 'eu'], [us_index_symbols, eu_index_symbols]):
        # Download market data
        index_series, _ = eodhd.download_multi(symbols, userdata.get('eodhd'), data_type='price', verbose=True, p=1)
        df = _format_dataset(index_series)
        df.to_csv(os.path.join(out_dir, f'{region}_indexes.csv'), index=True)

        # Compute market information indicators
        df_market = calculate_market_information(df, symbols)
        df_market.to_csv(os.path.join(out_dir, f'{region}_market.csv'), index=True)

    print('Here!')


def demo_feature_extraction(universe: str):
    out_dir = os.path.join('./data/dataset', universe)

    # Read OHLC dataset
    prices = read_stock_dataset(os.path.join(out_dir, f'{universe}.csv'))

    # Compute alpha158
    df_alpha158 = calculate_alpha158(prices)
    df_alpha158.to_csv(os.path.join(out_dir, f'{universe}_alpha158.csv'), index=True)

    # Compute alpha360
    df_alpha360 = calculate_alpha360(prices)
    df_alpha360.to_csv(os.path.join(out_dir, f'{universe}_alpha360.csv'), index=True)

    # Compute technical indicators
    df_techs = calculate_technicals(prices)
    df_techs.to_csv(os.path.join(out_dir, f'{universe}_tech.csv'), index=True)

    print('Here!')


def demo_compute_relation(universe: str):
    out_dir = os.path.join('./data/dataset', universe)

    # Read OHLC dataset
    prices = read_stock_dataset(os.path.join(out_dir, f'{universe}.csv'))
    tickers = prices.index.get_level_values('instrument').unique().tolist()

    # Read company names
    info = pd.read_csv(os.path.join(out_dir, f'{universe}_info.csv'))
    assert info['instrument'].is_unique and info['Name'].notnull().all()
    ticker_to_name = info.set_index('instrument')['Name'].str.strip().to_dict()
    tck_to_sector = info.set_index('instrument')['GicSector'].dropna().to_dict()
    tck_to_industry = info.set_index('instrument')['GicIndustry'].dropna().to_dict()

    # 1. INDUSTRY SECTOR RELATIONS
    out_file = os.path.join(out_dir, f'{universe}_sector_industry_matrix.npz')
    adj_mat, rels, tcks = build_sector_industry_adjacency_matrix(tickers, tck_to_sector, tck_to_industry, True)
    np.savez_compressed(out_file, adj_matrix=adj_mat, dates=np.nan, relations=rels, tickers=tcks)

    # 2. THGNN RELATIONS
    out_file = os.path.join(out_dir, f'{universe}_corr_matrix.npz')
    features, window_size, th = ['adj_high', 'adj_low', 'adj_close', 'adj_open', 'volume'], 20, 0.6
    adj_mat, dates, rels, tcks = build_pos_neg_correlation_adjacency_matrix(prices, features, window_size, th)
    np.savez_compressed(out_file, adj_matrix=adj_mat, dates=pd.to_datetime(dates).values, relations=rels, tickers=tcks)

    # 3. WIKIDATA RELATIONS
    wiki_chk = './data/wikidata_chk/'
    wikidata_file = os.path.join(out_dir, f'{universe}_wikidata.json')
    out_file = os.path.join(out_dir, f'{universe}_wikidata_matrix.npz')
    # Finds Wikidata IDs for tickers
    if not os.path.exists(wikidata_file):
        ticker_to_wikidata = find_ticker_wikidata_mapping(ticker_to_name, verbose=True)
        json.dump(ticker_to_wikidata, open(wikidata_file, 'w'), indent=2)
    else:
        ticker_to_wikidata = json.load(open(wikidata_file, 'r'))
    # Compute Wikidata adjacency matrix
    adj_mat, dates, rels, tcks = build_wikidata_adjacency_matrix(prices, ticker_to_wikidata, wiki_chk, 'MS', True)
    np.savez_compressed(out_file, adj_matrix=adj_mat, dates=pd.to_datetime(dates).values, relations=rels, tickers=tcks)

    print('Here!')


def demo_incidence_matrix(universe: str):
    out_dir = os.path.join('./data/dataset', universe)

    sect_file = os.path.join(out_dir, f'{universe}_sector_industry_matrix.npz')
    wiki_file = os.path.join(out_dir, f'{universe}_wikidata_matrix.npz')

    wiki_date_idx = -1
    out_inc_file = os.path.join(out_dir, f'{universe}_inc_matrix.npz')

    # Load Sector & Industry adjacency matrix and compute sector incidence matrix
    with np.load(sect_file) as data:
        sect = {
            'adj_matrix': data['adj_matrix'],
            'relations': data['relations'],
            'tickers': data['tickers'],
        }
    sect_inc_matrix = create_incidence_from_sector_industry(adj_matrix=sect['adj_matrix'])

    # Load Wikidata adjacency matrix and compute incidence matrix
    with np.load(wiki_file) as data:
        wiki = {
            'adj_matrix': data['adj_matrix'],
            'dates': data['dates'],
            'relations': data['relations'],
            'tickers': data['tickers'],
        }
    first_order_indices = get_first_order_relation_indices(wiki['relations'])
    wiki_inc_matrix = create_incidence_from_wikidata(wiki['adj_matrix'][wiki_date_idx], first_order_indices)

    # Align sector and wikidata incidence matrices
    inc_matrix, tickers, relations = align_and_concat_matrices(
        matrix1=sect_inc_matrix,
        tickers1=sect['tickers'],
        categories1=sect['relations'],
        matrix2=wiki_inc_matrix,
        tickers2=wiki['tickers'],
        categories2=[f'E{i}' for i in range(wiki_inc_matrix.shape[1])],
    )
    np.savez_compressed(out_inc_file, inc_matrix=inc_matrix, relations=relations, tickers=tickers)
    print('Here!')


def cnnpred_market_features():
    userdata = load_config('config.json')

    # Download commodity futures returns (Copper, Natural Gas, Gold, Silver)
    df_commodities = yahoo.download_commodities_returns(['HG=F', 'NG=F', 'GC=F', 'SI=F'])

    # Download economic indicators from FRED (Federal Reserve Economic Data)
    downloader = fred.FREDDownloader(api_key=userdata.get('fred'))
    df_fred = downloader.download_all_indicators(start_date='2000-01-01')

    # Download global financial instruments: world indices, forex rates, USD index, and US stocks
    world_indices = ['GSPC.INDX', 'IXIC.INDX', 'DJI.INDX', 'NYA.INDX', 'HSI.INDX', '000001.SHG', 'FCHI.INDX',
                     'GDAXI.INDX', 'IWM.US', 'ISF.LSE']  # Note: updated RUT.INDX to IWM.US ETF and FTSE.INDX to ISF.LSE
    exchange_rate = ['USDJPY.FOREX', 'USDGBP.FOREX', 'USDCAD.FOREX', 'USDCNY.FOREX', 'USDAUD.FOREX', 'USDNZD.FOREX',
                     'USDCHF.FOREX', 'USDEUR.FOREX', 'XAUUSD.FOREX', 'XAGUSD.FOREX']
    us_dollar_index = ['DXY.INDX']
    us_companies = ['XOM.US', 'JPM.US', 'AAPL.US', 'MSFT.US', 'GE.US', 'JNJ.US', ' WFC.US', 'AMZN.US']

    tickers = world_indices + exchange_rate + us_dollar_index + us_companies
    data_series, _ = eodhd.download_multi(tickers, userdata.get('eodhd'), data_type='price', verbose=False, p=6)
    df_us_indx_forex = _format_universal_features(data_series, 'adj_close')
    df_us_indx_forex = df_us_indx_forex.pct_change(fill_method=None)

    # Combine all datasets and save to CSV for CNNPred model
    df = pd.concat([df_fred, df_commodities, df_us_indx_forex], axis=1).sort_index(ascending=True)
    df.to_csv('./data/dataset/cnnpred_market.csv', index=True)

    print('Here!')


if __name__ == '__main__':
    # Download constituents
    # demo_search_constituents()
    # demo_validate_constituents()

    for u in ['sp500', 'nasdaq100', 'dji', 'sx5e', 'sxxp']:
        demo_download_dataset(u)
        demo_feature_extraction(u)
        demo_compute_relation(u)
        demo_incidence_matrix(u)

        # Download news
        demo_download_news()

        # Download market features
        demo_market_information()
        cnnpred_market_features()

