import pandas as pd

from .fredapi import Fred


class FREDDownloader:
    def __init__(self, api_key: str):
        """
        Initialize FRED API client

        Args:
            api_key (str): FRED API key

        To get a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
        """

        self.fred = Fred(api_key=api_key)

        # Define the indicators to download
        self.direct_indicators = {
            'DTB4WK': '4-Week Treasury Bill: Secondary Market Rate',
            'DTB3': '3-Month Treasury Bill: Secondary Market Rate',
            'DTB6': '6-Month Treasury Bill: Secondary Market Rate',
            'DGS5': '5-Year Treasury Constant Maturity Rate',
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DAAA': "Moody's Seasoned Aaa Corporate Bond Yield",
            'DBAA': "Moody's Seasoned Baa Corporate Bond Yield",
            'DGS3MO': 'Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity',
            'DGS6MO': 'Market Yield on U.S. Treasury Securities at 6-Month Constant Maturity',
            'DGS1': 'Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity',
            # 'CTB3M': 'Change in 3-Month Treasury Constant Maturity Rate',
            # 'CTB6M': 'Change in 6-Month Treasury Constant Maturity Rate',
            # 'CTB1Y': 'Change in 1-Year Treasury Constant Maturity Rate',
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',  # WTI Oil price
            'DCOILBRENTEU': 'Crude Oil Prices: Brent - Europe',  # Brent Oil price,
        }

        # Define calculated indicators (spreads)
        self.calculated_indicators = {
            'TE1': {'formula': 'DGS10 - DTB4WK', 'description': 'Term Spread: 10Y-4W'},
            'TE2': {'formula': 'DGS10 - DTB3', 'description': 'Term Spread: 10Y-3M'},
            'TE3': {'formula': 'DGS10 - DTB6', 'description': 'Term Spread: 10Y-6M'},
            'TE5': {'formula': 'DTB3 - DTB4WK', 'description': 'Term Spread: 3M-4W'},
            'TE6': {'formula': 'DTB6 - DTB4WK', 'description': 'Term Spread: 6M-4W'},
            'DE1': {'formula': 'DBAA - DAAA', 'description': 'Default Spread: Baa-Aaa'},
            'DE2': {'formula': 'DBAA - DGS10', 'description': 'Default Spread: Baa-10Y'},
            'DE4': {'formula': 'DBAA - DTB6', 'description': 'Default Spread: Baa-6M'},
            'DE5': {'formula': 'DBAA - DTB3', 'description': 'Default Spread: Baa-3M'},
            'DE6': {'formula': 'DBAA - DTB4WK', 'description': 'Default Spread: Baa-4W'}
        }

        # Define columns that should be converted to percentage change
        self.pct_change_columns = ['DCOILWTICO', 'DCOILBRENTEU']

    def _download_direct_indicators(self, start_date: str = '2000-01-01', end_date: str = None):
        """
        Download direct FRED indicators

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (None for latest)

        Returns:
            pd.DataFrame: DataFrame with all direct indicators

        Raises:
            ValueError: If there's an error downloading data from FRED API for any series,
            including network issues, invalid series IDs, or API rate limits
        """
        data = {}

        for series_id in self.direct_indicators.keys():
            try:
                series_data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                data[series_id] = series_data
            except Exception as e:
                raise ValueError(f"Error downloading {series_id}: {str(e)}")

        data = pd.concat(data, axis=1).sort_index(ascending=True)
        data.index.name = 'date'
        return data

    def _calculate_spreads(self, data):
        """
        Calculate spread indicators from base data

        Args:
            data (pd.DataFrame): DataFrame containing base indicators

        Returns:
            pd.DataFrame: DataFrame with calculated spreads

        Raises:
            ValueError: If required series are missing or formula is invalid.
        """
        spread_data = pd.DataFrame(index=data.index)

        for spread_id, info in self.calculated_indicators.items():
            formula = info['formula']

            # Parse the formula (simple subtraction only)
            if ' - ' in formula:
                series1, series2 = formula.split(' - ')
                series1, series2 = series1.strip(), series2.strip()

                if series1 in data.columns and series2 in data.columns:
                    spread_data[spread_id] = data[series1] - data[series2]
                else:
                    missing = [s for s in [series1, series2] if s not in data.columns]
                    raise ValueError(f'Missing required series for {spread_id}: {missing}')
            else:
                raise ValueError(f'Invalid formula for {spread_id}: {formula}')

        return spread_data

    def download_all_indicators(self, start_date: str = '2000-01-01', end_date: str = None) -> pd.DataFrame:
        """
        Download all indicators (direct + calculated)

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (None for latest)

        Returns:
            pd.DataFrame: Complete dataset with all indicators

        Raises:
            ValueError: If there's an error downloading or processing the data.
        """
        # Download direct indicators
        direct_data = self._download_direct_indicators(start_date, end_date)

        # Calculate spreads
        spread_data = self._calculate_spreads(direct_data)

        # Calculate relative change in oil prices and market yield on U.S. treasury securities
        direct_data[self.pct_change_columns] = direct_data[self.pct_change_columns].pct_change(fill_method=None)

        # Combine all data
        all_data = pd.concat([direct_data, spread_data], axis=1)

        # Sort by date and ensure proper index name
        all_data = all_data.sort_index(ascending=True)
        all_data.index.name = 'date'

        return all_data
