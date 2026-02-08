from typing import Optional, Any
import time
import os
import pickle
from itertools import permutations

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

from utils.validation import validate_prices_decorator


class WikidataEntityConnector:
    """
    A class to extract connections between pairs of entities in Wikidata.
    Focuses on first-order direct connections (A->B) and
    second-order connections (A->C->B) where C is an intermediary entity.
    """

    def __init__(self, language: str = 'en', checkpoint_dir: Optional[str] = None):
        """
        Initialize the WikidataEntityConnector.

        Args:
            language: Language code for labels and descriptions
            checkpoint_dir: Directory to store persistent caches. If None, uses in-memory caching only.
        """
        self.language = language
        self.api_url = 'https://www.wikidata.org/w/api.php'
        self.entity_url = 'https://www.wikidata.org/wiki/Special:EntityData/'
        self.session = requests.Session()

        # Set up checkpoint directory
        # Set up checkpoint directory (optional)
        self.checkpoint_dir = None
        self.use_persistent_cache = checkpoint_dir is not None

        if self.use_persistent_cache:
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Cache files
            self.entity_cache_file = os.path.join(self.checkpoint_dir, 'entity_cache.pkl')

            # Load existing caches
            self.entity_cache = self._load_cache(self.entity_cache_file)
        else:
            # Use in-memory caching only
            self.entity_cache = {}

        # Track cache modifications for periodic saves (only relevant if using persistent cache)
        self._entity_cache_modified = False
        self._save_counter = 0

    def _load_cache(self, cache_file: str) -> dict:
        """ Load cache from file if it exists """
        if not self.use_persistent_cache:
            return {}

        if os.path.isfile(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                return cache
            except Exception as e:
                raise ValueError(f'Error loading cache {os.path.basename(cache_file)}: {e}')
        return {}

    def _save_cache(self, cache: dict, cache_file: str):
        """ Save cache to file """
        if not self.use_persistent_cache:
            return

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f'Error saving cache {os.path.basename(cache_file)}: {e}')

    def _periodic_save(self, force: bool = False):
        """Save caches periodically to avoid losing progress."""
        if not self.use_persistent_cache:
            return

        self._save_counter += 1

        # Save every 10 API calls or when forced
        if self._save_counter % 10 == 0 or force:
            if self._entity_cache_modified:
                self._save_cache(self.entity_cache, self.entity_cache_file)
                self._entity_cache_modified = False

    def search_entity(self, entity_name: str, limit: int = 5) -> list[dict]:
        """
        Search for entities in Wikidata by name.

        Args:
            entity_name: Name of the entity to search
            limit: Maximum number of results to return

        Returns:
            List of matching entities with their IDs and descriptions
        """
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': entity_name,
            'language': self.language,
            'limit': limit
        }

        response = self.session.get(self.api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            results = []

            for item in data.get('search', []):
                results.append({
                    'id': item.get('id'),
                    'label': item.get('label', ''),
                    'description': item.get('description', ''),
                    'url': item.get('concepturi', '')
                })

            return results
        else:
            # print(f"Error searching for entity '{entity_name}': {response.status_code}")
            return []

    def fulltext_search_entity(self, entity_name: str, limit: int = 5) -> list[dict]:
        """
        Search for entities in Wikidata by name with a near-match search type.

        The action query allows some "fuzziness" in the search term
        https://stackoverflow.com/questions/51949112/getting-search-results-from-wikidata-website-but-not-api

        Args:
            entity_name: Name of the entity to search
            limit: Maximum number of results to return

        Returns:
            List of matching entities with their IDs and descriptions
        """
        params = {
            'action': 'query',
            'list': 'search',
            'format': 'json',
            'srsearch': entity_name,  # Full-text search query
            'srlimit': limit,  # Number of results
            'srprop': 'snippet|titlesnippet',
            'srsort': 'relevance',  # Sort by relevance
        }

        response = self.session.get(self.api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            results = []

            for item in data.get('query', []).get('search', []):
                label = BeautifulSoup(item.get('titlesnippet', ''), 'html.parser').get_text()
                results.append({
                    'id': item.get('title'),
                    'label': label,
                    'description': item.get('snippet', ''),
                    'url': ''
                })

            return results
        else:
            # print(f"Error searching for entity '{entity_name}': {response.status_code}")
            return []

    def get_entity_data(self, entity_id: str) -> dict:
        """
        Retrieve complete data for a Wikidata entity.
        Caches results to avoid repeated requests.

        Args:
            entity_id: Wikidata ID (e.g., "Q312")

        Returns:
            Dictionary containing entity data
        """
        # Check if we already have this entity cached
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]

        url = f'{self.entity_url}{entity_id}.json'
        response = self.session.get(url)

        if response.status_code == 200:
            data = response.json()
            entity_data = data.get('entities', {}).get(entity_id, {})

            # Store in cache
            self.entity_cache[entity_id] = entity_data
            if self.use_persistent_cache:
                self._entity_cache_modified = True
                self._periodic_save()

            return entity_data
        else:
            # print(f"Error retrieving entity '{entity_id}': {response.status_code}")
            return {}

    # def get_property_label(self, property_id: str) -> str:
    #     """
    #     Get the human-readable label for a Wikidata property.
    #
    #     Args:
    #         property_id: Wikidata property ID (e.g., "P31")
    #
    #     Returns:
    #         Human-readable label for the property
    #     """
    #
    #     # Retrieve property data
    #     params = {
    #         'action': 'wbgetentities',
    #         'format': 'json',
    #         'ids': property_id,
    #         'languages': self.language,
    #         'props': 'labels'
    #     }
    #
    #     response = self.session.get(self.api_url, params=params)
    #
    #     if response.status_code == 200:
    #         data = response.json()
    #         label = (data.get('entities', {})
    #                  .get(property_id, {})
    #                  .get('labels', {})
    #                  .get(self.language, {})
    #                  .get('value', property_id))
    #
    #         return label
    #     else:
    #         # print(f"Error retrieving property '{property_id}': {response.status_code}")
    #         return property_id

    def extract_time_qualifiers(self, claim):
        """
        Extract time-related information from a claim. This includes:
        1. Qualifiers (most common for temporal info)
        2. References with time stamps
        3. Rank-based timing (for deprecated/preferred statements)

        Args:
            claim: Wikidata claim object

        Returns:
            Dictionary with time information if available
        """
        time_info = {}

        # Common time-related qualifier properties in Wikidata
        time_properties = {
            'P580': 'start_time',
            'P582': 'end_time',
            'P585': 'point_in_time',
            'P1319': 'earliest_date',
            'P1326': 'latest_date',
            'P813': 'retrieved_date',
            'P577': 'publication_date',
        }

        # 1. Extract from QUALIFIERS (most common)
        if 'qualifiers' in claim:
            for prop_id, qualifier_values in claim['qualifiers'].items():
                if prop_id in time_properties:
                    qualifier_name = time_properties[prop_id]

                    for qualifier in qualifier_values:
                        if qualifier.get('snaktype') == 'value' and qualifier.get('datatype') == 'time':
                            time_value = qualifier.get('datavalue', {}).get('value', {}).get('time')
                            if time_value:
                                time_info[qualifier_name] = self._format_time(time_value)
                                break  # Take the first value for each qualifier type

        # # 2. Extract from REFERENCES (look for stated in dates, publication dates, etc.)
        # if 'references' in claim:
        #     for ref_group in claim['references']:
        #         if 'snaks' in ref_group:
        #             for prop_id, ref_values in ref_group['snaks'].items():
        #                 if prop_id in time_properties:
        #                     ref_name = f'reference_{time_properties[prop_id]}'
        #
        #                     for ref_value in ref_values:
        #                         if ref_value.get('snaktype') == 'value' and ref_value.get('datatype') == 'time':
        #                             time_value = ref_value.get('datavalue', {}).get('value', {}).get('time')
        #                             if time_value:
        #                                 time_info[ref_name] = self._format_time(time_value)
        #                                 break

        return time_info

    def _format_time(self, time_value):
        """ Convert Wikidata time format (+1967-01-17T00:00:00Z) to YYYY-MM-DD """
        # Strip the leading + and trailing T00:00:00Z
        try:
            # Remove leading + if present
            if time_value.startswith('+'):
                time_value = time_value[1:]

            # Split on 'T' to get date part
            date_part = time_value.split('T')[0]

            # Handle cases where precision is less than day (year or month only)
            # Wikidata uses -00 for unknown month/day
            if date_part.endswith('-00-00'):
                return f'{date_part[:-6]}-01-01'  # Return just the year
            elif date_part.endswith('-00'):
                return f'{date_part[:-3]}-01'  # Return year-month
            else:
                return date_part  # Return full date

        except Exception as e:
            # If parsing fails, return original format
            raise ValueError('Could not parse time from Wikidata time: {}'.format(e))

    def find_first_order_connections(self, entity_a_id: str, entity_b_id: str) -> list[dict]:
        """
        Find direct connections between two entities (first-order relations: A->B).

        Args:
            entity_a_id: Wikidata ID for entity A
            entity_b_id: Wikidata ID for entity B

        Returns:
            List of connections with their properties and directions
        """
        # Get full data for both entities
        entity_a = self.get_entity_data(entity_a_id)

        connections = []

        # Find connections where A points to B
        if 'claims' in entity_a:
            for prop_id, claims in entity_a.get('claims', {}).items():
                for claim in claims:
                    # Check if this claim points to entity B
                    if claim.get('mainsnak', {}).get('snaktype') == 'value':
                        if claim.get('mainsnak', {}).get('datatype') == 'wikibase-item':
                            target_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')

                            if target_id == entity_b_id:
                                # # We found a direct connection from A to B
                                # prop_label = self.get_property_label(prop_id)

                                # Extract temporal qualifiers if any
                                time_info = self.extract_time_qualifiers(claim)

                                connections.append({
                                    'type': 'first-order',
                                    'direction': 'A->B',
                                    'relation_id': f'{prop_id}',
                                    # 'relation_label': prop_label,
                                    'pattern': f'(A)-[{prop_id}]->(B)',
                                    'temporal_info': time_info
                                })

        return connections

    def find_second_order_connections(
            self,
            entity_a_id: str,
            entity_b_id: str,
            max_intermediate: int = 100
    ) -> list[dict]:
        """
        Find second-order connections between entities of the form (A->C<-B)
        where C is a common entity that both A and B point to.

        Args:
            entity_a_id: Wikidata ID for entity A
            entity_b_id: Wikidata ID for entity B
            max_intermediate: Maximum number of intermediate entities to check

        Returns:
            List of second-order connections with common target entities
        """
        # Get full data for entity A
        entity_a = self.get_entity_data(entity_a_id)
        connections = []

        # Track entities that A points to
        a_targets = {}

        # Find entities that A points to and store the property used
        if 'claims' in entity_a:
            for prop_id, claims in entity_a.get('claims', {}).items():
                for claim in claims:
                    if claim.get('mainsnak', {}).get('snaktype') == 'value':
                        if claim.get('mainsnak', {}).get('datatype') == 'wikibase-item':
                            target_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')

                            # Don't include direct connections to B (that's first-order)
                            if target_id != entity_b_id:
                                if target_id not in a_targets:
                                    a_targets[target_id] = []

                                # Store both property ID and claim object (for temporal data)
                                a_targets[target_id].append((prop_id, claim))

        # Get full data for entity B
        entity_b = self.get_entity_data(entity_b_id)

        # Track entities that B points to
        b_targets = {}

        # Find entities that B points to and store the property used
        if 'claims' in entity_b:
            for prop_id, claims in entity_b.get('claims', {}).items():
                for claim in claims:
                    if claim.get('mainsnak', {}).get('snaktype') == 'value':
                        if claim.get('mainsnak', {}).get('datatype') == 'wikibase-item':
                            target_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')

                            # Don't include direct connections to A (that's first-order)
                            if target_id != entity_a_id:
                                if target_id not in b_targets:
                                    b_targets[target_id] = []

                                # Store both property ID and claim object (for temporal data)
                                b_targets[target_id].append((prop_id, claim))

        # Find common targets (entities that both A and B point to)
        common_targets = set(a_targets.keys()).intersection(set(b_targets.keys()))

        # Limit the number of common targets to process
        common_targets_list = list(common_targets)[:max_intermediate]

        # Process each common target
        for target_id in common_targets_list:
            # Get target entity data
            target_entity = self.get_entity_data(target_id)
            target_label = target_entity.get('labels', {}).get(self.language, {}).get('value', target_id)

            # For each property-claim pair from A to target
            for r1_id, a_claim in a_targets[target_id]:
                # r1_label = self.get_property_label(r1_id)

                # Extract temporal qualifiers for A->C relation
                a_time_info = self.extract_time_qualifiers(a_claim)

                # For each property-claim pair from B to target
                for r2_id, b_claim in b_targets[target_id]:
                    # r2_label = self.get_property_label(r2_id)

                    # Extract temporal qualifiers for B->C relation
                    b_time_info = self.extract_time_qualifiers(b_claim)

                    # We found a pattern: A->C<-B
                    connections.append({
                        'type': 'second-order',
                        'pattern_type': 'common-target',
                        'common_entity': f'{target_id} ({target_label})',
                        'relation_id': f'{r1_id}_{r2_id}',
                        # 'relation_a_to_c': f'{r1_id} ({r1_label})',
                        # 'relation_b_to_c': f'{r2_id} ({r2_label})',
                        'pattern': f'(A)-[{r1_id}]->(C)<-[{r2_id}]-(B)',
                        'temporal_info_a_to_c': a_time_info,
                        'temporal_info_b_to_c': b_time_info
                    })

            # Pause briefly to avoid hitting rate limits
            time.sleep(0.1)

        return connections

    def find_company_wikidata_id(self, company: str) -> Optional[str]:
        """
        Search for a company's Wikidata entity ID with a simple fallback strategies

        Args:
            company: Company name to search for

        Returns:
            Wikidata entity ID if found with sufficient confidence, None otherwise
        """

        # entities_a = self.search_entity(company_a, limit=5)
        entities_a = self.fulltext_search_entity(company, limit=5)
        if not entities_a:
            entities_a = self.fulltext_search_entity(company.rsplit(maxsplit=1)[0], limit=5)
            if not entities_a:
                # print(f'Could not find entity for "{company}"')
                return ''

        return entities_a[0]['id']

    def analyze_wikidata_pair(self, entity_a_id: str, entity_b_id: str):
        """
        Analysis of relationships between two wikidata id.
        It finds both first and second-order connections.

        Args:
            entity_a_id: Name of first wikidata id (company)
            entity_b_id: Name of second wikidata id (company)

        Returns:
            Dictionary with entity information and connections
        """
        result = {
            'first_order_connections': [],
            'second_order_connections': []
        }

        if not entity_a_id or not entity_b_id:
            return result

        # Find first-order connections
        first_order = self.find_first_order_connections(entity_a_id, entity_b_id)
        result['first_order_connections'] = first_order

        # Find second-order connections (limit the search to avoid excessive requests)
        second_order = self.find_second_order_connections(entity_a_id, entity_b_id, max_intermediate=50)
        result['second_order_connections'] = second_order

        return result


SELECTED_PATHS = [
    'P355_P355', 'P169_P169', 'P400_P1056', 'P3320_P169', 'P1830_P749', 'P31_P452', 'P452_P452', 'P452_P1056',
    'P121_P121', 'P31_P1056', 'P366_P31', 'P1056_P121', 'P127_P749', 'P1056_P1056', 'P1056_P400', 'P306_P1056', 'P127',
    'P452_P31', 'P127_P355', 'P166_P166', 'P169_P112', 'P169_P3320', 'P1830_P127', 'P127_P1830', 'P112_P127', 'P155',
    'P1056_P306', 'P452_P2770', 'P127_P127', 'P361_P361', 'P749_P127', 'P355_P155', 'P463_P463', 'P2770_P452', 'P749',
    'P127_P112', 'P355_P127', 'P127_P3320', 'P156', 'P155_P155', 'P355_P199', 'P112_P169', 'P121_P1056', 'P31_P366',
    'P155_P355', 'P114_P114', 'P749_P1830', 'P355', 'P112_P112', 'P169_P127', 'P1056_P31', 'P127_P169', 'P113_P113',
    'P1056_P452', 'P1344_P1344', 'P199_P355', 'P3320_P127'
]


def find_ticker_wikidata_mapping(
        ticker_to_company: dict[str, str],
        verbose: bool = False
) -> dict[str, str]:
    """ Finds Wikidata IDs for tickers based on company names

    Args:
      ticker_to_company: Dictionary mapping ticker symbols to company names. Example: {"AAPL": "Apple Inc.", "GOOGL": ...}
      verbose: If True, print progress bar

    Returns:
      Dictionary mapping ticker symbols to Wikidata entity IDs
          Example: {"AAPL": "Q312", "GOOGL": "Q95"}

    Raises:
      ConnectionError: If unable to connect to Wikidata API.
      ValueError: If company_names is empty or contains invalid data.
    """
    if not ticker_to_company:
        raise ValueError("company_names dictionary cannot be empty")

    connector = WikidataEntityConnector()
    ticker_to_wikidata = {}
    missing_tickers = {}
    for ticker, company in tqdm(ticker_to_company.items(), disable=not verbose, desc='Downloading Wikidata IDs'):

        wikidata_id = connector.find_company_wikidata_id(company)
        if wikidata_id:
            ticker_to_wikidata[ticker] = wikidata_id
        else:
            missing_tickers[ticker] = None

    if missing_tickers:
        ticker_to_wikidata.update(missing_tickers)
        if verbose:
            print(f'Missing Wikidata IDs for: {list(missing_tickers.keys())}')

    return ticker_to_wikidata


def download_connections(
        ticker_to_wikidata: dict[str, str],
        checkpoint_dir: Optional[str] = None,
        verbose: bool = False
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Downloads connection data between companies from Wikidata

    Args:
        ticker_to_wikidata: Dictionary mapping from ticker symbols to Wikidata IDs
        checkpoint_dir: Directory for persistent caching
        verbose: If True, print progress bar

    Returns:
        Dictionary with structure {ticker_a: {ticker_b: [connection_data]}}.

    Raises:
        NameError: If selected_paths is not defined in the module scope.
    """
    connections = {ticker: {} for ticker in ticker_to_wikidata.keys()}

    connector = WikidataEntityConnector(checkpoint_dir=checkpoint_dir)

    pairs = list(permutations(ticker_to_wikidata, 2))
    for source_ticker, target_ticker in tqdm(pairs, disable=not verbose, desc='Analyzing Wikidata relationships'):
        result = connector.analyze_wikidata_pair(ticker_to_wikidata[source_ticker], ticker_to_wikidata[target_ticker])

        # Extract valid connections from both first and second order
        if result['first_order_connections'] or result['second_order_connections']:
            connections[source_ticker][target_ticker] = []

            # Process first-order connections
            for conn in result['first_order_connections']:
                if conn['relation_id'] in SELECTED_PATHS:
                    temporal_info = min(conn['temporal_info'].values()) if conn['temporal_info'] else None
                    connections[source_ticker][target_ticker].append({
                        'relation_id': conn['relation_id'],
                        'temporal_info': temporal_info
                    })

            # Process second-order connections
            for conn in result['second_order_connections']:
                if conn['relation_id'] in SELECTED_PATHS:
                    temporal_info = min(conn['temporal_info_a_to_c'].values()) if conn['temporal_info_a_to_c'] else None
                    connections[source_ticker][target_ticker].append({
                        'relation_id': conn['relation_id'],
                        'temporal_info': temporal_info
                    })
            if not connections[source_ticker][target_ticker]:
                connections[source_ticker].pop(target_ticker)

    return connections


def _resample_timestamps(timestamps: list, freq: str = 'MS') -> list:
    """
    Resample string dates return as numpy array.

    Args:
        timestamps: List of timestamps
        freq: Resampling frequency

    Returns:
        List of resampled dates (timestamps)
    """

    ts_series = pd.Series(timestamps, index=timestamps)
    resampled = ts_series.resample(freq).first()
    resampled = resampled.dropna().tolist()
    return resampled


@validate_prices_decorator
def build_wikidata_adjacency_matrix(
        prices: pd.DataFrame,
        ticker_to_wikidata: dict[str, str],
        checkpoint_dir: str = None,
        freq: str = None,
        verbose: bool = False
) -> tuple[np.ndarray, list, list[str], list[str]]:
    """Builds an adjacency matrix for company connections from Wikidata.

    Args:
        prices: DataFrame with MultiIndex containing 'instrument' and 'date'
        ticker_to_wikidata: Dictionary mapping from ticker symbols to Wikidata IDs
        checkpoint_dir: Directory to save checkpoints to
        freq: Resampling frequency
        verbose (bool): If True, enable verbose output

    Returns:
        Tuple containing:
            - adj_matrix: 4D numpy array (dates, relations, tickers, tickers)
            - dates: Sorted array of unique dates
            - selected_paths: List of relation IDs
            - tickers: List of ticker symbols

    Raises:
        ValueError: If any ticker in price_data is not found in company_names.
    """

    # Reset MultiIndex to work with columns instead of index levels
    prices = prices.reset_index()

    # Convert date column to datetime type for proper temporal operations
    prices['date'] = pd.to_datetime(prices['date'])

    # Extract unique instruments and dates
    tickers = prices['instrument'].unique().tolist()
    dates = prices['date'].unique().tolist()

    # Calculate start and end dates for each ticker to determine data coverage periods
    date_stats = prices.groupby('instrument')['date'].agg(['min', 'max'])
    start_dates = date_stats['min'].to_dict()
    end_dates = date_stats['max'].to_dict()

    # Optionally resample dates to reduce temporal granularity and convert to sorted numpy array
    if freq:
        dates = _resample_timestamps(dates, freq)
    dates = np.sort(dates)

    # Validate that all tickers have corresponding wikidata id
    missing_wikidata = [t for t in tickers if t not in ticker_to_wikidata]
    if missing_wikidata:
        print(f'Warning: Missing wikidata id for tickers {missing_wikidata}')

    ticker_to_wikidata = {
        ticker: ticker_to_wikidata[ticker]
        for ticker in tickers
        if ticker in ticker_to_wikidata and ticker_to_wikidata[ticker]
    }
    connections = download_connections(ticker_to_wikidata, checkpoint_dir=checkpoint_dir, verbose=verbose)

    # Initialize adjacency matrix
    adj_matrix = np.zeros((len(dates), len(SELECTED_PATHS), len(tickers), len(tickers)), dtype=np.int8)

    # Create ticker-to-index mapping for efficient lookups
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    relation_to_idx = {rel_id: idx for idx, rel_id in enumerate(SELECTED_PATHS)}

    # Populate adjacency matrix with connection data
    for source_ticker, target_connections in tqdm(connections.items(), disable=not verbose, desc='Building adj matrix'):
        source_idx = ticker_to_idx[source_ticker]
        for target_ticker, connection_list in target_connections.items():
            target_idx = ticker_to_idx[target_ticker]
            overlap_end_date = min(end_dates[source_ticker], end_dates[target_ticker])
            overlap_start_date = max(start_dates[source_ticker], start_dates[target_ticker])
            for connection in connection_list:
                connection_start_date = (max(pd.Timestamp(connection['temporal_info']), overlap_start_date)
                                         if connection['temporal_info']
                                         else overlap_start_date)
                relation_idx = relation_to_idx[connection['relation_id']]

                # Set adjacency matrix values for the active date range
                active_dates_mask = ((dates >= connection_start_date) & (dates <= overlap_end_date))
                adj_matrix[active_dates_mask, relation_idx, source_idx, target_idx] = 1

    return adj_matrix, dates.tolist(), SELECTED_PATHS, tickers


def get_first_order_relation_indices(selected_paths: list[str]) -> list[int]:
    """ Find indices of first-order relations in a list of Wikidata relation paths """
    return [i for i, path in enumerate(selected_paths) if '_' not in path]


def get_first_order_relations(selected_paths: list[str]) -> list[str]:
    """ Extract first-order relations from a list of Wikidata relation paths """
    return [path for path in selected_paths if '_' not in path]
