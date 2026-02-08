import itertools
from collections import defaultdict

import numpy as np
import pandas as pd


def build_sector_industry_adjacency_matrix(
        tickers: list[str],
        ticker_to_sector: dict[str, str],
        ticker_to_industry: dict[str, str],
        verbose: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Constructs adjacency matrices based on sector and industry classifications

    It generates a 3D numpy array where each slice along the first
    axis is a binary adjacency matrix. Each matrix corresponds to a specific
    sector or industry category. Within each category's matrix, an edge (1) is
    placed between every pair of tickers that belong to that category,
    forming a fully connected subgraph (a clique)

    Args:
        tickers (list[str]): A list of ticker symbols defining the universe.
                             The order of tickers determines the axes of the
                             output matrices
        ticker_to_sector (dict[str, str]): A mapping from ticker to its sector
        ticker_to_industry (dict[str, str]): A mapping from ticker to its industry
        verbose (bool, optional): If True, prints any unclassified tickers. Defaults to False

    Returns:
        tuple[np.ndarray, list[str], list[str]]: A tuple containing:
        - adj_matrix: A 3D array of shape (C, N, N), where C is the number
          of unique categories (sectors + industries) and N is the number of
          tickers. `adj_matrix[i, :, :]` is the adjacency matrix for the
          i-th category. It includes self-loops (diagonals are 1)
        - categories: A list of all unique sector and industry names
          corresponding to the first axis of `adj_matrix`
        - tickers: The original list of tickers passed to the function

    Raises:
        ValueError: If any value in the input mapping dictionaries is nan
    """

    # Check for None or NaN using pandas.
    if pd.isnull(list(ticker_to_sector.values()) + list(ticker_to_industry.values())).any():
        raise ValueError("Invalid category names found (contains NaN or None)")

    missing_sectors = [ticker for ticker in tickers if ticker not in ticker_to_sector]
    if verbose and missing_sectors:
        print(f"Found {len(missing_sectors)} ticker(s) missing a Sector classification:")
        print(f"  - {', '.join(sorted(missing_sectors))}")

    missing_industries = [ticker for ticker in tickers if ticker not in ticker_to_industry]
    if verbose and missing_industries:
        print(f"Found {len(missing_industries)} ticker(s) missing an Industry classification:")
        print(f"  - {', '.join(sorted(missing_industries))}")

    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}

    # Group tickers by category
    groups = defaultdict(list)
    for ticker, category in itertools.chain(ticker_to_sector.items(), ticker_to_industry.items()):
        if ticker in ticker_to_idx:
            groups[category].append(ticker)

    categories = list(groups.keys())

    # Initialize the 3D matrix and populate for each category
    adj_matrix = np.zeros((len(categories), len(tickers), len(tickers)), dtype=np.int8)
    for i, category in enumerate(categories):
        # Get the integer indices for all tickers in the current category.
        indices = [ticker_to_idx[ticker] for ticker in groups[category]]

        # If the category has members, create the fully connected component.
        if indices:
            mesh_indices = np.ix_(indices, indices)
            adj_matrix[i][mesh_indices] = 1

    return adj_matrix, categories, tickers
