import numpy as np
import pandas as pd


def create_incidence_from_wikidata(
        relation_matrix: np.ndarray,
        first_order_relation_indices: list[int] = None
) -> np.ndarray:
    """Builds a 2D incidence matrix from a 3D Wikidata relation tensor,
    correctly interpreting the split between first-order and second-order relations.

    Inspired by STHAN-SR "Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach"

    Args:
        relation_matrix (str): 3D NumPy array of shape (num_relations, num_stocks, num_stocks).
        first_order_relation_indices (int): Index of the first order relations

    Returns:
        numpy.ndarray: The final 2D incidence matrix H of shape (num_stocks, num_unique_hyperedges).
    """
    if relation_matrix.ndim != 3:
        raise ValueError('relation_matrix must be a 3D NumPy array of shape (num_relations, num_stocks, num_stocks)')

    num_relations, num_stocks, _ = relation_matrix.shape
    second_order_relation_indices = [i for i in range(num_relations) if i not in first_order_relation_indices]
    hyperedges = []

    # Generate First-Order Hyperedges
    # "a source stock and a set of target stocks related through the same relation"
    for source_stock_idx in range(num_stocks):
        # Loop ONLY over the first-order relation types
        for rel_idx in first_order_relation_indices:
            # Find all target stocks for the current source and first-order relation
            target_indices = np.where(relation_matrix[rel_idx, source_stock_idx, :] == 1)[0].tolist()

            if len(target_indices) > 0:
                # Create the hyperedge: {source} U {targets}
                current_hyperedge = {source_stock_idx, *target_indices}
                hyperedges.append(current_hyperedge)

    # Generate Second-Order Hyperedges
    # "a hyperedge ... between two stocks in a second-order relationship (pairwise)"
    # Loop ONLY over the second-order relation types
    # Iterate through the upper triangle of the matrix to avoid duplicates (i,j) vs (j,i)
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            # Check if these two stocks are related by any SECOND-ORDER relation type
            if (np.any(relation_matrix[second_order_relation_indices, i, j] == 1) or
                    np.any(relation_matrix[second_order_relation_indices, j, i] == 1)):
                # Create the pairwise hyperedge {i, j}
                hyperedges.append({i, j})

    # Combine and Deduplicate
    # Use frozenset as it is hashable and can be put in a set
    unique_hyperedges_frozensets = {frozenset(h) for h in hyperedges}
    unique_hyperedges = [set(fs) for fs in unique_hyperedges_frozensets]

    if len(unique_hyperedges) == 0:
        # Warning: No unique hyperedges were created. Check the input data
        return np.zeros((num_stocks, 0), dtype=np.int8)

    # Initialize the incidence matrix H with zeros
    incidence_matrix = np.zeros((num_stocks, len(unique_hyperedges)), dtype=np.int8)

    # Populate the matrix
    for e, hyperedge in enumerate(unique_hyperedges):
        for v in hyperedge:
            if v < num_stocks:  # Safety check
                incidence_matrix[v, e] = 1

    return incidence_matrix


def create_incidence_from_sector_industry(adj_matrix: np.ndarray) -> np.ndarray:
    """Builds a 2D incidence array (tickers x categories) from a 3D adjacency matrix

    This function transforms the (C, N, N) adjacency matrix into a (N, C)
    binary incidence array using only NumPy operations

    Args:
        adj_matrix (np.ndarray): A 3D numpy array of shape (C, N, N), where
                                 C is the number of categories and N is the
                                 number of tickers

    Returns:
        np.ndarray: A 2D NumPy array of shape (N, C) where rows correspond
                    to tickers and columns correspond to categories, indicating
                    group membership with a '1'
    """
    # Extract the diagonals from each (N, N) slice along axis 0
    # The result has a shape of (C, N), where C is categories and N is tickers
    incidence_cn = np.diagonal(adj_matrix, axis1=1, axis2=2)

    # Transpose the result
    incidence_nc = incidence_cn.T

    return incidence_nc


def align_and_concat_matrices(
        matrix1: np.ndarray,
        tickers1: list[str],
        categories1: list[str],
        matrix2: np.ndarray,
        tickers2: list[str],
        categories2: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Aligns and horizontally concatenates two incidence matrices, returning a
    NumPy array and corresponding labels.

    This function uses pandas for its powerful index alignment capabilities but
    returns standard Python/NumPy types.

    Args:
        matrix1 (np.ndarray): The first incidence matrix
        tickers1 (list[str]): The list of tickers for the rows of matrix1
        categories1 (list[str]): The list of categories for the columns of matrix1
        matrix2 (np.ndarray): The second incidence matrix
        tickers2 (list[str]): The list of tickers for the rows of matrix2
        categories2 (list[str]): The list of categories for the columns of matrix2

    Returns:
        Tuple[np.ndarray, list[str], list[str]]: A tuple containing:
        - final_matrix (np.ndarray): The new, larger concatenated NumPy array
        - final_tickers (list[str]): The aligned list of tickers for the rows
        - final_categories (list[str]): The combined list of categories for the columns
    """
    # Concatenate horizontally by using use index alignment
    df1 = pd.DataFrame(data=matrix1, index=tickers1, columns=categories1)
    df2 = pd.DataFrame(data=matrix2, index=tickers2, columns=categories2)
    combined_df = pd.concat([df1, df2], axis=1)

    # Clean up the result by filling missing values and ensuring integer type
    combined_df = combined_df.fillna(0).astype(np.int8)

    final_matrix = combined_df.values
    final_tickers = combined_df.index.tolist()
    final_categories = combined_df.columns.tolist()

    return final_matrix, final_tickers, final_categories
