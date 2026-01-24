"""
Data utilities for train/test splitting and preprocessing.
"""

import numpy as np
from typing import Tuple, Dict


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Feature values
        y: Target values
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = np.array(X).flatten()
    y = np.array(y).flatten()

    # Set random seed
    np.random.seed(random_state)

    # Get number of samples
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Create random indices
    indices = np.random.permutation(n_samples)

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def get_split_info(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, any]:
    """
    Get information about the train/test split.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets

    Returns:
        Dictionary with split statistics
    """
    total_samples = len(X_train) + len(X_test)
    train_percentage = (len(X_train) / total_samples) * 100
    test_percentage = (len(X_test) / total_samples) * 100

    return {
        'total_samples': total_samples,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_percentage': train_percentage,
        'test_percentage': test_percentage,
        'train_X_range': (X_train.min(), X_train.max()),
        'test_X_range': (X_test.min(), X_test.max()),
        'train_y_range': (y_train.min(), y_train.max()),
        'test_y_range': (y_test.min(), y_test.max())
    }
