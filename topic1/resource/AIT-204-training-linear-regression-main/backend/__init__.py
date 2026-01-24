"""
Backend module for financial regression analysis.
"""

from .data_generator import FinancialDataGenerator
from .linear_regression import LinearRegression
from .gradient_descent import GradientDescentOptimizer
from .data_utils import train_test_split, get_split_info

__all__ = [
    'FinancialDataGenerator',
    'LinearRegression',
    'GradientDescentOptimizer',
    'train_test_split',
    'get_split_info'
]
