"""
Financial data generator for creating synthetic company financial datasets.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class FinancialDataGenerator:
    """Generates synthetic financial data for regression analysis."""

    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_revenue_data(
        self,
        n_samples: int = 100,
        base_revenue: float = 1000000,
        growth_rate: float = 50000,
        noise_std: float = 100000,
        outlier_percentage: float = 0.0,
        heteroscedastic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic revenue data based on marketing spend.

        The relationship is: Revenue = base + growth_rate * marketing_spend + noise

        Args:
            n_samples: Number of data points to generate
            base_revenue: Base revenue (intercept)
            growth_rate: Revenue increase per unit of marketing spend (slope)
            noise_std: Standard deviation of random noise
            outlier_percentage: Percentage of outliers to inject (0-100)
            heteroscedastic: If True, noise increases with marketing spend

        Returns:
            Tuple of (marketing_spend, revenue) as numpy arrays
        """
        # Generate marketing spend (in thousands of dollars)
        marketing_spend = np.random.uniform(10, 200, n_samples)

        # Generate base noise
        if heteroscedastic:
            # Noise scales with marketing spend (more realistic)
            noise = np.random.normal(0, 1, n_samples) * noise_std * (marketing_spend / marketing_spend.mean())
        else:
            noise = np.random.normal(0, noise_std, n_samples)

        # Generate revenue with linear relationship + noise
        revenue = base_revenue + growth_rate * marketing_spend + noise

        # Add outliers
        if outlier_percentage > 0:
            n_outliers = int(n_samples * outlier_percentage / 100)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

            for idx in outlier_indices:
                # Create outliers that deviate significantly from the trend
                outlier_type = np.random.choice(['high', 'low', 'extreme'])

                if outlier_type == 'high':
                    # Revenue much higher than expected
                    revenue[idx] += np.random.uniform(2, 4) * noise_std
                elif outlier_type == 'low':
                    # Revenue much lower than expected
                    revenue[idx] -= np.random.uniform(2, 4) * noise_std
                else:  # extreme
                    # Very extreme outlier
                    revenue[idx] += np.random.choice([-1, 1]) * np.random.uniform(4, 6) * noise_std

        # Ensure revenue is positive
        revenue = np.maximum(revenue, 100000)

        return marketing_spend, revenue

    def generate_profit_data(
        self,
        n_samples: int = 100,
        base_profit: float = 200000,
        profit_rate: float = 30000,
        noise_std: float = 80000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic profit data based on sales volume.

        Args:
            n_samples: Number of data points to generate
            base_profit: Base profit (intercept)
            profit_rate: Profit per unit of sales (slope)
            noise_std: Standard deviation of random noise

        Returns:
            Tuple of (sales_volume, profit) as numpy arrays
        """
        # Generate sales volume
        sales_volume = np.random.uniform(5, 150, n_samples)

        # Generate profit with linear relationship + noise
        profit = base_profit + profit_rate * sales_volume + \
                 np.random.normal(0, noise_std, n_samples)

        return sales_volume, profit

    def create_dataframe(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_name: str = "Feature",
        y_name: str = "Target"
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from feature and target arrays.

        Args:
            x: Feature values
            y: Target values
            x_name: Name for the feature column
            y_name: Name for the target column

        Returns:
            DataFrame with the data
        """
        return pd.DataFrame({
            x_name: x,
            y_name: y
        })
