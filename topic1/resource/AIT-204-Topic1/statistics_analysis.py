"""
Statistical Analysis Module for Synthetic Datasets
AIT-204 Topic 1: Background Math and Gradient-Based Learning

This module provides comprehensive descriptive statistical analysis
functions to help students understand data properties before modeling.
Understanding data statistics is crucial for:
- Feature scaling and normalization
- Identifying outliers
- Understanding data distribution
- Validating data generation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for regression datasets.

    Provides descriptive statistics, correlation analysis, distribution
    analysis, and other metrics essential for understanding data before
    applying gradient descent and other learning algorithms.
    """

    @staticmethod
    def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive descriptive statistics for all numerical columns.

        These statistics provide insights into:
        - Central tendency (mean, median)
        - Spread (std, variance, range)
        - Shape (skewness, kurtosis)
        - Extreme values (min, max, quartiles)

        Args:
            df: DataFrame containing the dataset

        Returns:
            DataFrame with statistical measures for each column
        """
        stats_dict = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col].dropna()  # Remove NaN values

            stats_dict[col] = {
                # Central Tendency Measures
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Mode': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else np.nan,

                # Spread Measures
                'Std Dev': np.std(data, ddof=1),  # Sample std dev
                'Variance': np.var(data, ddof=1),  # Sample variance
                'Range': np.ptp(data),  # Peak to peak (max - min)
                'IQR': np.percentile(data, 75) - np.percentile(data, 25),

                # Quartiles
                'Min': np.min(data),
                'Q1 (25%)': np.percentile(data, 25),
                'Q2 (50%)': np.percentile(data, 50),
                'Q3 (75%)': np.percentile(data, 75),
                'Max': np.max(data),

                # Shape Measures
                'Skewness': stats.skew(data),  # Asymmetry of distribution
                'Kurtosis': stats.kurtosis(data),  # Tailedness of distribution

                # Other
                'Count': len(data),
                'Missing': df[col].isna().sum()
            }

        return pd.DataFrame(stats_dict).T

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix between all numerical features.

        Correlation measures linear relationship strength between variables:
        - +1: Perfect positive linear correlation
        -  0: No linear correlation
        - -1: Perfect negative linear correlation

        This is crucial for understanding:
        - Feature redundancy (multicollinearity)
        - Relationship between features and target
        - Feature selection priorities

        Args:
            df: DataFrame containing the dataset

        Returns:
            Correlation matrix as DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        return df[numerical_cols].corr(method='pearson')

    @staticmethod
    def compute_covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute covariance matrix between all numerical features.

        Covariance measures how two variables change together:
        - Positive: Variables increase together
        - Negative: One increases as other decreases
        - Near zero: Variables are independent

        Important for understanding feature interactions and
        for algorithms that use covariance (e.g., PCA).

        Args:
            df: DataFrame containing the dataset

        Returns:
            Covariance matrix as DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        return df[numerical_cols].cov()

    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        IQR method defines outliers as values that fall outside:
        [Q1 - multiplier * IQR, Q3 + multiplier * multiplier]

        Common multipliers:
        - 1.5: Standard outlier detection (moderate)
        - 3.0: Extreme outlier detection (conservative)

        Args:
            df: DataFrame containing the dataset
            column: Name of column to analyze
            multiplier: IQR multiplier for outlier boundaries

        Returns:
            Dictionary with outlier information
        """
        data = df[column].dropna()

        # Calculate quartiles and IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Number of Outliers': len(outliers),
            'Percentage of Outliers': (len(outliers) / len(data)) * 100,
            'Outlier Values': outliers.tolist()
        }

    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using the Z-score method.

        Z-score measures how many standard deviations a point is from the mean:
        z = (x - μ) / σ

        Common thresholds:
        - 2.0: About 95% of data should fall within
        - 3.0: About 99.7% of data should fall within (default)

        Assumes approximately normal distribution.

        Args:
            df: DataFrame containing the dataset
            column: Name of column to analyze
            threshold: Z-score threshold for outlier detection

        Returns:
            Dictionary with outlier information
        """
        data = df[column].dropna()

        # Calculate z-scores
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        z_scores = np.abs((data - mean) / std)

        # Identify outliers
        outliers = data[z_scores > threshold]

        return {
            'Mean': mean,
            'Std Dev': std,
            'Threshold': threshold,
            'Number of Outliers': len(outliers),
            'Percentage of Outliers': (len(outliers) / len(data)) * 100,
            'Outlier Values': outliers.tolist(),
            'Max Z-Score': np.max(z_scores)
        }

    @staticmethod
    def normality_test(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Test if data follows a normal (Gaussian) distribution.

        Uses Shapiro-Wilk test:
        - H0 (null hypothesis): Data is normally distributed
        - H1 (alternative): Data is not normally distributed

        p-value interpretation:
        - p > 0.05: Cannot reject H0 (likely normal)
        - p ≤ 0.05: Reject H0 (likely not normal)

        Normal distribution is important because:
        - Many statistical methods assume normality
        - Gradient descent can behave differently with non-normal data
        - Helps choose appropriate preprocessing

        Args:
            df: DataFrame containing the dataset
            column: Name of column to test

        Returns:
            Dictionary with test results
        """
        data = df[column].dropna()

        # Perform Shapiro-Wilk test
        statistic, p_value = stats.shapiro(data)

        # Additional normality metrics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        return {
            'Test': 'Shapiro-Wilk',
            'Statistic': statistic,
            'P-value': p_value,
            'Is Normal (α=0.05)': p_value > 0.05,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
        }

    @staticmethod
    def compute_feature_target_stats(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
        """
        Compute statistics specifically for feature-target relationships.

        Analyzes how each feature relates to the target variable:
        - Correlation: Strength of linear relationship
        - Covariance: How they vary together
        - This helps identify most important features for regression

        Args:
            df: DataFrame containing the dataset
            target_col: Name of the target column

        Returns:
            DataFrame with feature-target statistics
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        stats_dict = {}
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in [target_col, 'y_true']]

        for col in feature_cols:
            # Compute correlation
            correlation = df[col].corr(df[target_col])

            # Compute covariance
            covariance = df[col].cov(df[target_col])

            # Compute R-squared (coefficient of determination)
            # R² represents proportion of variance in y explained by x
            r_squared = correlation ** 2

            stats_dict[col] = {
                'Correlation with Target': correlation,
                'Covariance with Target': covariance,
                'R² (Explained Variance)': r_squared,
                'Absolute Correlation': abs(correlation)
            }

        result_df = pd.DataFrame(stats_dict).T
        # Sort by absolute correlation (strongest relationships first)
        result_df = result_df.sort_values('Absolute Correlation', ascending=False)

        return result_df

    @staticmethod
    def compute_mse_from_true(df: pd.DataFrame, pred_col: str = 'y', true_col: str = 'y_true') -> float:
        """
        Compute Mean Squared Error between noisy and true values.

        MSE is the fundamental loss function for regression:
        MSE = (1/n) Σ(y_pred - y_true)²

        This shows the average squared deviation introduced by noise,
        representing the minimum achievable error given the noise level.

        Args:
            df: DataFrame containing the dataset
            pred_col: Column with noisy/predicted values
            true_col: Column with true (noiseless) values

        Returns:
            Mean Squared Error value
        """
        if true_col not in df.columns:
            return np.nan

        mse = np.mean((df[pred_col] - df[true_col]) ** 2)
        return mse

    @staticmethod
    def compute_summary_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of the dataset.

        Combines multiple statistical analyses into one convenient report.

        Args:
            df: DataFrame containing the dataset

        Returns:
            Dictionary containing summary statistics and analyses
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        report = {
            'Dataset Shape': df.shape,
            'Number of Features': len([col for col in numerical_cols if col not in ['y', 'y_true']]),
            'Number of Samples': len(df),
            'Total Missing Values': df.isna().sum().sum(),
            'Columns': df.columns.tolist(),
            'Numerical Columns': numerical_cols.tolist(),
        }

        # Add target variable statistics if present
        if 'y' in df.columns:
            report['Target (y) Statistics'] = {
                'Mean': df['y'].mean(),
                'Std Dev': df['y'].std(),
                'Min': df['y'].min(),
                'Max': df['y'].max(),
                'Range': df['y'].max() - df['y'].min()
            }

        # Add noise level if true values are available
        if 'y_true' in df.columns and 'y' in df.columns:
            mse = StatisticalAnalyzer.compute_mse_from_true(df)
            rmse = np.sqrt(mse)
            report['Noise Level'] = {
                'MSE': mse,
                'RMSE': rmse,
                'SNR (dB)': 10 * np.log10(df['y_true'].var() / mse) if mse > 0 else np.inf
            }

        return report
