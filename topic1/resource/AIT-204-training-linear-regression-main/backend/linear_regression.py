"""
Linear regression implementation from scratch.
"""

import numpy as np
from typing import Tuple, Dict


class LinearRegression:
    """Simple linear regression model (y = mx + b)."""

    def __init__(self):
        """Initialize the linear regression model."""
        self.slope = None
        self.intercept = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model using the closed-form solution.

        Uses the normal equation:
        slope = Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)
        intercept = y_mean - slope * x_mean

        Args:
            X: Feature values (1D array)
            y: Target values (1D array)
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate slope
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.slope = numerator / denominator

        # Calculate intercept
        self.intercept = y_mean - self.slope * x_mean

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Feature values

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X).flatten()
        return self.slope * X + self.intercept

    def calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for the model.

        Args:
            X: Feature values
            y: True target values

        Returns:
            Dictionary containing MSE, RMSE, MAE, and R²
        """
        y_pred = self.predict(X)
        y = np.array(y).flatten()

        # Mean Squared Error
        mse = np.mean((y - y_pred) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(y - y_pred))

        # R² (coefficient of determination)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def get_parameters(self) -> Dict[str, float]:
        """
        Get the model parameters.

        Returns:
            Dictionary with slope and intercept
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return {
            'slope': self.slope,
            'intercept': self.intercept
        }
