"""
Gradient descent optimizer for linear regression.
"""

import numpy as np
from typing import Tuple, List, Dict


class GradientDescentOptimizer:
    """Implements gradient descent optimization for linear regression."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        """
        Initialize the gradient descent optimizer.

        Args:
            learning_rate: Step size for parameter updates
            max_iterations: Maximum number of iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = []
        self.slope = None
        self.intercept = None

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """
        Compute the Mean Squared Error cost function.

        Cost = (1/2n) * Σ(y_pred - y)²
        where y_pred = slope * X + intercept

        Args:
            X: Feature values
            y: Target values
            slope: Current slope parameter
            intercept: Current intercept parameter

        Returns:
            Cost value
        """
        n = len(y)
        predictions = slope * X + intercept
        cost = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
        return cost

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slope: float,
        intercept: float
    ) -> Tuple[float, float]:
        """
        Compute gradients of the cost function with respect to parameters.

        Gradient formulas:
        ∂Cost/∂slope = (1/n) * Σ(y_pred - y) * X
        ∂Cost/∂intercept = (1/n) * Σ(y_pred - y)

        Args:
            X: Feature values
            y: Target values
            slope: Current slope parameter
            intercept: Current intercept parameter

        Returns:
            Tuple of (gradient_slope, gradient_intercept)
        """
        n = len(y)
        predictions = slope * X + intercept
        errors = predictions - y

        gradient_slope = (1 / n) * np.sum(errors * X)
        gradient_intercept = (1 / n) * np.sum(errors)

        return gradient_slope, gradient_intercept

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_slope: float = 0.0,
        initial_intercept: float = 0.0
    ) -> Dict[str, List]:
        """
        Fit the model using gradient descent.

        The update rules are:
        slope = slope - learning_rate * ∂Cost/∂slope
        intercept = intercept - learning_rate * ∂Cost/∂intercept

        Args:
            X: Feature values
            y: Target values
            initial_slope: Starting value for slope
            initial_intercept: Starting value for intercept

        Returns:
            Dictionary containing training history
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        # Initialize parameters
        self.slope = initial_slope
        self.intercept = initial_intercept

        # Store history for visualization
        self.history = {
            'iteration': [],
            'cost': [],
            'slope': [],
            'intercept': [],
            'gradient_slope': [],
            'gradient_intercept': []
        }

        # Gradient descent iterations
        for i in range(self.max_iterations):
            # Compute current cost
            cost = self._compute_cost(X, y, self.slope, self.intercept)

            # Compute gradients
            grad_slope, grad_intercept = self._compute_gradients(
                X, y, self.slope, self.intercept
            )

            # Store history
            self.history['iteration'].append(i)
            self.history['cost'].append(cost)
            self.history['slope'].append(self.slope)
            self.history['intercept'].append(self.intercept)
            self.history['gradient_slope'].append(grad_slope)
            self.history['gradient_intercept'].append(grad_intercept)

            # Update parameters
            self.slope = self.slope - self.learning_rate * grad_slope
            self.intercept = self.intercept - self.learning_rate * grad_intercept

        # Store final iteration
        final_cost = self._compute_cost(X, y, self.slope, self.intercept)
        self.history['iteration'].append(self.max_iterations)
        self.history['cost'].append(final_cost)
        self.history['slope'].append(self.slope)
        self.history['intercept'].append(self.intercept)
        self.history['gradient_slope'].append(0.0)
        self.history['gradient_intercept'].append(0.0)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Feature values

        Returns:
            Predicted values
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X).flatten()
        return self.slope * X + self.intercept

    def get_parameters(self) -> Dict[str, float]:
        """
        Get the final model parameters.

        Returns:
            Dictionary with slope and intercept
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be fitted first")

        return {
            'slope': self.slope,
            'intercept': self.intercept
        }

    def get_history(self) -> Dict[str, List]:
        """
        Get the training history.

        Returns:
            Dictionary containing training history
        """
        return self.history
