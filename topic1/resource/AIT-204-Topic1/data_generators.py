"""
Synthetic Dataset Generators for Regression Tasks
AIT-204 Topic 1: Background Math and Gradient-Based Learning

This module provides various synthetic dataset generation functions
suitable for regression model training and gradient descent learning.
Each function creates data with known mathematical relationships,
allowing students to understand how models learn from data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SyntheticDataGenerator:
    """
    A comprehensive class for generating various types of synthetic datasets
    suitable for regression tasks in deep learning courses.

    This generator is designed to help students understand:
    - Linear and non-linear relationships
    - Effect of noise on data
    - Feature scaling and normalization
    - Multi-dimensional input spaces
    - Polynomial features and complexity
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the data generator with optional random seed for reproducibility.

        Args:
            random_seed: Integer seed for numpy random number generator.
                        If None, results will vary each time.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed

    def generate_simple_linear(self,
                               n_samples: int = 100,
                               slope: float = 2.0,
                               intercept: float = 1.0,
                               noise_std: float = 0.5,
                               x_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame:
        """
        Generate simple linear regression data: y = mx + b + ε

        This is the fundamental regression problem where the relationship
        between input (x) and output (y) is linear. Perfect for understanding
        gradient descent optimization.

        Args:
            n_samples: Number of data points to generate
            slope: Slope (m) of the linear relationship
            intercept: Y-intercept (b) of the line
            noise_std: Standard deviation of Gaussian noise (ε)
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true'] where y_true is noiseless output
        """
        # Generate uniformly distributed x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate true y values (without noise)
        y_true = slope * x + intercept

        # Add Gaussian noise to create realistic data
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        # Create DataFrame for easy manipulation and export
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_multiple_linear(self,
                                 n_samples: int = 100,
                                 n_features: int = 3,
                                 coefficients: Optional[np.ndarray] = None,
                                 intercept: float = 5.0,
                                 noise_std: float = 1.0,
                                 feature_range: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
        """
        Generate multiple linear regression data: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b + ε

        This represents multi-dimensional input space, crucial for understanding
        gradient descent in higher dimensions and partial derivatives.

        Args:
            n_samples: Number of data points to generate
            n_features: Number of input features
            coefficients: Array of weights for each feature. If None, random weights used
            intercept: Bias term (b)
            noise_std: Standard deviation of Gaussian noise
            feature_range: Tuple of (min, max) for feature values

        Returns:
            DataFrame with columns ['x1', 'x2', ..., 'xn', 'y', 'y_true']
        """
        # Generate random coefficients if not provided
        if coefficients is None:
            coefficients = np.random.uniform(-3, 3, n_features)
        else:
            assert len(coefficients) == n_features, "Coefficients must match n_features"

        # Generate feature matrix (n_samples × n_features)
        X = np.random.uniform(feature_range[0], feature_range[1], (n_samples, n_features))

        # Calculate true output: matrix multiplication + intercept
        y_true = X @ coefficients + intercept

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        # Create DataFrame with named columns
        df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
        df['y'] = y
        df['y_true'] = y_true

        return df

    def generate_polynomial(self,
                           n_samples: int = 100,
                           degree: int = 2,
                           coefficients: Optional[np.ndarray] = None,
                           noise_std: float = 2.0,
                           x_range: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
        """
        Generate polynomial regression data: y = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀ + ε

        Polynomial relationships demonstrate non-linear patterns that still have
        analytical solutions. Important for understanding feature engineering
        and basis function expansion.

        Args:
            n_samples: Number of data points to generate
            degree: Degree of the polynomial (2 = quadratic, 3 = cubic, etc.)
            coefficients: Array of polynomial coefficients [a₀, a₁, a₂, ..., aₙ]
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Generate random coefficients if not provided
        if coefficients is None:
            coefficients = np.random.uniform(-2, 2, degree + 1)
        else:
            assert len(coefficients) == degree + 1, f"Need {degree + 1} coefficients"

        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate polynomial: y = Σ(aᵢ * xⁱ)
        y_true = np.zeros(n_samples)
        for i, coef in enumerate(coefficients):
            y_true += coef * (x ** i)

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_sinusoidal(self,
                           n_samples: int = 100,
                           amplitude: float = 5.0,
                           frequency: float = 1.0,
                           phase: float = 0.0,
                           offset: float = 0.0,
                           noise_std: float = 0.5,
                           x_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame:
        """
        Generate sinusoidal regression data: y = A·sin(ωx + φ) + c + ε

        Sinusoidal patterns are common in time series and periodic phenomena.
        They represent smooth non-linear relationships useful for testing
        neural network approximation capabilities.

        Args:
            n_samples: Number of data points to generate
            amplitude: Amplitude (A) of the sine wave
            frequency: Frequency (ω) - controls oscillation rate
            phase: Phase shift (φ) in radians
            offset: Vertical offset (c)
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate sinusoidal function
        y_true = amplitude * np.sin(frequency * x + phase) + offset

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_exponential(self,
                            n_samples: int = 100,
                            base: float = np.e,
                            scale: float = 1.0,
                            offset: float = 0.0,
                            noise_std: float = 0.5,
                            x_range: Tuple[float, float] = (0, 3)) -> pd.DataFrame:
        """
        Generate exponential regression data: y = scale·baseˣ + offset + ε

        Exponential growth/decay is fundamental in many domains (population,
        radioactive decay, compound interest). Tests model's ability to handle
        rapidly changing gradients.

        Args:
            n_samples: Number of data points to generate
            base: Base of the exponential (e for natural exponential)
            scale: Scaling factor
            offset: Vertical offset
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate exponential function
        y_true = scale * (base ** x) + offset

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_logarithmic(self,
                            n_samples: int = 100,
                            scale: float = 5.0,
                            offset: float = 1.0,
                            noise_std: float = 0.5,
                            x_range: Tuple[float, float] = (0.1, 10)) -> pd.DataFrame:
        """
        Generate logarithmic regression data: y = scale·log(x + offset) + ε

        Logarithmic relationships appear in information theory, chemistry (pH),
        and many natural phenomena. Important for understanding non-linear
        transformations and diminishing returns.

        Args:
            n_samples: Number of data points to generate
            scale: Scaling factor for the log
            offset: Shift to avoid log(0), also vertical offset
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values (must be > 0)

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Ensure x_range is positive for logarithm
        x_range = (max(x_range[0], 0.001), x_range[1])

        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Calculate logarithmic function
        y_true = scale * np.log(x + offset)

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_step_function(self,
                              n_samples: int = 100,
                              n_steps: int = 3,
                              step_height: float = 2.0,
                              noise_std: float = 0.3,
                              x_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame:
        """
        Generate step function data (piecewise constant)

        Step functions represent discrete changes and are challenging for
        gradient-based methods. Useful for understanding model limitations
        and the importance of activation functions.

        Args:
            n_samples: Number of data points to generate
            n_steps: Number of discrete steps
            step_height: Height difference between consecutive steps
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        # Create step boundaries
        step_width = (x_range[1] - x_range[0]) / n_steps

        # Calculate which step each x falls into
        y_true = np.floor((x - x_range[0]) / step_width) * step_height

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_interaction_features(self,
                                     n_samples: int = 100,
                                     w1: float = 2.0,
                                     w2: float = 3.0,
                                     w_interaction: float = 1.5,
                                     intercept: float = 1.0,
                                     noise_std: float = 1.0,
                                     feature_range: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
        """
        Generate data with interaction terms: y = w₁x₁ + w₂x₂ + w₁₂(x₁·x₂) + b + ε

        Interaction terms show how features combine multiplicatively.
        Important for understanding feature engineering and non-linear
        combinations of linear inputs.

        Args:
            n_samples: Number of data points to generate
            w1: Weight for first feature
            w2: Weight for second feature
            w_interaction: Weight for interaction term (x1 * x2)
            intercept: Bias term
            noise_std: Standard deviation of Gaussian noise
            feature_range: Tuple of (min, max) for feature values

        Returns:
            DataFrame with columns ['x1', 'x2', 'x1_x2', 'y', 'y_true']
        """
        # Generate two features
        x1 = np.random.uniform(feature_range[0], feature_range[1], n_samples)
        x2 = np.random.uniform(feature_range[0], feature_range[1], n_samples)

        # Calculate interaction term
        x1_x2 = x1 * x2

        # Calculate output with interaction
        y_true = w1 * x1 + w2 * x2 + w_interaction * x1_x2 + intercept

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x1_x2': x1_x2,
            'y': y,
            'y_true': y_true
        })

        return df

    def generate_custom_function(self,
                                n_samples: int = 100,
                                func_str: str = "x**2 + 2*x + 1",
                                noise_std: float = 1.0,
                                x_range: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
        """
        Generate data from a custom mathematical function defined by the user.

        Allows experimentation with arbitrary functions for advanced exploration.
        Use with caution - evaluates Python expressions.

        Args:
            n_samples: Number of data points to generate
            func_str: String representation of function (use 'x' as variable)
                     Example: "x**2 + 2*x + 1" or "np.sin(x) + np.cos(2*x)"
            noise_std: Standard deviation of Gaussian noise
            x_range: Tuple of (min, max) for x values

        Returns:
            DataFrame with columns ['x', 'y', 'y_true']
        """
        # Generate x values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)

        try:
            # Evaluate the function string (be careful - this uses eval!)
            # Create safe namespace with numpy functions
            namespace = {'x': x, 'np': np, 'sin': np.sin, 'cos': np.cos,
                        'tan': np.tan, 'exp': np.exp, 'log': np.log,
                        'sqrt': np.sqrt, 'abs': np.abs}
            y_true = eval(func_str, {"__builtins__": {}}, namespace)
        except Exception as e:
            raise ValueError(f"Error evaluating function '{func_str}': {str(e)}")

        # Add noise
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'y_true': y_true
        })

        return df
