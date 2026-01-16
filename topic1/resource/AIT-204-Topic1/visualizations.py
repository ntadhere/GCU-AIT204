"""
Visualization Module for Synthetic Datasets
AIT-204 Topic 1: Background Math and Gradient-Based Learning

This module provides comprehensive visualization functions for
regression datasets. Visualizations help students:
- Understand data patterns visually
- Identify relationships between features and target
- Detect outliers and anomalies
- Assess data quality and distribution
- Validate generated synthetic data
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List


class DataVisualizer:
    """
    Comprehensive visualization toolkit for regression datasets.

    Uses Plotly for interactive visualizations that allow:
    - Zooming and panning
    - Hover information
    - Dynamic exploration
    - Export to static images
    """

    @staticmethod
    def plot_scatter_1d(df: pd.DataFrame,
                       x_col: str = 'x',
                       y_col: str = 'y',
                       show_true: bool = True,
                       title: str = "Scatter Plot") -> go.Figure:
        """
        Create interactive scatter plot for single-feature regression.

        Shows the relationship between one input feature (x) and output (y).
        Optionally displays the true underlying function without noise.

        Args:
            df: DataFrame containing the data
            x_col: Name of the x-axis column
            y_col: Name of the y-axis column
            show_true: Whether to show the true function (if available)
            title: Plot title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add scatter plot of noisy data
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Observed Data',
            marker=dict(
                size=8,
                color='rgba(31, 119, 180, 0.6)',
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>'
        ))

        # Add true function if available
        if show_true and 'y_true' in df.columns:
            # Sort by x for proper line plotting
            df_sorted = df.sort_values(x_col)

            fig.add_trace(go.Scatter(
                x=df_sorted[x_col],
                y=df_sorted['y_true'],
                mode='lines',
                name='True Function',
                line=dict(color='red', width=3),
                hovertemplate=f'{x_col}: %{{x:.3f}}<br>True y: %{{y:.3f}}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest',
            template='plotly_white',
            showlegend=True,
            height=500
        )

        return fig

    @staticmethod
    def plot_residuals(df: pd.DataFrame,
                      pred_col: str = 'y',
                      true_col: str = 'y_true',
                      feature_col: Optional[str] = None) -> go.Figure:
        """
        Create residual plot to visualize prediction errors.

        Residuals = Observed - True
        Residual plots help identify:
        - Systematic errors (patterns in residuals)
        - Heteroscedasticity (changing variance)
        - Outliers (extreme residuals)

        Ideal residual plot: random scatter around zero

        Args:
            df: DataFrame containing the data
            pred_col: Column with predictions/noisy values
            true_col: Column with true values
            feature_col: Optional feature to plot against (x-axis)

        Returns:
            Plotly Figure object
        """
        if true_col not in df.columns:
            raise ValueError(f"Column '{true_col}' not found. Cannot compute residuals.")

        # Calculate residuals
        residuals = df[pred_col] - df[true_col]

        # Determine x-axis
        if feature_col is None:
            # Use predicted values on x-axis
            x_data = df[pred_col]
            x_label = pred_col
        else:
            x_data = df[feature_col]
            x_label = feature_col

        fig = go.Figure()

        # Add residual scatter plot
        fig.add_trace(go.Scatter(
            x=x_data,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                size=8,
                color='rgba(31, 119, 180, 0.6)',
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'{x_label}: %{{x:.3f}}<br>Residual: %{{y:.3f}}<extra></extra>'
        ))

        # Add horizontal line at y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Zero Error",
            annotation_position="right"
        )

        fig.update_layout(
            title="Residual Plot (Error Analysis)",
            xaxis_title=x_label,
            yaxis_title="Residuals (Observed - True)",
            hovermode='closest',
            template='plotly_white',
            height=500
        )

        return fig

    @staticmethod
    def plot_distribution(df: pd.DataFrame,
                         column: str,
                         bins: int = 30) -> go.Figure:
        """
        Create histogram to visualize data distribution.

        Histograms show:
        - Central tendency (where data clusters)
        - Spread (how dispersed the data is)
        - Skewness (asymmetry)
        - Modality (number of peaks)

        Important for understanding if data follows expected distributions
        (e.g., Gaussian for noise).

        Args:
            df: DataFrame containing the data
            column: Column name to visualize
            bins: Number of histogram bins

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=df[column],
            nbinsx=bins,
            name='Distribution',
            marker=dict(
                color='rgba(31, 119, 180, 0.7)',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))

        # Add mean line
        mean_val = df[column].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top"
        )

        # Add median line
        median_val = df[column].median()
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="bottom"
        )

        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            template='plotly_white',
            showlegend=False,
            height=500
        )

        return fig

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame,
                                exclude_cols: Optional[List[str]] = None) -> go.Figure:
        """
        Create correlation heatmap for all numerical features.

        Correlation heatmap visualizes:
        - Strength of linear relationships (color intensity)
        - Positive vs negative correlations (color hue)
        - Multicollinearity (high correlation between features)

        Useful for:
        - Feature selection
        - Identifying redundant features
        - Understanding feature interactions

        Args:
            df: DataFrame containing the data
            exclude_cols: Optional list of columns to exclude

        Returns:
            Plotly Figure object
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude specified columns
        if exclude_cols:
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr()

        # Create heatmap
        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Correlation Heatmap",
            template='plotly_white',
            height=600,
            width=700
        )

        return fig

    @staticmethod
    def plot_pairwise_scatter(df: pd.DataFrame,
                            max_features: int = 5) -> go.Figure:
        """
        Create pairwise scatter plot matrix (SPLOM).

        Shows all pairwise relationships between features in one view.
        Useful for:
        - Identifying non-linear relationships
        - Detecting clusters or patterns
        - Understanding high-dimensional data structure

        Args:
            df: DataFrame containing the data
            max_features: Maximum number of features to include (for performance)

        Returns:
            Plotly Figure object
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit number of features for readability
        if len(numerical_cols) > max_features:
            # Prioritize features that are not 'y_true'
            priority_cols = [col for col in numerical_cols if col != 'y_true'][:max_features]
            numerical_cols = priority_cols
        # Create scatter matrix using Plotly Express
        fig = px.scatter_matrix(
            df[numerical_cols],
            dimensions=numerical_cols,
            title="Pairwise Feature Relationships"
        )

        fig.update_traces(
            diagonal_visible=False,
            marker=dict(size=5, opacity=0.6)
        )

        fig.update_layout(
            height=800,
            width=900,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_box_whisker(df: pd.DataFrame,
                        columns: Optional[List[str]] = None) -> go.Figure:
        """
        Create box-and-whisker plots for distribution analysis.

        Box plots show:
        - Median (center line)
        - Quartiles (box boundaries)
        - Range (whiskers)
        - Outliers (individual points)

        Useful for:
        - Comparing distributions across features
        - Identifying outliers
        - Understanding data spread

        Args:
            df: DataFrame containing the data
            columns: List of columns to plot (if None, uses all numerical)

        Returns:
            Plotly Figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                boxmean='sd',  # Show mean and standard deviation
                hovertemplate=f'{col}<br>Value: %{{y:.3f}}<extra></extra>'
            ))

        fig.update_layout(
            title="Box-and-Whisker Plots",
            yaxis_title="Value",
            template='plotly_white',
            showlegend=True,
            height=500
        )

        return fig

    @staticmethod
    def plot_3d_scatter(df: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       z_col: str,
                       color_col: Optional[str] = None) -> go.Figure:
        """
        Create 3D scatter plot for visualizing three dimensions.

        Useful for datasets with 2+ features to visualize the relationship
        between two features and the target in 3D space.

        Args:
            df: DataFrame containing the data
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for z-axis (usually target)
            color_col: Optional column for color mapping

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Determine color
        if color_col and color_col in df.columns:
            color_data = df[color_col]
            colorbar_title = color_col
        else:
            color_data = df[z_col]
            colorbar_title = z_col

        fig.add_trace(go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=color_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=f'{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<br>{z_col}: %{{z:.3f}}<extra></extra>'
        ))

        fig.update_layout(
            title=f"3D Scatter: {z_col} vs ({x_col}, {y_col})",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            template='plotly_white',
            height=700
        )

        return fig

    @staticmethod
    def plot_qq(df: pd.DataFrame, column: str) -> go.Figure:
        """
        Create Q-Q (Quantile-Quantile) plot to assess normality.

        Q-Q plot compares data distribution to theoretical normal distribution:
        - Points on diagonal line: Data is normally distributed
        - Points deviate from line: Data is not normal

        Useful for:
        - Testing normality assumption
        - Identifying distribution type
        - Detecting heavy tails or skewness

        Args:
            df: DataFrame containing the data
            column: Column to analyze

        Returns:
            Plotly Figure object
        """
        data = df[column].dropna().values

        # Calculate theoretical quantiles (normal distribution)
        from scipy.stats import probplot
        theoretical_quantiles, ordered_values = probplot(data, dist="norm")

        fig = go.Figure()

        # Add Q-Q scatter points
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles[0],
            y=theoretical_quantiles[1],
            mode='markers',
            name='Data',
            marker=dict(size=6, color='rgba(31, 119, 180, 0.6)'),
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>'
        ))

        # Add reference line (perfect normal distribution)
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles[0],
            y=theoretical_quantiles[0] * np.std(data) + np.mean(data),
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"Q-Q Plot: {column}",
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Sample Quantiles",
            template='plotly_white',
            showlegend=True,
            height=500
        )

        return fig

    @staticmethod
    def plot_feature_importance_correlation(df: pd.DataFrame,
                                          target_col: str = 'y') -> go.Figure:
        """
        Create bar chart showing correlation of each feature with target.

        Visualizes which features have strongest linear relationship with target.
        Helps prioritize features for modeling.

        Args:
            df: DataFrame containing the data
            target_col: Name of target column

        Returns:
            Plotly Figure object
        """
        # Get feature columns (exclude target and y_true)
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in [target_col, 'y_true']]

        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            correlations[col] = df[col].corr(df[target_col])

        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)
        features, corr_values = zip(*sorted_features)

        # Create color based on positive/negative
        colors = ['green' if c > 0 else 'red' for c in corr_values]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(features),
            y=list(corr_values),
            marker_color=colors,
            hovertemplate='%{x}<br>Correlation: %{y:.3f}<extra></extra>'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="black")

        fig.update_layout(
            title=f"Feature Correlation with {target_col}",
            xaxis_title="Features",
            yaxis_title="Correlation Coefficient",
            template='plotly_white',
            height=500
        )

        return fig
