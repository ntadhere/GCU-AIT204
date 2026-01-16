"""
Example Usage of Synthetic Dataset Generator
AIT-204 Topic 1: Background Math and Gradient-Based Learning

This script demonstrates how to use the data generation, visualization,
and analysis modules programmatically (without the Streamlit UI).

Perfect for:
- Batch data generation
- Integration with your own scripts
- Automated experiments
- Custom workflows
"""

import numpy as np
import pandas as pd
from data_generators import SyntheticDataGenerator
from statistics_analysis import StatisticalAnalyzer
from visualizations import DataVisualizer


def example_1_simple_linear():
    """
    Example 1: Generate and analyze simple linear regression data.

    This is the most basic regression problem:
    y = mx + b + noise

    Perfect for learning gradient descent fundamentals.
    """
    print("=" * 70)
    print("EXAMPLE 1: Simple Linear Regression")
    print("=" * 70)

    # Initialize generator with fixed seed for reproducibility
    generator = SyntheticDataGenerator(random_seed=42)

    # Generate data: y = 2.5x + 1.0 + noise
    df = generator.generate_simple_linear(
        n_samples=200,
        slope=2.5,
        intercept=1.0,
        noise_std=0.5,
        x_range=(0, 10)
    )

    print(f"\nGenerated {len(df)} samples")
    print("\nFirst 5 rows:")
    print(df.head())

    # Compute statistics
    analyzer = StatisticalAnalyzer()
    stats = analyzer.compute_descriptive_stats(df)

    print("\nDescriptive Statistics:")
    print(stats)

    # Calculate noise level
    mse = analyzer.compute_mse_from_true(df)
    rmse = np.sqrt(mse)
    print(f"\nNoise Level:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    # Export data
    df.to_csv('simple_linear_data.csv', index=False)
    print("\n✓ Data saved to 'simple_linear_data.csv'")

    return df


def example_2_polynomial():
    """
    Example 2: Generate polynomial regression data.

    Demonstrates non-linear relationships that can be addressed
    through feature engineering (polynomial features).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Polynomial Regression")
    print("=" * 70)

    generator = SyntheticDataGenerator(random_seed=123)

    # Generate quadratic data
    df = generator.generate_polynomial(
        n_samples=150,
        degree=2,  # Quadratic
        coefficients=[1.0, 2.0, 0.5],  # y = 0.5x² + 2x + 1
        noise_std=1.5,
        x_range=(-5, 5)
    )

    print(f"\nGenerated {len(df)} samples with quadratic relationship")
    print(f"True function: y = 0.5x² + 2x + 1.0")

    # Analyze correlation
    analyzer = StatisticalAnalyzer()
    correlation = df['x'].corr(df['y'])
    print(f"\nLinear correlation: {correlation:.4f}")
    print("(Note: Correlation measures linear relationship,")
    print(" but this data has a quadratic relationship)")

    # Export
    df.to_csv('polynomial_data.csv', index=False)
    print("\n✓ Data saved to 'polynomial_data.csv'")

    return df


def example_3_multiple_features():
    """
    Example 3: Generate multiple linear regression data.

    Demonstrates high-dimensional input space with multiple features.
    Important for understanding partial derivatives in gradient descent.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Linear Regression")
    print("=" * 70)

    generator = SyntheticDataGenerator(random_seed=456)

    # Generate data with 5 features
    df = generator.generate_multiple_linear(
        n_samples=300,
        n_features=5,
        coefficients=[2.0, -1.5, 3.0, 0.5, -2.5],  # Specific weights
        intercept=10.0,
        noise_std=2.0,
        feature_range=(-10, 10)
    )

    print(f"\nGenerated {len(df)} samples with 5 features")

    # Analyze feature importance
    analyzer = StatisticalAnalyzer()
    feature_stats = analyzer.compute_feature_target_stats(df, target_col='y')

    print("\nFeature Importance (by correlation with target):")
    print(feature_stats)

    # Correlation matrix
    corr_matrix = analyzer.compute_correlation_matrix(df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Export
    df.to_csv('multiple_linear_data.csv', index=False)
    print("\n✓ Data saved to 'multiple_linear_data.csv'")

    return df


def example_4_nonlinear():
    """
    Example 4: Generate sinusoidal (non-linear) data.

    Demonstrates smooth non-linear relationships.
    Requires neural networks or kernel methods for modeling.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Non-Linear (Sinusoidal) Regression")
    print("=" * 70)

    generator = SyntheticDataGenerator(random_seed=789)

    # Generate sinusoidal data
    df = generator.generate_sinusoidal(
        n_samples=250,
        amplitude=5.0,
        frequency=2.0,
        phase=0.5,
        offset=2.0,
        noise_std=0.8,
        x_range=(0, 2 * np.pi)
    )

    print(f"\nGenerated {len(df)} samples with sinusoidal pattern")
    print(f"Function: y = 5.0 * sin(2.0x + 0.5) + 2.0")

    # Test normality of residuals
    analyzer = StatisticalAnalyzer()
    if 'y_true' in df.columns:
        residuals = df['y'] - df['y_true']
        df_temp = pd.DataFrame({'residuals': residuals})
        normality = analyzer.normality_test(df_temp, 'residuals')

        print(f"\nNormality test of residuals:")
        print(f"  Test: {normality['Test']}")
        print(f"  P-value: {normality['P-value']:.6f}")
        print(f"  Interpretation: {normality['Interpretation']}")

    # Export
    df.to_csv('sinusoidal_data.csv', index=False)
    print("\n✓ Data saved to 'sinusoidal_data.csv'")

    return df


def example_5_outlier_detection():
    """
    Example 5: Generate data and detect outliers.

    Demonstrates statistical outlier detection methods.
    Important for data quality assessment.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Outlier Detection")
    print("=" * 70)

    generator = SyntheticDataGenerator(random_seed=999)

    # Generate data with some natural variation
    df = generator.generate_simple_linear(
        n_samples=200,
        slope=3.0,
        intercept=2.0,
        noise_std=2.0,  # Higher noise
        x_range=(0, 20)
    )

    # Manually add some outliers for demonstration
    outlier_indices = np.random.choice(len(df), size=5, replace=False)
    df.loc[outlier_indices, 'y'] += np.random.choice([-15, 15], size=5)

    print(f"\nGenerated {len(df)} samples (with 5 artificial outliers)")

    # Detect outliers using IQR method
    analyzer = StatisticalAnalyzer()
    outliers_iqr = analyzer.detect_outliers_iqr(df, 'y', multiplier=1.5)

    print("\nIQR Method Results:")
    print(f"  Number of outliers: {outliers_iqr['Number of Outliers']}")
    print(f"  Percentage: {outliers_iqr['Percentage of Outliers']:.2f}%")
    print(f"  Lower bound: {outliers_iqr['Lower Bound']:.2f}")
    print(f"  Upper bound: {outliers_iqr['Upper Bound']:.2f}")

    # Detect outliers using Z-score method
    outliers_z = analyzer.detect_outliers_zscore(df, 'y', threshold=3.0)

    print("\nZ-Score Method Results:")
    print(f"  Number of outliers: {outliers_z['Number of Outliers']}")
    print(f"  Percentage: {outliers_z['Percentage of Outliers']:.2f}%")
    print(f"  Max Z-score: {outliers_z['Max Z-Score']:.2f}")

    return df


def example_6_custom_function():
    """
    Example 6: Generate data from a custom mathematical function.

    Demonstrates flexibility to create any functional relationship.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Function")
    print("=" * 70)

    generator = SyntheticDataGenerator(random_seed=111)

    # Create a custom complex function
    custom_func = "np.sin(x) * np.exp(-x/10) + x/5"

    df = generator.generate_custom_function(
        n_samples=200,
        func_str=custom_func,
        noise_std=0.5,
        x_range=(0, 15)
    )

    print(f"\nGenerated {len(df)} samples")
    print(f"Custom function: {custom_func}")

    # Show some statistics
    analyzer = StatisticalAnalyzer()
    summary = analyzer.compute_summary_report(df)

    print("\nDataset Summary:")
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    # Export
    df.to_csv('custom_function_data.csv', index=False)
    print("\n✓ Data saved to 'custom_function_data.csv'")

    return df


def example_7_batch_generation():
    """
    Example 7: Generate multiple datasets in batch.

    Useful for creating training/validation/test splits or
    for comparative experiments.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Batch Dataset Generation")
    print("=" * 70)

    # Generate datasets with varying noise levels
    noise_levels = [0.5, 1.0, 2.0, 5.0]

    datasets = {}

    for noise in noise_levels:
        generator = SyntheticDataGenerator(random_seed=42)  # Same seed for fairness

        df = generator.generate_simple_linear(
            n_samples=150,
            slope=2.0,
            intercept=1.0,
            noise_std=noise,
            x_range=(0, 10)
        )

        datasets[f'noise_{noise}'] = df

        # Save each dataset
        filename = f'linear_noise_{noise}.csv'
        df.to_csv(filename, index=False)

        # Compute and display metrics
        analyzer = StatisticalAnalyzer()
        mse = analyzer.compute_mse_from_true(df)

        print(f"\nNoise Std: {noise}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  ✓ Saved to '{filename}'")

    print(f"\nGenerated {len(datasets)} datasets with varying noise levels")

    return datasets


def example_8_visualization():
    """
    Example 8: Create and save visualizations.

    Demonstrates programmatic visualization creation.
    Note: Plotly figures can be saved as HTML or displayed in browser.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Programmatic Visualization")
    print("=" * 70)

    # Generate sample data
    generator = SyntheticDataGenerator(random_seed=42)
    df = generator.generate_polynomial(
        n_samples=150,
        degree=2,
        noise_std=2.0,
        x_range=(-5, 5)
    )

    # Create visualizer
    viz = DataVisualizer()

    # Create scatter plot
    fig_scatter = viz.plot_scatter_1d(df, 'x', 'y', show_true=True)

    # Save as HTML (can be opened in browser)
    fig_scatter.write_html('scatter_plot.html')
    print("\n✓ Scatter plot saved to 'scatter_plot.html'")

    # Create distribution plot
    fig_dist = viz.plot_distribution(df, 'y', bins=30)
    fig_dist.write_html('distribution_plot.html')
    print("✓ Distribution plot saved to 'distribution_plot.html'")

    # Create residual plot
    fig_resid = viz.plot_residuals(df, 'y', 'y_true', 'x')
    fig_resid.write_html('residual_plot.html')
    print("✓ Residual plot saved to 'residual_plot.html'")

    # Create Q-Q plot
    fig_qq = viz.plot_qq(df, 'y')
    fig_qq.write_html('qq_plot.html')
    print("✓ Q-Q plot saved to 'qq_plot.html'")

    print("\nAll plots saved as HTML files. Open them in a web browser.")

    return df


def main():
    """
    Run all examples.

    Comment out examples you don't want to run.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Synthetic Dataset Generator - Example Usage".center(68) + "║")
    print("║" + "  AIT-204 Topic 1: Background Math and Gradient-Based Learning".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Run all examples
        example_1_simple_linear()
        example_2_polynomial()
        example_3_multiple_features()
        example_4_nonlinear()
        example_5_outlier_detection()
        example_6_custom_function()
        example_7_batch_generation()
        example_8_visualization()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  - simple_linear_data.csv")
        print("  - polynomial_data.csv")
        print("  - multiple_linear_data.csv")
        print("  - sinusoidal_data.csv")
        print("  - custom_function_data.csv")
        print("  - linear_noise_*.csv (multiple files)")
        print("  - *.html (visualization files)")
        print("\nNext Steps:")
        print("  1. Examine the CSV files")
        print("  2. Open HTML files in a web browser")
        print("  3. Use the data for gradient descent implementation")
        print("  4. Experiment with different parameters")
        print("  5. Try the Streamlit app: streamlit run app.py")
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
