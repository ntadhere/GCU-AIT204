"""
Quick Test Script for Synthetic Dataset Generator
Tests that all modules load correctly and basic functionality works.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from data_generators import SyntheticDataGenerator
        from statistics_analysis import StatisticalAnalyzer
        from visualizations import DataVisualizer
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_generation():
    """Test basic data generation."""
    print("\nTesting data generation...")
    try:
        from data_generators import SyntheticDataGenerator
        import pandas as pd

        generator = SyntheticDataGenerator(random_seed=42)

        # Test simple linear
        df = generator.generate_simple_linear(n_samples=50)
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == 50, "Should have 50 samples"
        assert 'x' in df.columns, "Should have x column"
        assert 'y' in df.columns, "Should have y column"
        print("  ✓ Simple linear generation works")

        # Test multiple linear
        df = generator.generate_multiple_linear(n_samples=50, n_features=3)
        assert len(df.columns) >= 5, "Should have at least 5 columns (3 features + y + y_true)"
        print("  ✓ Multiple linear generation works")

        # Test polynomial
        df = generator.generate_polynomial(n_samples=50, degree=2)
        assert len(df) == 50, "Should have 50 samples"
        print("  ✓ Polynomial generation works")

        return True
    except Exception as e:
        print(f"  ❌ Data generation error: {e}")
        return False


def test_statistics():
    """Test statistical analysis."""
    print("\nTesting statistical analysis...")
    try:
        from data_generators import SyntheticDataGenerator
        from statistics_analysis import StatisticalAnalyzer

        generator = SyntheticDataGenerator(random_seed=42)
        df = generator.generate_simple_linear(n_samples=100)

        analyzer = StatisticalAnalyzer()

        # Test descriptive stats
        stats = analyzer.compute_descriptive_stats(df)
        assert stats is not None, "Should return stats"
        print("  ✓ Descriptive statistics work")

        # Test correlation
        corr = analyzer.compute_correlation_matrix(df)
        assert corr is not None, "Should return correlation matrix"
        print("  ✓ Correlation analysis works")

        # Test outlier detection
        outliers = analyzer.detect_outliers_iqr(df, 'y')
        assert 'Number of Outliers' in outliers, "Should return outlier info"
        print("  ✓ Outlier detection works")

        return True
    except Exception as e:
        print(f"  ❌ Statistics error: {e}")
        return False


def test_visualization():
    """Test visualization creation."""
    print("\nTesting visualization...")
    try:
        from data_generators import SyntheticDataGenerator
        from visualizations import DataVisualizer
        import plotly.graph_objects as go

        generator = SyntheticDataGenerator(random_seed=42)
        df = generator.generate_simple_linear(n_samples=100)

        viz = DataVisualizer()

        # Test scatter plot
        fig = viz.plot_scatter_1d(df)
        assert isinstance(fig, go.Figure), "Should return Plotly figure"
        print("  ✓ Scatter plot creation works")

        # Test distribution plot
        fig = viz.plot_distribution(df, 'y')
        assert isinstance(fig, go.Figure), "Should return Plotly figure"
        print("  ✓ Distribution plot creation works")

        return True
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Synthetic Dataset Generator - Module Tests")
    print("=" * 60)

    results = []

    results.append(("Import Test", test_imports()))
    results.append(("Data Generation Test", test_data_generation()))
    results.append(("Statistics Test", test_statistics()))
    results.append(("Visualization Test", test_visualization()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! The modules are working correctly.")
        print("\nNext steps:")
        print("  1. Run 'streamlit run app.py' to start the web interface")
        print("  2. Or run 'python3 example_usage.py' to see programmatic usage")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
