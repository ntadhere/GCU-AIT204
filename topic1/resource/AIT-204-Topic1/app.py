"""
Synthetic Dataset Generator for Regression Analysis
AIT-204 Topic 1: Background Math and Gradient-Based Learning

Main Streamlit Application

This interactive application allows students to:
1. Generate various types of synthetic regression datasets
2. Visualize data patterns and relationships
3. Perform comprehensive statistical analysis
4. Export datasets for use in machine learning models

The generated datasets are perfect for understanding gradient descent,
loss functions, and the fundamentals of regression modeling.

Author: AIT-204 Course Materials
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_generators import SyntheticDataGenerator
from statistics_analysis import StatisticalAnalyzer
from visualizations import DataVisualizer
import io

# Page configuration
st.set_page_config(
    page_title="Synthetic Dataset Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for storing generated data
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None


def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">📊 Synthetic Dataset Generator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AIT-204 Topic 1: Background Math and Gradient-Based Learning</p>',
        unsafe_allow_html=True
    )

    # Introduction
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
        ### Purpose
        This tool generates synthetic datasets with known mathematical relationships,
        perfect for learning regression modeling and gradient descent optimization.

        ### Why Synthetic Data?
        - **Known Ground Truth**: We know the exact underlying function
        - **Controlled Noise**: Understand how noise affects learning
        - **Reproducibility**: Same parameters = same data (with fixed seed)
        - **Educational Value**: Learn relationships before modeling complexity

        ### Workflow
        1. **Select Dataset Type**: Choose from linear, polynomial, non-linear, etc.
        2. **Configure Parameters**: Adjust sample size, noise level, function parameters
        3. **Generate Data**: Create the synthetic dataset
        4. **Visualize**: Explore patterns and relationships
        5. **Analyze**: Examine statistical properties
        6. **Export**: Download for use in your regression models

        ### Key Concepts Covered
        - **Regression**: Predicting continuous output from input features
        - **Loss Functions**: Measuring prediction error (MSE, RMSE)
        - **Gradient Descent**: Optimization algorithm for finding best parameters
        - **Feature Engineering**: Creating and transforming input features
        - **Statistical Analysis**: Understanding data before modeling
        """)

    # Sidebar for dataset configuration
    st.sidebar.header("🎛️ Dataset Configuration")

    # Dataset type selection
    st.sidebar.subheader("1. Select Dataset Type")
    dataset_type = st.sidebar.selectbox(
        "Choose the type of relationship:",
        [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Polynomial Regression",
            "Sinusoidal Function",
            "Exponential Function",
            "Logarithmic Function",
            "Step Function",
            "Interaction Features",
            "Custom Function"
        ],
        help="Different dataset types demonstrate different mathematical relationships"
    )

    # Common parameters
    st.sidebar.subheader("2. General Parameters")

    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="More samples = more data points for learning. Typical: 100-500"
    )

    noise_std = st.sidebar.slider(
        "Noise Level (Standard Deviation)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Higher noise = harder to learn. Represents measurement error or randomness"
    )

    random_seed = st.sidebar.number_input(
        "Random Seed (for reproducibility)",
        min_value=0,
        max_value=9999,
        value=42,
        help="Same seed = same data. Change to get different random samples"
    )

    # Type-specific parameters
    st.sidebar.subheader("3. Function-Specific Parameters")

    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=random_seed)

    # Generate data based on selected type
    if dataset_type == "Simple Linear Regression":
        st.sidebar.markdown("**Equation**: y = mx + b + ε")

        slope = st.sidebar.slider("Slope (m)", -10.0, 10.0, 2.0, 0.5)
        intercept = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 1.0, 0.5)
        x_min = st.sidebar.number_input("X Min", value=0.0)
        x_max = st.sidebar.number_input("X Max", value=10.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_simple_linear(
                n_samples=n_samples,
                slope=slope,
                intercept=intercept,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Simple Linear"

    elif dataset_type == "Multiple Linear Regression":
        st.sidebar.markdown("**Equation**: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b + ε")

        n_features = st.sidebar.slider("Number of Features", 2, 10, 3)
        intercept = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 5.0, 0.5)
        feat_min = st.sidebar.number_input("Feature Min", value=-5.0)
        feat_max = st.sidebar.number_input("Feature Max", value=5.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_multiple_linear(
                n_samples=n_samples,
                n_features=n_features,
                intercept=intercept,
                noise_std=noise_std,
                feature_range=(feat_min, feat_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Multiple Linear"

    elif dataset_type == "Polynomial Regression":
        st.sidebar.markdown("**Equation**: y = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀ + ε")

        degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)
        x_min = st.sidebar.number_input("X Min", value=-5.0)
        x_max = st.sidebar.number_input("X Max", value=5.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_polynomial(
                n_samples=n_samples,
                degree=degree,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = f"Polynomial (degree {degree})"

    elif dataset_type == "Sinusoidal Function":
        st.sidebar.markdown("**Equation**: y = A·sin(ωx + φ) + c + ε")

        amplitude = st.sidebar.slider("Amplitude (A)", 0.1, 10.0, 5.0, 0.5)
        frequency = st.sidebar.slider("Frequency (ω)", 0.1, 5.0, 1.0, 0.1)
        phase = st.sidebar.slider("Phase (φ)", 0.0, 6.28, 0.0, 0.1)
        offset = st.sidebar.slider("Offset (c)", -5.0, 5.0, 0.0, 0.5)
        x_min = st.sidebar.number_input("X Min", value=0.0)
        x_max = st.sidebar.number_input("X Max", value=10.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_sinusoidal(
                n_samples=n_samples,
                amplitude=amplitude,
                frequency=frequency,
                phase=phase,
                offset=offset,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Sinusoidal"

    elif dataset_type == "Exponential Function":
        st.sidebar.markdown("**Equation**: y = scale·baseˣ + offset + ε")

        base = st.sidebar.slider("Base", 1.1, 3.0, np.e, 0.1)
        scale = st.sidebar.slider("Scale", 0.1, 5.0, 1.0, 0.1)
        offset = st.sidebar.slider("Offset", -5.0, 5.0, 0.0, 0.5)
        x_min = st.sidebar.number_input("X Min", value=0.0)
        x_max = st.sidebar.number_input("X Max", value=3.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_exponential(
                n_samples=n_samples,
                base=base,
                scale=scale,
                offset=offset,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Exponential"

    elif dataset_type == "Logarithmic Function":
        st.sidebar.markdown("**Equation**: y = scale·log(x + offset) + ε")

        scale = st.sidebar.slider("Scale", 0.1, 10.0, 5.0, 0.5)
        offset = st.sidebar.slider("Offset", 0.1, 5.0, 1.0, 0.1)
        x_min = st.sidebar.number_input("X Min", value=0.1)
        x_max = st.sidebar.number_input("X Max", value=10.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_logarithmic(
                n_samples=n_samples,
                scale=scale,
                offset=offset,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Logarithmic"

    elif dataset_type == "Step Function":
        st.sidebar.markdown("**Equation**: Piecewise constant function")

        n_steps = st.sidebar.slider("Number of Steps", 2, 10, 3)
        step_height = st.sidebar.slider("Step Height", 0.5, 5.0, 2.0, 0.5)
        x_min = st.sidebar.number_input("X Min", value=0.0)
        x_max = st.sidebar.number_input("X Max", value=10.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_step_function(
                n_samples=n_samples,
                n_steps=n_steps,
                step_height=step_height,
                noise_std=noise_std,
                x_range=(x_min, x_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Step Function"

    elif dataset_type == "Interaction Features":
        st.sidebar.markdown("**Equation**: y = w₁x₁ + w₂x₂ + w₁₂(x₁·x₂) + b + ε")

        w1 = st.sidebar.slider("Weight 1 (w₁)", -5.0, 5.0, 2.0, 0.5)
        w2 = st.sidebar.slider("Weight 2 (w₂)", -5.0, 5.0, 3.0, 0.5)
        w_interaction = st.sidebar.slider("Interaction Weight (w₁₂)", -5.0, 5.0, 1.5, 0.5)
        intercept = st.sidebar.slider("Intercept (b)", -5.0, 5.0, 1.0, 0.5)
        feat_min = st.sidebar.number_input("Feature Min", value=-5.0)
        feat_max = st.sidebar.number_input("Feature Max", value=5.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            df = generator.generate_interaction_features(
                n_samples=n_samples,
                w1=w1,
                w2=w2,
                w_interaction=w_interaction,
                intercept=intercept,
                noise_std=noise_std,
                feature_range=(feat_min, feat_max)
            )
            st.session_state.generated_data = df
            st.session_state.dataset_name = "Interaction Features"

    elif dataset_type == "Custom Function":
        st.sidebar.markdown("**Define your own function using Python/NumPy syntax**")

        func_str = st.sidebar.text_input(
            "Function (use 'x' as variable)",
            value="x**2 + 2*x + 1",
            help="Examples: 'x**2', 'np.sin(x)', '2*x + 1', 'np.exp(x)'"
        )
        x_min = st.sidebar.number_input("X Min", value=-5.0)
        x_max = st.sidebar.number_input("X Max", value=5.0)

        if st.sidebar.button("🎲 Generate Dataset", type="primary"):
            try:
                df = generator.generate_custom_function(
                    n_samples=n_samples,
                    func_str=func_str,
                    noise_std=noise_std,
                    x_range=(x_min, x_max)
                )
                st.session_state.generated_data = df
                st.session_state.dataset_name = f"Custom: {func_str}"
            except Exception as e:
                st.sidebar.error(f"Error in function: {str(e)}")

    # Main content area
    if st.session_state.generated_data is not None:
        df = st.session_state.generated_data

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Dataset Preview",
            "📊 Visualizations",
            "📈 Statistical Analysis",
            "💾 Export Data"
        ])

        # Tab 1: Dataset Preview
        with tab1:
            st.subheader(f"Generated Dataset: {st.session_state.dataset_name}")

            # Dataset info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len([c for c in df.columns if c not in ['y', 'y_true']]))
            with col3:
                if 'y_true' in df.columns:
                    mse = np.mean((df['y'] - df['y_true']) ** 2)
                    st.metric("Noise MSE", f"{mse:.4f}")
                else:
                    st.metric("Noise MSE", "N/A")
            with col4:
                if 'y_true' in df.columns:
                    rmse = np.sqrt(np.mean((df['y'] - df['y_true']) ** 2))
                    st.metric("Noise RMSE", f"{rmse:.4f}")
                else:
                    st.metric("Noise RMSE", "N/A")

            # Show data table
            st.markdown("**Dataset Preview (first 100 rows)**")
            st.dataframe(df.head(100), width="stretch")

            # Basic statistics
            st.markdown("**Quick Statistics**")
            st.dataframe(df.describe(), width="stretch")

        # Tab 2: Visualizations
        with tab2:
            st.subheader("Data Visualizations")

            viz = DataVisualizer()

            # Determine visualization type based on data structure
            feature_cols = [c for c in df.columns if c not in ['y', 'y_true']]

            if len(feature_cols) == 1:
                # Single feature - scatter plot
                st.markdown("### Scatter Plot with True Function")
                st.markdown(
                    "Blue points show observed data with noise. "
                    "Red line shows the true underlying function."
                )
                fig = viz.plot_scatter_1d(df, feature_cols[0], 'y', show_true=True)
                st.plotly_chart(fig, width="stretch")

                # Residual plot
                if 'y_true' in df.columns:
                    st.markdown("### Residual Plot")
                    st.markdown(
                        "Residuals = Observed - True. "
                        "Random scatter around zero indicates good data quality."
                    )
                    fig_resid = viz.plot_residuals(df, 'y', 'y_true', feature_cols[0])
                    st.plotly_chart(fig_resid, width="stretch")

            elif len(feature_cols) == 2:
                # Two features - 3D plot
                st.markdown("### 3D Scatter Plot")
                st.markdown("Visualizing the relationship between two features and the target.")
                fig_3d = viz.plot_3d_scatter(df, feature_cols[0], feature_cols[1], 'y')
                st.plotly_chart(fig_3d, width="stretch")

            elif len(feature_cols) >= 3:
                # Multiple features - pairwise scatter
                st.markdown("### Pairwise Feature Relationships")
                st.markdown("Shows all pairwise scatter plots between features.")
                fig_pair = viz.plot_pairwise_scatter(df, max_features=min(5, len(feature_cols)))
                st.plotly_chart(fig_pair, width="stretch")

            # Distribution plots
            st.markdown("### Feature Distributions")
            selected_col = st.selectbox(
                "Select column to visualize distribution:",
                df.columns.tolist()
            )
            fig_dist = viz.plot_distribution(df, selected_col)
            st.plotly_chart(fig_dist, width="stretch")

            # Q-Q plot for normality
            st.markdown("### Q-Q Plot (Normality Test)")
            st.markdown(
                "Points following the red line indicate normal distribution. "
                "Useful for checking if noise is Gaussian."
            )
            fig_qq = viz.plot_qq(df, selected_col)
            st.plotly_chart(fig_qq, width="stretch")

            # Correlation heatmap (if multiple features)
            if len(feature_cols) > 1:
                st.markdown("### Correlation Heatmap")
                st.markdown("Shows linear relationships between all variables.")
                fig_corr = viz.plot_correlation_heatmap(df)
                st.plotly_chart(fig_corr, width="stretch")

                # Feature importance
                st.markdown("### Feature Correlation with Target")
                st.markdown("Which features have the strongest relationship with the target?")
                fig_importance = viz.plot_feature_importance_correlation(df)
                st.plotly_chart(fig_importance, width="stretch")

            # Box plots
            st.markdown("### Box-and-Whisker Plots")
            st.markdown("Visualize distribution, outliers, and quartiles for all variables.")
            fig_box = viz.plot_box_whisker(df)
            st.plotly_chart(fig_box, width="stretch")

        # Tab 3: Statistical Analysis
        with tab3:
            st.subheader("Statistical Analysis")

            analyzer = StatisticalAnalyzer()

            # Summary report
            st.markdown("### Dataset Summary")
            summary = analyzer.compute_summary_report(df)

            col1, col2 = st.columns(2)
            with col1:
                st.json({k: v for k, v in summary.items() if k not in ['Noise Level', 'Target (y) Statistics']})
            with col2:
                if 'Target (y) Statistics' in summary:
                    st.markdown("**Target Variable Statistics**")
                    st.json(summary['Target (y) Statistics'])
                if 'Noise Level' in summary:
                    st.markdown("**Noise Characteristics**")
                    st.json(summary['Noise Level'])

            # Descriptive statistics
            st.markdown("### Comprehensive Descriptive Statistics")
            st.markdown(
                "Includes measures of central tendency (mean, median), "
                "spread (std, variance), and shape (skewness, kurtosis)."
            )
            desc_stats = analyzer.compute_descriptive_stats(df)
            st.dataframe(desc_stats, width="stretch")

            # Correlation analysis
            if len(feature_cols) > 1:
                st.markdown("### Correlation Matrix")
                st.markdown(
                    "Pearson correlation coefficients: "
                    "+1 = perfect positive, -1 = perfect negative, 0 = no linear relationship"
                )
                corr_matrix = analyzer.compute_correlation_matrix(df)
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1),
                           width="stretch")

                # Feature-target relationships
                st.markdown("### Feature-Target Relationships")
                st.markdown("Which features are most predictive of the target variable?")
                feat_target_stats = analyzer.compute_feature_target_stats(df)
                st.dataframe(feat_target_stats, width="stretch")

            # Normality tests
            st.markdown("### Normality Tests")
            st.markdown(
                "Shapiro-Wilk test: p > 0.05 suggests data is normally distributed. "
                "Important for understanding noise and residuals."
            )

            norm_col = st.selectbox(
                "Select column for normality test:",
                df.columns.tolist(),
                key="normality_test"
            )

            norm_result = analyzer.normality_test(df, norm_col)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Statistic", f"{norm_result['Statistic']:.6f}")
            with col2:
                st.metric("P-value", f"{norm_result['P-value']:.6f}")
            with col3:
                st.metric("Result", norm_result['Interpretation'])

            st.json(norm_result)

            # Outlier detection
            st.markdown("### Outlier Detection")

            outlier_method = st.radio(
                "Select outlier detection method:",
                ["IQR Method", "Z-Score Method"]
            )

            outlier_col = st.selectbox(
                "Select column for outlier detection:",
                df.columns.tolist(),
                key="outlier_detection"
            )

            if outlier_method == "IQR Method":
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.5)
                outlier_result = analyzer.detect_outliers_iqr(df, outlier_col, multiplier)
            else:
                threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.5)
                outlier_result = analyzer.detect_outliers_zscore(df, outlier_col, threshold)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Outliers", outlier_result['Number of Outliers'])
            with col2:
                st.metric("Percentage", f"{outlier_result['Percentage of Outliers']:.2f}%")

            st.json(outlier_result)

        # Tab 4: Export Data
        with tab4:
            st.subheader("Export Generated Dataset")

            st.markdown("""
            Download the generated dataset in your preferred format:
            - **CSV**: Universal format, works with all tools
            - **Excel**: Formatted spreadsheet
            - **JSON**: For web applications and APIs
            - **NumPy**: For direct Python/NumPy usage
            """)

            col1, col2 = st.columns(2)

            with col1:
                # CSV export
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"synthetic_data_{st.session_state.dataset_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

                # Excel export
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"synthetic_data_{st.session_state.dataset_name.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col2:
                # JSON export
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"synthetic_data_{st.session_state.dataset_name.replace(' ', '_')}.json",
                    mime="application/json"
                )

                # NumPy export instructions
                st.markdown("**For NumPy/Python:**")
                st.code("""
# After downloading CSV:
import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_data.csv')
X = df.drop(['y', 'y_true'], axis=1).values
y = df['y'].values
y_true = df['y_true'].values  # if available
                """, language='python')

            # Export metadata
            st.markdown("### Dataset Metadata")
            st.markdown("Save this information along with your dataset for reproducibility:")

            metadata = {
                "Dataset Type": st.session_state.dataset_name,
                "Number of Samples": len(df),
                "Number of Features": len(feature_cols),
                "Random Seed": random_seed,
                "Noise Std Dev": noise_std,
                "Generated At": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            st.json(metadata)

    else:
        # No data generated yet - show instructions
        st.markdown("""
        <div class="info-box">
        <h3>👈 Getting Started</h3>
        <ol>
            <li>Select a dataset type from the sidebar</li>
            <li>Adjust the parameters to control the data generation</li>
            <li>Click the <strong>"🎲 Generate Dataset"</strong> button</li>
            <li>Explore the data using the tabs above</li>
            <li>Export your dataset for regression modeling</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        # Educational content
        st.markdown("### 📚 Understanding Regression Datasets")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Linear Relationships
            - **Simple Linear**: y = mx + b
            - **Multiple Linear**: y = w₁x₁ + w₂x₂ + ... + b
            - Best for learning gradient descent basics
            - Has analytical solution (closed form)
            """)

            st.markdown("""
            #### Non-Linear Relationships
            - **Polynomial**: Curves and parabolas
            - **Sinusoidal**: Periodic patterns
            - **Exponential**: Growth/decay
            - Requires feature engineering or neural networks
            """)

        with col2:
            st.markdown("""
            #### Key Parameters
            - **Sample Size**: More samples = better learning
            - **Noise Level**: Higher noise = harder task
            - **Feature Range**: Affects gradient magnitudes
            - **Random Seed**: For reproducible results
            """)

            st.markdown("""
            #### Learning Objectives
            - Understand relationship types
            - Visualize data before modeling
            - Analyze statistical properties
            - Practice gradient descent optimization
            """)

        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
        <strong>💡 Pro Tip:</strong> Start with simple linear regression and low noise
        to understand the basics, then gradually increase complexity!
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
