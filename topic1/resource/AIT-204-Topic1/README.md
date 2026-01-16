# Synthetic Dataset Generator for Regression Analysis

**AIT-204 Topic 1: Background Math and Gradient-Based Learning**

A comprehensive tool for generating, visualizing, and analyzing synthetic regression datasets. Perfect for learning gradient descent, regression modeling, and understanding the mathematics behind machine learning.

## Features

### Dataset Generation
- **Simple Linear Regression**: y = mx + b + ε
- **Multiple Linear Regression**: y = w₁x₁ + w₂x₂ + ... + b + ε
- **Polynomial Regression**: y = aₙxⁿ + ... + a₁x + a₀ + ε
- **Sinusoidal Functions**: y = A·sin(ωx + φ) + c + ε
- **Exponential Functions**: y = scale·baseˣ + offset + ε
- **Logarithmic Functions**: y = scale·log(x + offset) + ε
- **Step Functions**: Piecewise constant
- **Interaction Features**: y = w₁x₁ + w₂x₂ + w₁₂(x₁·x₂) + b + ε
- **Custom Functions**: Define your own mathematical relationship

### Visualization Tools
- Interactive scatter plots with true function overlay
- 3D scatter plots for multi-feature data
- Residual plots for error analysis
- Distribution histograms with statistical overlays
- Correlation heatmaps
- Box-and-whisker plots
- Q-Q plots for normality testing
- Pairwise scatter matrices
- Feature importance visualization

### Statistical Analysis
- Comprehensive descriptive statistics
- Correlation and covariance matrices
- Outlier detection (IQR and Z-score methods)
- Normality tests (Shapiro-Wilk)
- Feature-target relationship analysis
- Noise level quantification (MSE, RMSE, SNR)

### Data Export
- CSV format
- Excel format (.xlsx)
- JSON format
- Easy integration with NumPy/scikit-learn

## Installation

### 1. Clone or Download
```bash
cd /path/to/AIT-204-code-and-resources/Topic1-math-gradient-descent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas scipy plotly streamlit openpyxl
```

## Usage

### Running the Streamlit Web App

```bash
streamlit run app.py
```

This will open a browser window with the interactive application.

### Using the Python Modules Directly

```python
from data_generators import SyntheticDataGenerator
from statistics_analysis import StatisticalAnalyzer
from visualizations import DataVisualizer

# Initialize generator
generator = SyntheticDataGenerator(random_seed=42)

# Generate simple linear data
df = generator.generate_simple_linear(
    n_samples=200,
    slope=2.5,
    intercept=1.0,
    noise_std=0.5,
    x_range=(0, 10)
)

# Analyze statistics
analyzer = StatisticalAnalyzer()
stats = analyzer.compute_descriptive_stats(df)
print(stats)

# Create visualizations
visualizer = DataVisualizer()
fig = visualizer.plot_scatter_1d(df, 'x', 'y', show_true=True)
fig.show()

# Export data
df.to_csv('my_regression_data.csv', index=False)
```

## File Structure

```
Topic1-math-gradient-descent/
│
├── app.py                      # Main Streamlit application
├── data_generators.py          # Dataset generation module
├── statistics_analysis.py      # Statistical analysis module
├── visualizations.py           # Visualization module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── example_usage.py           # Example usage script
```

## Educational Context

This tool is designed for **AIT-204: Introduction to Deep Learning - Topic 1**, which covers:

### Background Mathematics
- Linear algebra (vectors, matrices, dot products)
- Calculus (derivatives, partial derivatives)
- Probability and statistics (distributions, noise)

### Gradient-Based Learning
- Loss functions (MSE, MAE)
- Gradient descent optimization
- Learning rate and convergence
- Batch vs stochastic gradient descent

### Regression Fundamentals
- Linear regression as a learning problem
- Feature engineering and polynomial features
- Overfitting and underfitting
- Model evaluation metrics

## How to Use for Learning

### Beginner Path
1. **Start Simple**: Generate simple linear regression data with low noise
2. **Visualize**: Understand the relationship between x and y
3. **Add Noise**: Increase noise level, observe impact on data
4. **Statistics**: Examine descriptive statistics and correlations
5. **Export**: Use the data to implement gradient descent

### Intermediate Path
1. **Multiple Features**: Generate multiple linear regression data
2. **Correlation**: Study feature correlations and multicollinearity
3. **Polynomial**: Create polynomial data, learn feature engineering
4. **Residuals**: Analyze residual plots, understand error patterns
5. **Comparison**: Compare different dataset complexities

### Advanced Path
1. **Non-Linear**: Generate sinusoidal or exponential data
2. **Interactions**: Explore feature interaction terms
3. **Custom**: Define custom mathematical functions
4. **Analysis**: Perform comprehensive statistical testing
5. **Experiments**: Design experiments with varying parameters

## Key Concepts Demonstrated

### Noise and Uncertainty
- **Aleatoric Uncertainty**: Irreducible noise in data
- **Signal-to-Noise Ratio**: Quality of observations
- **Impact on Learning**: How noise affects model training

### Function Complexity
- **Linear**: Simple, has closed-form solution
- **Polynomial**: Non-linear but can be linearized
- **Transcendental**: Requires neural networks or non-linear models

### Statistical Properties
- **Distribution**: Shape of data (normal, skewed, etc.)
- **Correlation**: Linear relationships between variables
- **Outliers**: Extreme values that affect learning

### Gradient Descent Implications
- **Convexity**: Linear problems have one global minimum
- **Landscape**: Non-linear problems may have local minima
- **Scaling**: Feature ranges affect gradient magnitudes

## Example Workflow

### Scenario: Learning Simple Linear Regression

1. **Generate Data**
   - Type: Simple Linear Regression
   - Samples: 100
   - Slope: 2.0
   - Intercept: 1.0
   - Noise: 0.5
   - Random Seed: 42

2. **Visualize**
   - Scatter plot shows positive linear trend
   - True function (red line) vs noisy observations (blue points)
   - Residual plot shows random scatter (good)

3. **Analyze**
   - Correlation ≈ 0.97 (strong linear relationship)
   - RMSE ≈ 0.5 (matches noise level)
   - Normal distribution check passes

4. **Export**
   - Download CSV file
   - Use in gradient descent implementation
   - Compare learned parameters to true values (m=2.0, b=1.0)

## Tips and Best Practices

### For Dataset Generation
- Use consistent random seeds for reproducibility
- Start with 100-500 samples (good balance)
- Keep noise reasonable (0.5-2.0 standard deviation)
- Match complexity to learning objective

### For Visualization
- Always check scatter plots first
- Use residual plots to diagnose problems
- Check Q-Q plots if assuming normality
- Correlation heatmaps for multi-feature data

### For Analysis
- Compare statistics to expected values
- Look for patterns in residuals
- Check for outliers that may affect learning
- Validate data quality before modeling

### For Learning
- Understand the true function before adding noise
- Experiment with different noise levels
- Try various function complexities
- Compare model performance across datasets

## Common Issues and Solutions

### Issue: "Module not found"
**Solution**: Install requirements with `pip install -r requirements.txt`

### Issue: Plot not showing
**Solution**: Streamlit plots appear in the web interface, not in terminal

### Issue: Custom function error
**Solution**: Use valid Python/NumPy syntax (e.g., `x**2`, not `x^2`)

### Issue: Too many features slow down
**Solution**: Limit features to 5-10 for interactive visualization

## Further Reading

### Mathematical Background
- "Pattern Recognition and Machine Learning" - Bishop
- "Deep Learning" - Goodfellow, Bengio, Courville
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

### Gradient Descent
- Andrew Ng's Machine Learning Course
- Stanford CS229 Lecture Notes
- "Optimization for Machine Learning" - Sra, Nowozin, Wright

### Regression Analysis
- "Applied Linear Regression" - Weisberg
- "Regression Analysis by Example" - Chatterjee, Hadi

## License

Educational use for AIT-204 course. Feel free to modify and extend.

## Author

AIT-204 Course Materials

## Support

For issues or questions:
1. Check this README
2. Review example_usage.py
3. Consult course materials
4. Ask instructor or TA

---

**Happy Learning! 🎓📊**
