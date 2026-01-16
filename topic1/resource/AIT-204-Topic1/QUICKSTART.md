# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Verify Installation
```bash
cd /Users/isac/Desktop/AIT-204-code-and-resources/Topic1-math-gradient-descent
python3 test_modules.py
```

You should see: `🎉 All tests passed!`

### Step 2: Launch the Web Application
```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

### Step 3: Generate Your First Dataset

1. **Select Dataset Type**: Choose "Simple Linear Regression" from the sidebar
2. **Set Parameters**:
   - Number of Samples: 200
   - Noise Level: 1.0
   - Slope: 2.5
   - Intercept: 1.0
3. **Click** "🎲 Generate Dataset"
4. **Explore** the tabs:
   - Dataset Preview: See your data
   - Visualizations: Interactive plots
   - Statistical Analysis: Comprehensive stats
   - Export Data: Download CSV/Excel

---

## 📚 Alternative: Programmatic Usage

If you prefer Python scripts over the web interface:

```bash
python3 example_usage.py
```

This will generate multiple example datasets and save them as CSV files.

---

## 🎯 Quick Examples

### Example 1: Simple Linear Regression
```python
from data_generators import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_seed=42)
df = generator.generate_simple_linear(
    n_samples=200,
    slope=2.5,
    intercept=1.0,
    noise_std=0.5
)

df.to_csv('my_data.csv', index=False)
```

### Example 2: Polynomial Regression
```python
from data_generators import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_seed=42)
df = generator.generate_polynomial(
    n_samples=150,
    degree=2,  # Quadratic
    noise_std=1.5
)
```

### Example 3: Visualize Data
```python
from visualizations import DataVisualizer

viz = DataVisualizer()
fig = viz.plot_scatter_1d(df, 'x', 'y', show_true=True)
fig.show()  # Opens in browser
```

### Example 4: Statistical Analysis
```python
from statistics_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
stats = analyzer.compute_descriptive_stats(df)
print(stats)
```

---

## 🔧 Troubleshooting

### Missing Packages?
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas scipy plotly streamlit openpyxl
```

### Port Already in Use?
```bash
streamlit run app.py --server.port 8502
```

### Can't Find Python?
Try `python` instead of `python3`:
```bash
python test_modules.py
python example_usage.py
```

---

## 📖 Next Steps

1. **Read the README**: Full documentation in `README.md`
2. **Try Examples**: Run `example_usage.py` for 8 different examples
3. **Experiment**: Modify parameters and see how data changes
4. **Learn**: Use generated data for gradient descent implementation

---

## 🎓 Learning Path

### Beginner
1. Generate simple linear data (low noise)
2. Visualize the scatter plot
3. Examine descriptive statistics
4. Export and use in your code

### Intermediate
1. Try multiple linear regression
2. Study correlation matrices
3. Detect outliers
4. Compare different noise levels

### Advanced
1. Generate non-linear data (sine, exponential)
2. Create custom functions
3. Analyze feature interactions
4. Design experiments with multiple datasets

---

## 📁 Files Overview

- `app.py` - Main Streamlit web application
- `data_generators.py` - Dataset generation functions
- `statistics_analysis.py` - Statistical analysis tools
- `visualizations.py` - Plotting functions
- `example_usage.py` - Example scripts
- `test_modules.py` - Verification tests
- `requirements.txt` - Dependencies
- `README.md` - Full documentation

---

## 💡 Pro Tips

- **Use the same random seed** for reproducible results
- **Start with low noise** (0.5-1.0) to see patterns clearly
- **Visualize first** before statistical analysis
- **Export metadata** along with your datasets
- **Try batch generation** for training/validation/test splits

---

**Happy Learning! 🎓**
