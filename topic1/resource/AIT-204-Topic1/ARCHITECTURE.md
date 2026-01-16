# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  SYNTHETIC DATASET GENERATOR                     │
│            AIT-204 Topic 1: Gradient-Based Learning             │
└─────────────────────────────────────────────────────────────────┘

                              ┌──────────┐
                              │   USER   │
                              └────┬─────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼─────┐              ┌───────▼────────┐
              │ Web UI    │              │ Python API     │
              │ (Streamlit)│              │ (Direct Import)│
              └─────┬─────┘              └───────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      CORE MODULES           │
                    │  ┌─────────────────────┐   │
                    │  │ data_generators.py  │   │
                    │  ├─────────────────────┤   │
                    │  │ statistics_analysis│   │
                    │  │        .py          │   │
                    │  ├─────────────────────┤   │
                    │  │ visualizations.py   │   │
                    │  └─────────────────────┘   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │        OUTPUT               │
                    │  ┌─────────────────────┐   │
                    │  │ CSV Files           │   │
                    │  ├─────────────────────┤   │
                    │  │ Excel Files         │   │
                    │  ├─────────────────────┤   │
                    │  │ JSON Files          │   │
                    │  ├─────────────────────┤   │
                    │  │ HTML Visualizations │   │
                    │  └─────────────────────┘   │
                    └─────────────────────────────┘
```

---

## Module Architecture

### 1. Data Generation Layer

```
┌─────────────────────────────────────────────────────────┐
│            data_generators.py                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  SyntheticDataGenerator                                 │
│  ├── __init__(random_seed)                              │
│  │                                                       │
│  ├── Linear Functions                                   │
│  │   ├── generate_simple_linear()                       │
│  │   └── generate_multiple_linear()                     │
│  │                                                       │
│  ├── Polynomial Functions                               │
│  │   └── generate_polynomial()                          │
│  │                                                       │
│  ├── Non-Linear Functions                               │
│  │   ├── generate_sinusoidal()                          │
│  │   ├── generate_exponential()                         │
│  │   └── generate_logarithmic()                         │
│  │                                                       │
│  ├── Special Functions                                  │
│  │   ├── generate_step_function()                       │
│  │   └── generate_interaction_features()                │
│  │                                                       │
│  └── Custom                                             │
│      └── generate_custom_function()                     │
│                                                          │
│  Input: Parameters (n_samples, noise_std, etc.)         │
│  Output: pandas.DataFrame with columns [features, y, y_true]│
└─────────────────────────────────────────────────────────┘
```

### 2. Statistical Analysis Layer

```
┌─────────────────────────────────────────────────────────┐
│         statistics_analysis.py                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  StatisticalAnalyzer                                    │
│  ├── Descriptive Statistics                             │
│  │   └── compute_descriptive_stats()                    │
│  │                                                       │
│  ├── Correlation Analysis                               │
│  │   ├── compute_correlation_matrix()                   │
│  │   ├── compute_covariance_matrix()                    │
│  │   └── compute_feature_target_stats()                 │
│  │                                                       │
│  ├── Outlier Detection                                  │
│  │   ├── detect_outliers_iqr()                          │
│  │   └── detect_outliers_zscore()                       │
│  │                                                       │
│  ├── Distribution Analysis                              │
│  │   └── normality_test()                               │
│  │                                                       │
│  ├── Error Metrics                                      │
│  │   └── compute_mse_from_true()                        │
│  │                                                       │
│  └── Summary                                            │
│      └── compute_summary_report()                       │
│                                                          │
│  Input: pandas.DataFrame                                │
│  Output: Statistics (DataFrame, Dict, or scalar)        │
└─────────────────────────────────────────────────────────┘
```

### 3. Visualization Layer

```
┌─────────────────────────────────────────────────────────┐
│           visualizations.py                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  DataVisualizer                                         │
│  ├── Basic Plots                                        │
│  │   ├── plot_scatter_1d()                              │
│  │   ├── plot_3d_scatter()                              │
│  │   └── plot_residuals()                               │
│  │                                                       │
│  ├── Distribution Plots                                 │
│  │   ├── plot_distribution()                            │
│  │   ├── plot_box_whisker()                             │
│  │   └── plot_qq()                                      │
│  │                                                       │
│  ├── Relationship Plots                                 │
│  │   ├── plot_correlation_heatmap()                     │
│  │   ├── plot_pairwise_scatter()                        │
│  │   └── plot_feature_importance_correlation()          │
│  │                                                       │
│  Input: pandas.DataFrame                                │
│  Output: plotly.graph_objects.Figure                    │
│          (Interactive, zoomable, exportable)            │
└─────────────────────────────────────────────────────────┘
```

### 4. User Interface Layer

```
┌─────────────────────────────────────────────────────────┐
│                   app.py                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Streamlit Web Application                              │
│                                                          │
│  ┌─────────────────────────────────────────────┐       │
│  │              SIDEBAR                         │       │
│  │  ┌────────────────────────────────────┐     │       │
│  │  │ 1. Dataset Type Selection          │     │       │
│  │  │    - 9 different types             │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ 2. General Parameters              │     │       │
│  │  │    - n_samples                     │     │       │
│  │  │    - noise_std                     │     │       │
│  │  │    - random_seed                   │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ 3. Function-Specific Parameters    │     │       │
│  │  │    - Dynamic based on type         │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ 4. Generate Button                 │     │       │
│  │  └────────────────────────────────────┘     │       │
│  └─────────────────────────────────────────────┘       │
│                                                          │
│  ┌─────────────────────────────────────────────┐       │
│  │              MAIN AREA                       │       │
│  │  ┌────────────────────────────────────┐     │       │
│  │  │ Tab 1: Dataset Preview             │     │       │
│  │  │   - Metrics (samples, features)    │     │       │
│  │  │   - Data table                     │     │       │
│  │  │   - Quick statistics               │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ Tab 2: Visualizations              │     │       │
│  │  │   - Scatter plots                  │     │       │
│  │  │   - 3D plots                       │     │       │
│  │  │   - Distributions                  │     │       │
│  │  │   - Correlation heatmaps           │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ Tab 3: Statistical Analysis        │     │       │
│  │  │   - Descriptive stats              │     │       │
│  │  │   - Correlation matrices           │     │       │
│  │  │   - Outlier detection              │     │       │
│  │  │   - Normality tests                │     │       │
│  │  ├────────────────────────────────────┤     │       │
│  │  │ Tab 4: Export Data                 │     │       │
│  │  │   - CSV download                   │     │       │
│  │  │   - Excel download                 │     │       │
│  │  │   - JSON download                  │     │       │
│  │  │   - Metadata display               │     │       │
│  │  └────────────────────────────────────┘     │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

### Web Interface Flow

```
┌──────┐
│ User │
└──┬───┘
   │ 1. Select parameters
   ▼
┌──────────────────┐
│   Streamlit UI   │
└──────┬───────────┘
       │ 2. Call generator
       ▼
┌─────────────────────────┐
│ SyntheticDataGenerator  │
│  - Creates DataFrame    │
│  - Adds noise           │
│  - Returns data         │
└──────┬──────────────────┘
       │ 3. Data returned
       ▼
┌──────────────────┐
│  Session State   │ ─────────┐
│  (stores data)   │          │
└──────┬───────────┘          │
       │                      │
       │ 4. User selects tab  │
       ▼                      │
┌──────────────────┐          │
│   Tab Router     │          │
└──────┬───────────┘          │
       │                      │
   ┌───┴────┬────────┬────────┘
   │        │        │
   ▼        ▼        ▼
┌─────┐ ┌─────┐ ┌─────┐
│ Viz │ │Stats│ │Export│
└──┬──┘ └──┬──┘ └──┬──┘
   │       │       │
   │ 5. Process data
   │       │       │
   ▼       ▼       ▼
┌─────────────────────┐
│   Display Results   │
│  - Plots            │
│  - Tables           │
│  - Downloads        │
└─────────────────────┘
```

### Python API Flow

```
┌──────────────┐
│  User Script │
└──────┬───────┘
       │ 1. Import modules
       ▼
┌──────────────────────────────┐
│ from data_generators import  │
│   SyntheticDataGenerator     │
└──────┬───────────────────────┘
       │ 2. Create instance
       ▼
┌──────────────────────────────┐
│ generator =                  │
│   SyntheticDataGenerator(42) │
└──────┬───────────────────────┘
       │ 3. Generate data
       ▼
┌──────────────────────────────┐
│ df = generator.generate_...()│
└──────┬───────────────────────┘
       │
   ┌───┴─────────┬──────────┐
   │             │          │
   ▼             ▼          ▼
┌──────┐   ┌──────┐   ┌──────┐
│ Viz  │   │Stats │   │Export│
│Module│   │Module│   │ .csv │
└──┬───┘   └──┬───┘   └──────┘
   │          │
   │ 4. Analyze/Visualize
   │          │
   ▼          ▼
┌────────────────────┐
│  User Analysis     │
│  - Gradient Descent│
│  - Model Training  │
│  - Experiments     │
└────────────────────┘
```

---

## Class Diagram

```
┌─────────────────────────────────────────┐
│      SyntheticDataGenerator             │
├─────────────────────────────────────────┤
│ - random_seed: Optional[int]            │
├─────────────────────────────────────────┤
│ + __init__(random_seed)                 │
│ + generate_simple_linear(...)           │
│ + generate_multiple_linear(...)         │
│ + generate_polynomial(...)              │
│ + generate_sinusoidal(...)              │
│ + generate_exponential(...)             │
│ + generate_logarithmic(...)             │
│ + generate_step_function(...)           │
│ + generate_interaction_features(...)    │
│ + generate_custom_function(...)         │
└────────────────┬────────────────────────┘
                 │
                 │ returns pandas.DataFrame
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ StatisticalAnalyzer│   │ DataVisualizer   │
├─────────────────┤   ├──────────────────┤
│ (all @staticmethod)│   │ (all @staticmethod)│
├─────────────────┤   ├──────────────────┤
│ + compute_      │   │ + plot_scatter_1d│
│   descriptive_  │   │ + plot_3d_scatter│
│   stats()       │   │ + plot_residuals │
│ + compute_      │   │ + plot_          │
│   correlation_  │   │   distribution   │
│   matrix()      │   │ + plot_          │
│ + detect_       │   │   correlation_   │
│   outliers_iqr()│   │   heatmap()      │
│ + normality_    │   │ + plot_qq()      │
│   test()        │   │ + plot_box_      │
│ + ...           │   │   whisker()      │
└─────────────────┘   │ + ...            │
                      └──────────────────┘
```

---

## File Dependencies

```
app.py
├── imports data_generators
├── imports statistics_analysis
├── imports visualizations
├── imports streamlit
├── imports plotly
└── imports pandas

data_generators.py
├── imports numpy
├── imports pandas
└── imports typing

statistics_analysis.py
├── imports numpy
├── imports pandas
├── imports scipy.stats
└── imports typing

visualizations.py
├── imports numpy
├── imports pandas
├── imports plotly.graph_objects
├── imports plotly.express
└── imports typing

example_usage.py
├── imports data_generators
├── imports statistics_analysis
├── imports visualizations
├── imports numpy
└── imports pandas

test_modules.py
├── imports data_generators
├── imports statistics_analysis
├── imports visualizations
└── imports pandas
```

---

## Technology Stack

```
┌─────────────────────────────────────────┐
│         PRESENTATION LAYER              │
│  ┌───────────────────────────────┐     │
│  │  Streamlit (Web Framework)    │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         VISUALIZATION LAYER             │
│  ┌───────────────────────────────┐     │
│  │  Plotly (Interactive Plots)   │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         BUSINESS LOGIC LAYER            │
│  ┌───────────────────────────────┐     │
│  │  Custom Modules:              │     │
│  │  - Data Generators            │     │
│  │  - Statistical Analysis       │     │
│  │  - Visualizations             │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         DATA PROCESSING LAYER           │
│  ┌───────────────┬───────────────┐     │
│  │   Pandas      │   NumPy       │     │
│  │(DataFrames)   │(Numerical)    │     │
│  └───────────────┴───────────────┘     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         SCIENTIFIC COMPUTING            │
│  ┌───────────────────────────────┐     │
│  │  SciPy (Statistics)           │     │
│  └───────────────────────────────┘     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│         OUTPUT LAYER                    │
│  ┌───────────────┬───────────────┐     │
│  │   openpyxl    │   JSON        │     │
│  │ (Excel Export)│  (Standard)   │     │
│  └───────────────┴───────────────┘     │
└─────────────────────────────────────────┘
```

---

## Extension Points

### Adding New Dataset Type

```python
# In data_generators.py

class SyntheticDataGenerator:
    # ... existing methods ...

    def generate_my_new_type(self,
                            n_samples: int = 100,
                            my_param: float = 1.0,
                            noise_std: float = 1.0) -> pd.DataFrame:
        """
        Generate my new type of data.

        Args:
            n_samples: Number of samples
            my_param: My custom parameter
            noise_std: Noise level

        Returns:
            DataFrame with generated data
        """
        # Your implementation here
        x = np.random.uniform(0, 10, n_samples)
        y_true = my_function(x, my_param)
        noise = np.random.normal(0, noise_std, n_samples)
        y = y_true + noise

        return pd.DataFrame({'x': x, 'y': y, 'y_true': y_true})
```

Then add to `app.py` in the dataset_type selectbox and create the UI for it.

### Adding New Statistical Test

```python
# In statistics_analysis.py

class StatisticalAnalyzer:
    # ... existing methods ...

    @staticmethod
    def my_new_test(df: pd.DataFrame,
                   column: str) -> Dict[str, Any]:
        """
        Perform my new statistical test.

        Args:
            df: DataFrame with data
            column: Column to test

        Returns:
            Dictionary with test results
        """
        # Your implementation
        result = perform_test(df[column])
        return {'test_statistic': result, ...}
```

### Adding New Visualization

```python
# In visualizations.py

class DataVisualizer:
    # ... existing methods ...

    @staticmethod
    def plot_my_visualization(df: pd.DataFrame,
                             column: str) -> go.Figure:
        """
        Create my new visualization.

        Args:
            df: DataFrame with data
            column: Column to visualize

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        # Your plotting code
        return fig
```

---

## Performance Considerations

### Memory Usage

```
Dataset Size    Memory Usage     Generation Time
─────────────   ─────────────    ───────────────
100 samples     < 1 MB           < 0.01s
1,000 samples   < 5 MB           < 0.05s
10,000 samples  < 50 MB          < 0.5s
100,000 samples < 500 MB         < 5s
```

### Optimization Strategies

1. **Vectorized Operations**: All computations use NumPy vectorization
2. **Lazy Loading**: Visualizations created only when needed
3. **Session State**: Streamlit caches generated data
4. **Efficient Data Structures**: pandas DataFrame for optimal memory layout

---

## Security Considerations

### Custom Function Evaluation

The `generate_custom_function()` method uses `eval()` with restricted namespace:

```python
# Safe namespace - only mathematical functions
namespace = {
    'x': x,
    'np': np,
    'sin': np.sin,
    'cos': np.cos,
    # ... only safe functions
}

# Restricted builtins - no file operations, imports, etc.
y_true = eval(func_str, {"__builtins__": {}}, namespace)
```

**Safety measures:**
- No access to built-in functions
- No file system access
- No import capability
- Only mathematical operations allowed

---

This architecture provides a clean, extensible, and educational foundation for synthetic dataset generation and analysis.
