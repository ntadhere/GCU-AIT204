"""
Streamlit app demonstrating linear regression and gradient descent.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from backend import FinancialDataGenerator, LinearRegression, GradientDescentOptimizer, train_test_split, get_split_info


# Page configuration
st.set_page_config(
    page_title="Linear Regression & Gradient Descent Demo",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Linear Regression & Gradient Descent: A Financial Analysis Demo")

st.markdown("""
This interactive application demonstrates key machine learning concepts using financial data:
- **Synthetic Data Generation** for company financials
- **Linear Regression** for predictive modeling
- **Gradient Descent** optimization with step-by-step visualization
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Data generation settings
st.sidebar.subheader("Data Settings")
n_samples = st.sidebar.slider("Number of data points", 50, 300, 100)
noise_level = st.sidebar.slider("Noise level", 0, 300000, 150000, 10000,
                                help="Higher values create more scattered data")
outlier_percentage = st.sidebar.slider("Outlier percentage (%)", 0, 20, 10, 1,
                                       help="Percentage of data points that are outliers")
heteroscedastic = st.sidebar.checkbox("Heteroscedastic noise",
                                     value=True,
                                     help="Noise increases with marketing spend (more realistic)")
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, 5) / 100
seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

# Gradient descent settings
st.sidebar.subheader("Gradient Descent Settings")
learning_rate = st.sidebar.slider("Learning rate", 0.00001, 0.001, 0.0001, 0.00001, format="%.5f")
max_iterations = st.sidebar.slider("Max iterations", 100, 2000, 500, 100)

# Generate data button
if st.sidebar.button("Generate New Data"):
    st.rerun()


# ============================================================================
# SECTION 1: What is Regression?
# ============================================================================

st.header("1️⃣ Understanding Regression Models")

st.markdown("""
### What is a Regression Model?

A **regression model** is a statistical method used to understand and predict the relationship
between variables. In simple terms, it helps us answer questions like:

- *"If we increase marketing spend, how much will revenue increase?"*
- *"How does sales volume affect our profit?"*

### Linear Regression

**Linear regression** is the simplest form of regression, which models the relationship as a straight line:

$$y = mx + b$$

Where:
- **y** is the target variable (what we want to predict, e.g., revenue)
- **x** is the feature variable (what we use to predict, e.g., marketing spend)
- **m** is the slope (rate of change)
- **b** is the intercept (baseline value when x=0)

### Why Use Regression?

1. **Prediction**: Forecast future outcomes based on historical data
2. **Understanding Relationships**: Quantify how variables affect each other
3. **Decision Making**: Make data-driven business decisions
4. **Actionable Insights**: Identify which factors have the most impact
""")


# ============================================================================
# SECTION 2: Generate and Display Data
# ============================================================================

st.header("2️⃣ Synthetic Financial Dataset")

st.markdown(f"""
We'll generate synthetic financial data for a company, showing the relationship between
**marketing spend** and **revenue**. This simulates real-world business data with:

- **Natural variation (noise)**: Real data is never perfect - there's always random variation
- **Outliers ({outlier_percentage}%)**: Unusual data points that don't follow the trend (e.g., viral campaigns, market disruptions)
- **Heteroscedastic noise**: {'Enabled - noise increases with spend (more realistic)' if heteroscedastic else 'Disabled - constant noise across all values'}

These features make the data more challenging and realistic, similar to what you'd encounter in actual business analytics.
""")

# Generate data
data_gen = FinancialDataGenerator(seed=seed)
marketing_spend, revenue = data_gen.generate_revenue_data(
    n_samples=n_samples,
    noise_std=noise_level,
    outlier_percentage=outlier_percentage,
    heteroscedastic=heteroscedastic
)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    marketing_spend,
    revenue,
    test_size=test_size,
    random_state=seed
)

# Get split information
split_info = get_split_info(X_train, X_test, y_train, y_test)

# Create dataframe
df = data_gen.create_dataframe(
    marketing_spend,
    revenue,
    "Marketing Spend ($1000s)",
    "Revenue ($)"
)

# Display data statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", split_info['total_samples'])
with col2:
    st.metric("Training Samples", split_info['train_samples'],
              help=f"{split_info['train_percentage']:.0f}% of data")
with col3:
    st.metric("Test Samples", split_info['test_samples'],
              help=f"{split_info['test_percentage']:.0f}% of data")
with col4:
    st.metric("Avg Revenue", f"${revenue.mean():,.0f}")

# Display sample data
with st.expander("View Raw Data Sample"):
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button(
        "Download Full Dataset",
        df.to_csv(index=False),
        "financial_data.csv",
        "text/csv"
    )

# Visualize raw data with train/test split
fig_scatter = go.Figure()

# Training data
fig_scatter.add_trace(go.Scatter(
    x=X_train,
    y=y_train,
    mode='markers',
    name='Training Set',
    marker=dict(size=8, opacity=0.6, color='blue')
))

# Test data
fig_scatter.add_trace(go.Scatter(
    x=X_test,
    y=y_test,
    mode='markers',
    name='Test Set',
    marker=dict(size=8, opacity=0.6, color='orange', symbol='diamond')
))

fig_scatter.update_layout(
    title="Marketing Spend vs Revenue (Train/Test Split)",
    xaxis_title="Marketing Spend (thousands $)",
    yaxis_title="Revenue ($)",
    hovermode='closest'
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(f"""
#### 📖 Understanding This Plot

**What you're seeing**:
- **Blue circles**: Training data ({split_info['train_samples']} points) - the model will learn from these
- **Orange diamonds**: Test data ({split_info['test_samples']} points) - used only for evaluation, model never sees these during training

**What to look for**:
1. **General trend**: Do higher marketing spends generally lead to higher revenues? (Should see upward pattern)
2. **Scatter/spread**: Points aren't perfectly aligned - this is the **noise** and **outliers** we added
3. **Outliers**: Look for points far from the general trend - these represent unusual events
4. **Distribution**: Are train and test points mixed throughout? (Good - means random split worked)

**Why this matters**:
- If we only evaluated on training data, we wouldn't know if the model **generalizes** to new data
- Test data simulates "future data" the model hasn't seen yet
- A model that performs well on training but poorly on test is **overfitting** (memorizing, not learning)

**Connection to math**:
- Each point is a pair $(x^{{(i)}}, y^{{(i)}})$ where $x$ is marketing spend and $y$ is revenue
- The model will try to find the best line $y = mx + b$ that fits these points
- The spread of points around the line will determine our **cost function** value
""")


# ============================================================================
# SECTION 3: Training a Model
# ============================================================================

st.header("3️⃣ Training a Regression Model")

st.markdown("""
### What Does "Training" Mean?

**Training a model** means finding the best parameters (slope and intercept) that fit our data.
The process involves:

1. **Split the Data**: Divide into training set (for learning) and test set (for evaluation)
2. **Define the Model**: Choose a model structure (linear: y = mx + b)
3. **Define a Cost Function**: Measure how well our model fits the data
   - Common choice: **Mean Squared Error (MSE)** = average of squared differences between predictions and actual values
4. **Optimize Parameters**: Adjust m and b to minimize the cost function on the training set
5. **Evaluate Performance**: Check how well the model performs on both training and test sets

### Why Train/Test Split?

**Train/Test Split** is crucial for assessing model generalization:

- **Training Set**: Used to learn the model parameters (fit the line)
- **Test Set**: Used to evaluate how well the model performs on unseen data
- **Goal**: Ensure the model generalizes well and doesn't just memorize the training data (overfitting)

If training and test performance are similar, the model generalizes well! If test performance is much worse, the model may be overfitting.

### Two Approaches to Training

#### Method 1: Closed-Form Solution (Analytical)
- Uses mathematical formulas to directly calculate optimal parameters
- Fast and exact for simple linear regression
- Formula: $m = \\frac{\\sum(x - \\bar{x})(y - \\bar{y})}{\\sum(x - \\bar{x})^2}$

#### Method 2: Gradient Descent (Iterative)
- Starts with random parameters
- Iteratively improves them by following the gradient
- More flexible, works for complex models
- We'll explore this in detail below!
""")

# Train linear regression model
st.subheader("Training with Closed-Form Solution")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
params = lr_model.get_parameters()

# Calculate metrics for both training and test sets
train_metrics = lr_model.calculate_metrics(X_train, y_train)
test_metrics = lr_model.calculate_metrics(X_test, y_test)

# Display parameters
col1, col2 = st.columns(2)
with col1:
    st.metric("Slope (m)", f"{params['slope']:.2f}",
              help="Revenue increases by this amount for each $1000 increase in marketing spend")
with col2:
    st.metric("Intercept (b)", f"${params['intercept']:,.0f}",
              help="Expected revenue when marketing spend is zero")

st.markdown(f"""
### Model Equation

Our trained model is:

$$\\text{{Revenue}} = {params['slope']:.2f} \\times \\text{{Marketing Spend}} + {params['intercept']:,.0f}$$

**Interpretation**: For every $1,000 increase in marketing spend, revenue increases by ${params['slope']:.2f}.
""")

# Display metrics
st.subheader("Model Performance Metrics")

st.markdown("""
**Training vs Test Performance**: Comparing metrics on both sets helps us understand if the model
generalizes well or if it's overfitting to the training data.
""")

# Create two columns for train and test metrics
col_train, col_test = st.columns(2)

with col_train:
    st.markdown("### 📘 Training Set")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        st.metric("R² Score", f"{train_metrics['r2']:.4f}")
        st.metric("MSE", f"${train_metrics['mse']:,.0f}")
    with subcol2:
        st.metric("RMSE", f"${train_metrics['rmse']:,.0f}")
        st.metric("MAE", f"${train_metrics['mae']:,.0f}")

with col_test:
    st.markdown("### 📙 Test Set")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        r2_delta = test_metrics['r2'] - train_metrics['r2']
        st.metric("R² Score", f"{test_metrics['r2']:.4f}",
                 delta=f"{r2_delta:.4f}", delta_color="normal")
        mse_delta = test_metrics['mse'] - train_metrics['mse']
        st.metric("MSE", f"${test_metrics['mse']:,.0f}",
                 delta=f"${mse_delta:,.0f}", delta_color="inverse")
    with subcol2:
        rmse_delta = test_metrics['rmse'] - train_metrics['rmse']
        st.metric("RMSE", f"${test_metrics['rmse']:,.0f}",
                 delta=f"${rmse_delta:,.0f}", delta_color="inverse")
        mae_delta = test_metrics['mae'] - train_metrics['mae']
        st.metric("MAE", f"${test_metrics['mae']:,.0f}",
                 delta=f"${mae_delta:,.0f}", delta_color="inverse")

st.markdown(f"""
**Interpretation**:
- **R² Score** closer to 1.0 indicates better fit
- Similar train/test scores suggest good generalization (not overfitting)
- Test score slightly lower than training is normal and expected

**Impact of Data Quality**:
- With **{outlier_percentage}% outliers** and **noise level {noise_level:,.0f}**, the model must be robust to imperfect data
- Outliers pull the regression line away from the main trend, reducing R² score
- Higher noise reduces prediction accuracy but tests the model's robustness
- {'Heteroscedastic noise (varying noise) is more realistic and challenging than constant noise' if heteroscedastic else 'Constant noise is simpler but less realistic than heteroscedastic noise'}
""")

# Add detailed R-squared explanation
with st.expander("📊 Understanding R² Score in Detail"):
    st.markdown(f"""
    ### What is R² (R-Squared)?

    **R² (Coefficient of Determination)** measures how well your model explains the variation in the data.
    Think of it as answering: *"How much of the ups and downs in revenue can be explained by marketing spend?"*

    ### The Mathematics

    R² compares your model's predictions to a simple baseline (the mean):

    $$R^2 = 1 - \\frac{{SS_{{res}}}}{{SS_{{tot}}}}$$

    Where:
    - **SS_res** (Residual Sum of Squares) = $\\sum(y_{{actual}} - y_{{predicted}})^2$
      - How much error your model makes
    - **SS_tot** (Total Sum of Squares) = $\\sum(y_{{actual}} - \\bar{{y}})^2$
      - How much variation exists in the data (compared to the mean)

    ### Interpretation Guide

    | R² Value | Meaning | Example |
    |----------|---------|---------|
    | **1.0** | Perfect fit - model explains 100% of variation | All points exactly on the line (rare in real data) |
    | **0.9** | Excellent - explains 90% of variation | Strong relationship, highly predictive |
    | **0.7-0.8** | Good - explains 70-80% of variation | Useful for predictions, some noise |
    | **0.5** | Moderate - explains 50% of variation | Relationship exists but lots of noise |
    | **0.3** | Weak - explains only 30% of variation | Other factors are more important |
    | **0.0** | No relationship - no better than guessing the mean | Model is useless |
    | **< 0.0** | Negative - worse than guessing the mean | Model is harmful! |

    ### Your Current Model

    - **Training R²**: {train_metrics['r2']:.4f} → Model explains **{train_metrics['r2']*100:.1f}%** of revenue variation in training data
    - **Test R²**: {test_metrics['r2']:.4f} → Model explains **{test_metrics['r2']*100:.1f}%** of revenue variation in test data

    ### What R² Tells You

    1. **Predictive Power**: Higher R² means better predictions
    2. **Explained vs Unexplained**:
       - **Explained**: {test_metrics['r2']*100:.1f}% of revenue changes are explained by marketing spend
       - **Unexplained**: {(1-test_metrics['r2'])*100:.1f}% is due to other factors (seasonality, competitors, economy, etc.)
    3. **Model Usefulness**:
       - R² > 0.7: Strong relationship, highly actionable
       - R² = 0.5-0.7: Moderate relationship, useful but consider other factors
       - R² < 0.5: Weak relationship, marketing spend alone isn't enough to predict revenue

    ### Why R² Might Be Lower Than Expected

    - **Outliers** ({outlier_percentage}%): Unusual points reduce R² significantly
    - **High noise** ({noise_level:,.0f}): More randomness = lower predictability
    - **Missing variables**: Other factors (brand, competition, seasonality) aren't in the model
    - **Non-linear relationships**: Linear regression might not capture the true pattern

    ### Key Insight

    R² answers: *"If I only knew marketing spend, how well could I predict revenue?"*
    Your answer: **{test_metrics['r2']*100:.1f}%** - the rest depends on factors we haven't measured.
    """)


# Visualize predictions
train_predictions = lr_model.predict(X_train)
test_predictions = lr_model.predict(X_test)

# Create a line across the full range
x_range = np.linspace(marketing_spend.min(), marketing_spend.max(), 100)
y_range = lr_model.predict(x_range)

fig_regression = go.Figure()

# Training data
fig_regression.add_trace(go.Scatter(
    x=X_train,
    y=y_train,
    mode='markers',
    name='Training Data',
    marker=dict(size=8, opacity=0.6, color='blue')
))

# Test data
fig_regression.add_trace(go.Scatter(
    x=X_test,
    y=y_test,
    mode='markers',
    name='Test Data',
    marker=dict(size=8, opacity=0.6, color='orange', symbol='diamond')
))

# Regression line
fig_regression.add_trace(go.Scatter(
    x=x_range,
    y=y_range,
    mode='lines',
    name='Regression Line',
    line=dict(color='red', width=3)
))

fig_regression.update_layout(
    title="Linear Regression Fit (Trained on Training Set)",
    xaxis_title="Marketing Spend (thousands $)",
    yaxis_title="Revenue ($)",
    hovermode='closest'
)
st.plotly_chart(fig_regression, use_container_width=True)

st.markdown(f"""
#### 📖 Understanding This Plot

**What you're seeing**:
- **Red line**: The trained model's prediction function $y = {params['slope']:.2f}x + {params['intercept']:,.0f}$
- **Blue circles**: Training data (model learned from these)
- **Orange diamonds**: Test data (model is being evaluated on these)
- **Vertical distances**: The gap between each point and the red line is the **prediction error** (residual)

**What to look for**:
1. **Line positioning**: Does the line go through the "middle" of the data cloud?
2. **Prediction errors**:
   - Points above the line: Model **under-predicted** revenue (actual > predicted)
   - Points below the line: Model **over-predicted** revenue (actual < predicted)
3. **Error distribution**: Are errors roughly equal above and below? (Good sign - no systematic bias)
4. **Outliers impact**: See how outliers pull the line slightly away from the main trend

**Why this matters**:
- This line is our **trained model** - the result of the closed-form solution
- The line minimizes the **sum of squared errors** (sum of all vertical distances squared)
- Notice the line works for **both** training (blue) and test (orange) data - this shows **generalization**

**Connection to math**:
- The line equation is $y = mx + b$ where $m = {params['slope']:.2f}$ and $b = {params['intercept']:,.0f}$
- For any point $(x^{{(i)}}, y^{{(i)}})$, the prediction is $\\hat{{y}}^{{(i)}} = {params['slope']:.2f} \\times x^{{(i)}} + {params['intercept']:,.0f}$
- The error (residual) is $e^{{(i)}} = y^{{(i)}} - \\hat{{y}}^{{(i)}}$ (the vertical distance)
- The model was trained to minimize $\\sum (e^{{(i)}})^2$ - the sum of all squared errors
""")



# ============================================================================
# SECTION 4: Gradient Descent
# ============================================================================

st.header("4️⃣ Gradient Descent: The Optimization Algorithm")

st.markdown("""
### What is Gradient Descent?

**Gradient descent** is an iterative optimization algorithm that finds the minimum of a function.
Think of it like finding the bottom of a valley while blindfolded:

1. **Start at a random location** (random parameters)
2. **Feel the slope** (compute gradient)
3. **Take a step downhill** (update parameters)
4. **Repeat** until you reach the bottom (minimum cost)

### The Training Process: How Parameters Learn

**Training** means adjusting parameters (m and b) to minimize prediction error. Here's the step-by-step process:

#### Step 1: Define the Cost Function

For linear regression, we want to minimize the **Mean Squared Error (MSE)**:

$$\\text{Cost}(m, b) = \\frac{1}{2n} \\sum_{i=1}^{n} (y_{pred}^{(i)} - y^{(i)})^2$$

Where:
- $y_{pred}^{(i)} = m \\cdot x^{(i)} + b$ (our model's prediction)
- $y^{(i)}$ is the actual value
- $n$ is the number of training samples
- The $\\frac{1}{2}$ is for mathematical convenience (cancels out during differentiation)

**What this means**: The cost function measures how wrong our predictions are. Lower cost = better model!

#### Step 2: Compute Partial Derivatives (The Gradient)

**Partial derivatives** measure how much the cost changes when we adjust each parameter slightly.

Think of it like this:
- You're on a hillside (the cost surface)
- You want to know: "If I move north (change m), does the hill go up or down?"
- And: "If I move east (change b), does the hill go up or down?"
- The partial derivatives answer these questions!

**Partial derivative with respect to slope (m)**:

$$\\frac{\\partial \\text{Cost}}{\\partial m} = \\frac{1}{n} \\sum_{i=1}^{n} (y_{pred}^{(i)} - y^{(i)}) \\cdot x^{(i)}$$

**What this means**:
- Measures how sensitive the cost is to changes in slope
- Positive value → increasing m increases cost (m is too high)
- Negative value → increasing m decreases cost (m is too low)
- Magnitude tells us how steep the slope is

**Partial derivative with respect to intercept (b)**:

$$\\frac{\\partial \\text{Cost}}{\\partial b} = \\frac{1}{n} \\sum_{i=1}^{n} (y_{pred}^{(i)} - y^{(i)})$$

**What this means**:
- Measures how sensitive the cost is to changes in intercept
- Positive value → increasing b increases cost (b is too high)
- Negative value → increasing b decreases cost (b is too low)

#### Step 3: The Gradient Vector

The **gradient** is a vector combining both partial derivatives:

$$\\nabla \\text{Cost} = \\begin{bmatrix} \\frac{\\partial \\text{Cost}}{\\partial m} \\\\ \\frac{\\partial \\text{Cost}}{\\partial b} \\end{bmatrix}$$

**What this means**:
- The gradient points in the direction of **steepest increase** in cost
- We want to go **opposite** the gradient (downhill) to minimize cost
- Think: The gradient is a compass pointing uphill; we walk the opposite direction

#### Step 4: Update Parameters (Take a Step)

We update parameters by moving in the opposite direction of the gradient:

$$m_{new} = m_{old} - \\alpha \\cdot \\frac{\\partial \\text{Cost}}{\\partial m}$$

$$b_{new} = b_{old} - \\alpha \\cdot \\frac{\\partial \\text{Cost}}{\\partial b}$$

**What this means**:
- We subtract the gradient (moving downhill)
- $\\alpha$ (learning rate) controls how big our steps are
- If gradient is large (steep slope), we take bigger steps
- If gradient is small (gentle slope), we take smaller steps
- The negative sign ensures we move **downhill** (toward minimum cost)

### The Complete Training Loop

Here's how gradient descent trains the model:

1. **Initialize**: Start with random m and b (or zeros)
2. **Forward Pass**: Use current m and b to make predictions
3. **Compute Cost**: Calculate how wrong the predictions are
4. **Compute Gradients**: Calculate partial derivatives (how to adjust m and b)
5. **Update Parameters**: Adjust m and b in the direction that reduces cost
6. **Repeat**: Go back to step 2 until cost stops decreasing

### Key Hyperparameters

- **Learning Rate (α)**: Controls step size
  - Too small → slow convergence (takes many iterations)
  - Too large → may overshoot the minimum or diverge (cost increases!)
  - Just right → efficient convergence to the minimum
- **Iterations**: Number of steps to take
  - Too few → doesn't reach minimum (underfitting)
  - Too many → wastes computation (but doesn't hurt accuracy)

### Why This Works: The Intuition

Imagine you're skiing down a mountain in thick fog (can't see far):
- **Gradient** = the slope under your skis (tells you which direction is downhill)
- **Learning rate** = how fast you ski
- **Iteration** = each turn or adjustment you make
- **Cost** = your altitude (you want to reach the valley floor)

You feel the slope (gradient), ski downhill a bit (update parameters), feel the new slope, ski more, repeat until you're at the bottom (minimum cost). That's gradient descent!
""")

# Add a detailed expandable section
with st.expander("🔬 Deep Dive: Why These Specific Formulas?"):
    st.markdown("""
    ### Where Do the Partial Derivatives Come From?

    Let's derive them step by step using calculus:

    **Starting with the cost function**:
    $$\\text{Cost}(m, b) = \\frac{1}{2n} \\sum_{i=1}^{n} (y_{pred}^{(i)} - y^{(i)})^2$$

    **Substitute the prediction formula** $y_{pred}^{(i)} = m \\cdot x^{(i)} + b$:
    $$\\text{Cost}(m, b) = \\frac{1}{2n} \\sum_{i=1}^{n} (m \\cdot x^{(i)} + b - y^{(i)})^2$$

    **To find** $\\frac{\\partial \\text{Cost}}{\\partial m}$:
    1. Apply chain rule: derivative of $(something)^2$ is $2 \\cdot (something) \\cdot (derivative of something)$
    2. The "something" is $(m \\cdot x^{(i)} + b - y^{(i)})$
    3. Derivative of this with respect to m is just $x^{(i)}$ (since b and y are constants)
    4. The $\\frac{1}{2}$ cancels with the 2 from the chain rule
    5. Result: $\\frac{1}{n} \\sum (m \\cdot x^{(i)} + b - y^{(i)}) \\cdot x^{(i)}$

    **To find** $\\frac{\\partial \\text{Cost}}{\\partial b}$:
    1. Same process, but derivative of $(m \\cdot x^{(i)} + b - y^{(i)})$ with respect to b is just 1
    2. Result: $\\frac{1}{n} \\sum (m \\cdot x^{(i)} + b - y^{(i)})$

    ### Why the Negative Sign in Updates?

    The gradient $\\frac{\\partial \\text{Cost}}{\\partial m}$ tells us:
    - **Positive gradient**: Cost increases as m increases → we should **decrease** m
    - **Negative gradient**: Cost decreases as m increases → we should **increase** m

    The negative sign in $m_{new} = m_{old} - \\alpha \\cdot \\frac{\\partial \\text{Cost}}{\\partial m}$ automatically does the right thing:
    - If gradient is positive, we subtract it (move left, decrease m)
    - If gradient is negative, subtracting a negative means addition (move right, increase m)

    ### Connection to Training

    **"Training a model"** specifically means:
    1. **Define what "good" means**: Choose a cost function
    2. **Measure current performance**: Compute the cost
    3. **Figure out how to improve**: Calculate gradients (partial derivatives)
    4. **Make an improvement**: Update parameters using gradients
    5. **Repeat**: Until the model can't improve anymore

    The gradients are the **key to learning** - they tell the model exactly how to adjust each parameter to improve!
    """)



# ============================================================================
# SECTION 5: Gradient Descent Implementation
# ============================================================================

st.header("5️⃣ Gradient Descent in Action")

st.markdown(f"""
Let's train the same model using gradient descent and visualize the optimization process!

**Current Settings:**
- Learning Rate: `{learning_rate}`
- Max Iterations: `{max_iterations}`
""")

# Train with gradient descent
gd_optimizer = GradientDescentOptimizer(
    learning_rate=learning_rate,
    max_iterations=max_iterations
)

with st.spinner("Running gradient descent..."):
    history = gd_optimizer.fit(X_train, y_train, initial_slope=0.0, initial_intercept=0.0)
    gd_params = gd_optimizer.get_parameters()

# Display final parameters
st.subheader("Gradient Descent Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Final Slope", f"{gd_params['slope']:.2f}")
with col2:
    st.metric("Final Intercept", f"${gd_params['intercept']:,.0f}")
with col3:
    final_cost = history['cost'][-1]
    st.metric("Final Cost", f"{final_cost:,.0f}")

st.markdown(f"""
### Comparison with Closed-Form Solution

| Method | Slope | Intercept |
|--------|-------|-----------|
| Closed-Form | {params['slope']:.2f} | ${params['intercept']:,.0f} |
| Gradient Descent | {gd_params['slope']:.2f} | ${gd_params['intercept']:,.0f} |

The two methods should produce similar results! Small differences are due to the iterative nature of gradient descent.
""")


# ============================================================================
# SECTION 6: Visualizations
# ============================================================================

st.header("6️⃣ Optimization Visualizations")

# Cost over iterations
st.subheader("Cost Function Convergence")

fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(
    x=history['iteration'],
    y=history['cost'],
    mode='lines+markers',
    name='Cost',
    line=dict(color='blue', width=2)
))
fig_cost.update_layout(
    title="Cost Function Over Iterations",
    xaxis_title="Iteration",
    yaxis_title="Cost (MSE)",
    hovermode='x'
)
st.plotly_chart(fig_cost, use_container_width=True)

st.markdown(f"""
#### 📖 Understanding This Plot

**What you're seeing**:
- **X-axis**: Iteration number (each step of gradient descent)
- **Y-axis**: Cost function value (Mean Squared Error / 2)
- **Blue line**: How the cost changes as the model learns

**What to look for**:
1. **Initial cost**: The starting point (iteration 0) - this is how bad the model is before training
2. **Downward trend**: Cost should **decrease** - this means the model is learning!
3. **Convergence**: The curve flattens at the end - model has found the optimal parameters
4. **Smoothness**: A smooth curve means stable learning (good learning rate)

**What this tells you about gradient descent**:
- **It's working**: If cost decreases, gradient descent is successfully optimizing
- **Learning speed**: Steep drop = fast learning, gradual = slow learning
- **Convergence**: When the curve flattens, gradients are near zero (we're at the minimum)
- Final cost: **{history['cost'][-1]:,.0f}** - this is as good as the model can get with these parameters

**Potential issues to watch for**:
- **Cost increasing**: Learning rate too high, model is overshooting the minimum
- **Oscillating wildly**: Learning rate way too high, model bouncing around
- **Not decreasing**: Stuck at a bad starting point (rare for convex problems like linear regression)
- **Decreasing very slowly**: Learning rate too small, need more iterations

**Connection to math**:
- Each point on this curve is $\\text{{Cost}}(m_t, b_t) = \\frac{{1}}{{2n}} \\sum (\\hat{{y}}^{{(i)}} - y^{{(i)}})^2$ at iteration $t$
- Gradient descent is minimizing this function by following: $\\theta_{{t+1}} = \\theta_t - \\alpha \\nabla \\text{{Cost}}$
- The downward slope of this curve = the negative gradient direction we're following
- When this curve flattens, $\\nabla \\text{{Cost}} \\approx 0$ (we've reached the minimum)
""")


# Parameters evolution
st.subheader("Parameter Evolution")

fig_params = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Slope Evolution", "Intercept Evolution")
)

fig_params.add_trace(
    go.Scatter(x=history['iteration'], y=history['slope'],
               mode='lines', name='Slope', line=dict(color='green')),
    row=1, col=1
)
fig_params.add_trace(
    go.Scatter(x=history['iteration'], y=history['intercept'],
               mode='lines', name='Intercept', line=dict(color='orange')),
    row=1, col=2
)

fig_params.update_xaxes(title_text="Iteration", row=1, col=1)
fig_params.update_xaxes(title_text="Iteration", row=1, col=2)
fig_params.update_yaxes(title_text="Slope Value", row=1, col=1)
fig_params.update_yaxes(title_text="Intercept Value", row=1, col=2)
fig_params.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_params, use_container_width=True)

st.markdown(f"""
#### 📖 Understanding These Plots

**What you're seeing**:
- **Left plot**: How the slope parameter $m$ evolves during training
- **Right plot**: How the intercept parameter $b$ evolves during training
- **Starting point**: Both begin at 0 (our initialization)
- **Ending point**: Slope = {gd_params['slope']:.2f}, Intercept = {gd_params['intercept']:,.0f}

**What to look for**:
1. **Initial movement**: Large changes early on (steep gradients at the start)
2. **Gradual convergence**: Changes get smaller as we approach optimal values
3. **Stabilization**: Both curves flatten when optimal parameters are found
4. **Different scales**: Slope and intercept may converge at different rates

**What this tells you about learning**:
- **The learning journey**: You can see exactly how the model "learns" - adjusting parameters step by step
- **Update rule in action**: Each step follows $m_{{new}} = m_{{old}} - \\alpha \\frac{{\\partial \\text{{Cost}}}}{{\\partial m}}$
- **Gradient guidance**: The curves show how gradients guide the parameters toward optimal values
- **Convergence speed**: Faster convergence (steeper curves) means larger gradients or higher learning rate

**Connection to the regression line**:
- At iteration 0: Line is $y = 0 \\times x + 0 = 0$ (horizontal line at y=0 - terrible!)
- At iteration {len(history['iteration'])//2}: Line is closer to the data
- Final iteration: Line is $y = {gd_params['slope']:.2f}x + {gd_params['intercept']:,.0f}$ (optimal fit)

**Connection to math**:
- These curves visualize the update equations:
  - $m_{{t+1}} = m_t - \\alpha \\frac{{\\partial \\text{{Cost}}}}{{\\partial m}}$
  - $b_{{t+1}} = b_t - \\alpha \\frac{{\\partial \\text{{Cost}}}}{{\\partial b}}$
- The slope of these curves = the learning rate × gradient magnitude
- When curves flatten, gradients $\\approx 0$ (partial derivatives near zero at the minimum)
""")


# Gradient magnitudes
st.subheader("Gradient Magnitudes")

fig_gradients = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Gradient of Slope", "Gradient of Intercept")
)

fig_gradients.add_trace(
    go.Scatter(x=history['iteration'][:-1], y=history['gradient_slope'][:-1],
               mode='lines', name='∂Cost/∂m', line=dict(color='red')),
    row=1, col=1
)
fig_gradients.add_trace(
    go.Scatter(x=history['iteration'][:-1], y=history['gradient_intercept'][:-1],
               mode='lines', name='∂Cost/∂b', line=dict(color='purple')),
    row=1, col=2
)

fig_gradients.update_xaxes(title_text="Iteration", row=1, col=1)
fig_gradients.update_xaxes(title_text="Iteration", row=1, col=2)
fig_gradients.update_yaxes(title_text="Gradient Value", row=1, col=1)
fig_gradients.update_yaxes(title_text="Gradient Value", row=1, col=2)
fig_gradients.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_gradients, use_container_width=True)

st.markdown("""
#### 📖 Understanding These Plots

**What you're seeing**:
- **Left plot**: $\\frac{\\partial \\text{Cost}}{\\partial m}$ - the gradient of the cost function with respect to slope
- **Right plot**: $\\frac{\\partial \\text{Cost}}{\\partial b}$ - the gradient of the cost function with respect to intercept
- **Y-axis values**: The magnitude of each gradient (how steep the cost surface is)

**What to look for**:
1. **Initial magnitude**: Gradients start large (steep cost surface - far from minimum)
2. **Decreasing trend**: Gradients should approach zero as we get closer to the minimum
3. **Sign (positive/negative)**: Tells us which direction to move the parameter
4. **Oscillations**: Some wiggling is normal, but wild swings mean learning rate is too high

**What gradients represent**:
- **Slope gradient** ($\\frac{\\partial \\text{Cost}}{\\partial m}$):
  - "If I increase slope $m$ by a tiny bit, how much does cost change?"
  - Positive → cost increases if we increase $m$ → we should **decrease** $m$
  - Negative → cost decreases if we increase $m$ → we should **increase** $m$
  - Magnitude → how strongly we should adjust $m$

- **Intercept gradient** ($\\frac{\\partial \\text{Cost}}{\\partial b}$):
  - "If I increase intercept $b$ by a tiny bit, how much does cost change?"
  - Same interpretation as slope gradient

**Why they approach zero**:
- At the minimum of a function, the slope (derivative) is **zero**
- As we approach optimal parameters, the cost surface flattens
- When gradients ≈ 0, we've found the bottom (minimum cost)
- This is the stopping criterion: "stop when gradients are near zero"

**Connection to parameter updates**:
- These gradients **directly determine** how parameters change:
  - $m_{new} = m_{old} - \\alpha \\times$ [left plot value]
  - $b_{new} = b_{old} - \\alpha \\times$ [right plot value]
- Large gradient → big parameter update
- Small gradient → tiny parameter update
- Zero gradient → no update needed (we're done!)

**Real-world analogy**:
- Imagine you're feeling your way down a hill in the dark
- The gradient is like the **steepness** under your feet
- Large gradient = steep slope → take big steps downhill
- Small gradient = gentle slope → take small steps
- Zero gradient = flat ground → you've reached the valley floor, stop!
""")


# 3D Surface Plot (Cost Surface)
st.subheader("Cost Function Surface (3D Visualization)")

st.markdown("""
This 3D plot shows the cost function as a surface in parameter space. The gradient descent
path is overlaid to show how the algorithm navigates toward the minimum.

#### 📖 Understanding This 3D Plot

**What you're seeing**:
- **Colored surface**: The cost function $\\text{Cost}(m, b)$ - every point on this surface represents a different model
- **X-axis**: Slope parameter $m$
- **Y-axis**: Intercept parameter $b$
- **Z-axis (height)**: Cost value (lower = better model)
- **Red path**: The journey gradient descent takes from start to optimal parameters

**How to interpret the surface**:
1. **Bowl/valley shape**: For linear regression, the cost surface is **convex** (bowl-shaped, one minimum)
2. **Lowest point**: The bottom of the bowl is the optimal parameters (minimum cost)
3. **Height = cost**: Higher regions = worse models, lower regions = better models
4. **Colors**: Typically dark blue = low cost (good), yellow/green = high cost (bad)

**The gradient descent path (red line)**:
- **Starting point**: Where we initialized (often m=0, b=0) - high up on the surface
- **Direction**: Always moving "downhill" - perpendicular to contour lines
- **Steps**: Each marker is one iteration - see how step size changes
- **Ending point**: At the bottom of the valley - optimal parameters

**What gradient descent is doing**:
- At each red point, gradient descent:
  1. Computes the **gradient** (direction of steepest ascent)
  2. Moves in the **opposite direction** (downhill)
  3. Takes a step of size controlled by learning rate $\\alpha$
- The path shows **exactly** which parameters were tried during training

**Connection to math**:
- **The surface**: Every point $(m, b, z)$ where $z = \\frac{1}{2n} \\sum (mx^{(i)} + b - y^{(i)})^2$
- **The gradient**: At any point, the gradient $\\nabla \\text{Cost} = \\begin{bmatrix} \\frac{\\partial \\text{Cost}}{\\partial m} \\\\ \\frac{\\partial \\text{Cost}}{\\partial b} \\end{bmatrix}$ points straight uphill
- **The path**: Follows $\\begin{bmatrix} m \\\\ b \\end{bmatrix}_{t+1} = \\begin{bmatrix} m \\\\ b \\end{bmatrix}_t - \\alpha \\nabla \\text{Cost}$

**Interactive exploration**:
- **Rotate**: Click and drag to rotate the view - see the bowl from different angles
- **Zoom**: Scroll to zoom in/out - examine the path more closely
- **Look from above**: See how the path spirals toward the center
- **Look from side**: See how the path descends from high cost to low cost
""")

# Create a grid for the cost surface
slope_range = np.linspace(
    min(history['slope']) - 20,
    max(history['slope']) + 20,
    50
)
intercept_range = np.linspace(
    min(history['intercept']) - 100000,
    max(history['intercept']) + 100000,
    50
)

slope_grid, intercept_grid = np.meshgrid(slope_range, intercept_range)
cost_grid = np.zeros_like(slope_grid)

# Compute cost for each point in the grid (using training data)
for i in range(slope_grid.shape[0]):
    for j in range(slope_grid.shape[1]):
        predictions_temp = slope_grid[i, j] * X_train + intercept_grid[i, j]
        cost_grid[i, j] = (1 / (2 * len(y_train))) * np.sum((predictions_temp - y_train) ** 2)

# Create 3D surface plot
fig_3d = go.Figure()

# Add surface
fig_3d.add_trace(go.Surface(
    x=slope_range,
    y=intercept_range,
    z=cost_grid,
    colorscale='Viridis',
    opacity=0.8,
    name='Cost Surface'
))

# Add gradient descent path
fig_3d.add_trace(go.Scatter3d(
    x=history['slope'],
    y=history['intercept'],
    z=history['cost'],
    mode='markers+lines',
    marker=dict(size=4, color='red'),
    line=dict(color='red', width=4),
    name='GD Path'
))

fig_3d.update_layout(
    title="Cost Function Surface with Gradient Descent Path",
    scene=dict(
        xaxis_title="Slope",
        yaxis_title="Intercept",
        zaxis_title="Cost",
    ),
    height=700
)

st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("""
**Key observations from the path**:
- **Starts high, ends low**: Cost decreases as we follow the path down
- **Perpendicular to contours**: Gradient always points perpendicular to level curves
- **Converges to center**: The path leads directly to the bowl's minimum
- **Step size varies**: Steps are larger when gradients are large, smaller near the minimum

**Why this visualization matters**:
This is the **only** plot where you can see the entire cost landscape! Other plots show slices or projections,
but this shows the full 2D parameter space and how gradient descent efficiently navigates it to find the optimum.
""")

# Animation slider for iterations
st.subheader("Step-by-Step Animation")

st.markdown("""
Use the slider below to **manually step through** the gradient descent training process.
Watch how the regression line evolves from a terrible fit to the optimal fit!

**How to use this**:
- Move the slider from left (iteration 0) to right (final iteration)
- Watch the red line adjust to fit the data better
- Pay attention to the cost value decreasing
- Notice how parameters m and b change

This shows **exactly what gradient descent is doing** - incrementally improving the model one step at a time.
""")

iteration_select = st.slider(
    "Select Iteration",
    0,
    len(history['iteration']) - 1,
    0,
    help="Move the slider to see how the regression line changes during training"
)

selected_slope = history['slope'][iteration_select]
selected_intercept = history['intercept'][iteration_select]
selected_cost = history['cost'][iteration_select]

st.markdown(f"""
**Iteration {history['iteration'][iteration_select]}:**
- Slope: `{selected_slope:.2f}`
- Intercept: `${selected_intercept:,.0f}`
- Cost: `{selected_cost:,.0f}`
""")

# Plot regression line at selected iteration
x_range_anim = np.linspace(marketing_spend.min(), marketing_spend.max(), 100)
predictions_iter = selected_slope * x_range_anim + selected_intercept

fig_animation = go.Figure()

# Training data
fig_animation.add_trace(go.Scatter(
    x=X_train,
    y=y_train,
    mode='markers',
    name='Training Data',
    marker=dict(size=8, opacity=0.6, color='blue')
))

# Test data
fig_animation.add_trace(go.Scatter(
    x=X_test,
    y=y_test,
    mode='markers',
    name='Test Data',
    marker=dict(size=8, opacity=0.6, color='orange', symbol='diamond')
))

# Regression line at this iteration
fig_animation.add_trace(go.Scatter(
    x=x_range_anim,
    y=predictions_iter,
    mode='lines',
    name=f'Fit at Iteration {iteration_select}',
    line=dict(color='red', width=3)
))

fig_animation.update_layout(
    title=f"Regression Line at Iteration {iteration_select} (Training Set)",
    xaxis_title="Marketing Spend (thousands $)",
    yaxis_title="Revenue ($)",
    hovermode='closest'
)
st.plotly_chart(fig_animation, use_container_width=True)

st.markdown(f"""
#### 📖 Understanding This Animation

**What you're seeing at iteration {iteration_select}**:
- **Red line**: The model at this specific iteration: $y = {selected_slope:.2f}x + {selected_intercept:,.0f}$
- **Cost at this iteration**: {selected_cost:,.0f}
- **Progress**: This is step {iteration_select} out of {len(history['iteration'])-1} total iterations

**Try this experiment**:
1. **Start at iteration 0**:
   - Line is $y = 0x + 0$ (flat at zero - completely wrong!)
   - Cost is very high (model makes terrible predictions)

2. **Move to iteration 10-20**:
   - Line starts to tilt and shift
   - Cost is dropping rapidly
   - Model is learning quickly

3. **Move to middle iterations** (around {len(history['iteration'])//2}):
   - Line looks pretty good
   - Cost is much lower
   - Model is close to optimal

4. **Move to final iteration** ({len(history['iteration'])-1}):
   - Line fits the data well
   - Cost is at minimum
   - Model has converged

**What this teaches you**:
- **Learning is incremental**: The model doesn't instantly know the answer - it improves step by step
- **Gradients guide learning**: At each step, gradients tell the model how to adjust the line
- **Convergence happens**: Eventually, the line stops changing much (converged)
- **Visual feedback**: You can SEE the model learning - this is what "training" looks like!

**Connection to the other plots**:
- **Cost plot**: The cost at this iteration is one point on the cost convergence graph
- **Parameter plots**: The m and b values here are points on the parameter evolution graphs
- **Gradient plots**: The gradients at this iteration determine how the line will change next
- **3D plot**: This iteration is one red marker on the 3D cost surface

This animation brings everything together - you can see HOW the model learns, not just THAT it learns!
""")



# ============================================================================
# SECTION 7: Actionable Insights
# ============================================================================

st.header("7️⃣ Actionable Business Insights")

st.markdown(f"""
### Key Findings

Based on our trained model, here are actionable insights:

1. **ROI on Marketing Spend**:
   - For every **$1,000** invested in marketing, we expect approximately **${params['slope']:.2f}** in additional revenue
   - This represents an ROI of **{(params['slope']/1000)*100:.1f}%**

2. **Base Revenue**:
   - Even with zero marketing spend, the model predicts a base revenue of **${params['intercept']:,.0f}**
   - This could represent organic sales or brand recognition

3. **Model Reliability** (on test set):
   - The model explains **{test_metrics['r2']*100:.1f}%** of the variance in revenue (R² = {test_metrics['r2']:.4f})
   - Average prediction error is **${test_metrics['mae']:,.0f}** (MAE)
   - RMSE: **${test_metrics['rmse']:,.0f}** indicates typical prediction error

### Recommendations

- **Optimize Marketing Budget**: Use the model to find the optimal marketing spend level
- **Forecast Revenue**: Predict future revenue based on planned marketing campaigns
- **Track Performance**: Monitor if actual results match predictions; deviations may indicate market changes
- **A/B Testing**: Test different marketing strategies and measure impact on the slope parameter

### What-If Analysis

Use the model to make predictions for different marketing spend levels:
""")

# What-if analysis
spend_input = st.number_input(
    "Enter Marketing Spend ($1000s)",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=10.0
)

predicted_revenue = lr_model.predict(np.array([spend_input]))[0]

st.success(f"**Predicted Revenue**: ${predicted_revenue:,.0f}")
st.info(f"This is based on the equation: Revenue = {params['slope']:.2f} × {spend_input} + {params['intercept']:,.0f}")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
### Summary

This app demonstrated:
- How to generate and analyze synthetic financial data
- Linear regression for modeling relationships
- The training process and performance metrics
- Gradient descent optimization with mathematical foundations
- Comprehensive visualizations of the optimization process
- Actionable business insights from the model

**Key Takeaway**: Machine learning models like linear regression can turn data into actionable insights,
helping businesses make informed decisions.
""")
