# 🎓 In-Class Activity: Build Your Own Regression Analysis App

**Course:** AIT-204 - Introduction to Deep Learning
**Topic:** Topic 1 - Background Math and Gradient-Based Learning
**Duration:** 90-120 minutes
**Difficulty:** Intermediate

---

## 📋 Learning Objectives

By the end of this activity, you will be able to:

1. ✅ Use the Synthetic Dataset Generator to create regression data
2. ✅ Implement linear regression from scratch using gradient descent
3. ✅ Build an interactive Streamlit web application
4. ✅ Evaluate model performance using appropriate metrics
5. ✅ Visualize training progress and model predictions
6. ✅ Understand the relationship between loss and gradient descent

---

## 🎯 Activity Overview

You will build a **complete regression analysis application** using Streamlit that:

- Generates synthetic regression data using the provided generator
- Implements linear regression with gradient descent
- Trains the model and tracks loss over iterations
- Evaluates model performance with multiple metrics
- Visualizes predictions, residuals, and training progress
- Allows interactive exploration of hyperparameters

---

## 🛠️ Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of linear regression concepts
- Familiarity with gradient descent (from lectures)
- Basic understanding of NumPy and Pandas

### Required Tools
- Python 3.9+
- Streamlit (from tutorial)
- Synthetic Dataset Generator (provided)
- Code editor (VS Code, PyCharm, etc.)

### Setup

1. **Ensure you have the Streamlit tutorial folder**
   ```
   Streamlit-tutorial/
   ```

2. **Ensure you have the Synthetic Dataset Generator**
   ```
   Topic1-math-gradient-descent/
   ```

3. **Create your project folder**
   ```bash
   mkdir regression-analysis-app
   cd regression-analysis-app
   ```

4. **Copy the starter code** (provided below)

---

## 📝 Activity Tasks

### Task 1: Setup and Data Generation (15 minutes)

**Goal:** Set up your project and generate synthetic data

**Steps:**
1. Create `app.py` from the starter code provided
2. Create `requirements.txt`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`
5. Test data generation with different parameters

**Deliverable:** Working data generation interface

---

### Task 2: Implement Linear Regression Model (30 minutes)

**Goal:** Implement gradient descent for linear regression

**What you need to implement:**
1. Initialize model parameters (weights and bias)
2. Compute predictions: `y_pred = X @ w + b`
3. Compute Mean Squared Error loss
4. Compute gradients for weights and bias
5. Update parameters using gradient descent

**Key Equations:**

**Prediction:**
```
ŷ = Xw + b
```

**Loss (MSE):**
```
L = (1/n) Σ(y - ŷ)²
```

**Gradients:**
```
∂L/∂w = -(2/n) X^T(y - ŷ)
∂L/∂b = -(2/n) Σ(y - ŷ)
```

**Parameter Update:**
```
w = w - α(∂L/∂w)
b = b - α(∂L/∂b)
```

**Deliverable:** Working gradient descent implementation

---

### Task 3: Train and Evaluate Model (20 minutes)

**Goal:** Train your model and compute evaluation metrics

**What you need to implement:**
1. Training loop with specified iterations
2. Loss tracking over iterations
3. Compute evaluation metrics:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score (Coefficient of Determination)

**Key Metrics:**

**MSE:**
```
MSE = (1/n) Σ(y - ŷ)²
```

**RMSE:**
```
RMSE = √MSE
```

**MAE:**
```
MAE = (1/n) Σ|y - ŷ|
```

**R² Score:**
```
R² = 1 - (SS_res / SS_tot)
where SS_res = Σ(y - ŷ)²
      SS_tot = Σ(y - ȳ)²
```

**Deliverable:** Trained model with evaluation metrics

---

### Task 4: Visualization (20 minutes)

**Goal:** Create visualizations to understand model performance

**What you need to implement:**
1. **Training Progress Plot**
   - Loss vs. Iteration
   - Show convergence

2. **Predictions Plot**
   - Actual vs. Predicted values
   - True function overlay (if available)
   - Regression line

3. **Residuals Plot**
   - Residual vs. Predicted value
   - Identify patterns in errors

**Deliverable:** Interactive visualizations

---

### Task 5: Interactive Features (15 minutes)

**Goal:** Add Streamlit widgets for interactive exploration

**What you need to implement:**
1. Sidebar controls for:
   - Learning rate (α)
   - Number of iterations
   - Data generation parameters
   - Train/test split ratio

2. Real-time updates when parameters change

**Deliverable:** Fully interactive app

---

### Task 6: Extension (Optional - 20 minutes)

**Goal:** Extend your app with advanced features

**Choose one or more:**

1. **Polynomial Regression**
   - Add polynomial feature engineering
   - Compare linear vs. polynomial models

2. **Regularization**
   - Implement L2 regularization (Ridge regression)
   - Add lambda (regularization strength) control

3. **Multiple Algorithms**
   - Add Normal Equation solution
   - Compare gradient descent vs. closed-form

4. **Animation**
   - Animate gradient descent iterations
   - Show parameters converging in real-time

**Deliverable:** Enhanced application with advanced features

---

## 📦 Starter Code

### `requirements.txt`

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.14.0
```

### `app.py` (Starter Code)

See the file `regression_app_starter.py` provided separately.

Key sections you need to complete:
- `TODO 1`: Initialize weights and bias
- `TODO 2`: Compute predictions
- `TODO 3`: Compute MSE loss
- `TODO 4`: Compute gradients
- `TODO 5`: Update parameters
- `TODO 6`: Compute evaluation metrics
- `TODO 7`: Create visualizations

---

## 🎓 Learning Guide

### Understanding Gradient Descent

**Intuition:**
Gradient descent is like walking down a hill in the fog:
- You can only see your immediate surroundings (gradient)
- You take small steps downhill (learning rate)
- You keep walking until you reach the bottom (convergence)

**In Linear Regression:**
- The "hill" is the loss surface
- The "bottom" is the minimum MSE
- Each step updates weights and bias
- Learning rate controls step size

### Common Issues and Solutions

**Issue 1: Loss increases or diverges**
- **Cause:** Learning rate too high
- **Solution:** Decrease learning rate (try 0.01, 0.001)

**Issue 2: Loss decreases very slowly**
- **Cause:** Learning rate too low
- **Solution:** Increase learning rate (but not too much!)

**Issue 3: Loss stops decreasing early**
- **Cause:** Local minimum or convergence
- **Solution:** This might be good! Check if model performs well

**Issue 4: Model doesn't fit data well**
- **Cause:**
  - Non-linear data with linear model
  - Not enough iterations
  - Bad initialization
- **Solution:**
  - Try polynomial features
  - Increase iterations
  - Try different random seeds

### Tips for Success

1. **Start Simple**
   - Begin with simple linear data
   - Get gradient descent working
   - Then try more complex data

2. **Print Debug Information**
   - Print loss every N iterations
   - Check gradient magnitudes
   - Verify parameter updates

3. **Visualize Early and Often**
   - Plot predictions after each implementation step
   - Check if loss is decreasing
   - Verify gradients point in right direction

4. **Test with Known Results**
   - Generate data from y = 2x + 1
   - Your model should learn w ≈ 2, b ≈ 1
   - Verify this happens!

5. **Use Small Datasets Initially**
   - Start with 50-100 samples
   - Easier to debug
   - Faster iterations

---

## ✅ Completion Checklist

Before submitting, ensure your app has:

- [ ] Working data generation interface
- [ ] Gradient descent implementation that converges
- [ ] Loss tracking over iterations
- [ ] All evaluation metrics computed correctly
- [ ] Training progress visualization
- [ ] Predictions vs. actual plot
- [ ] Residuals plot
- [ ] Interactive parameter controls
- [ ] Clean, commented code
- [ ] No errors when running
- [ ] Professional-looking UI

---

## 🎯 Expected Output

When complete, your app should:

1. **Generate Data**
   - User selects dataset type and parameters
   - Data displays in table and preview chart

2. **Train Model**
   - User clicks "Train Model"
   - Training progress shows (loss decreasing)
   - Final metrics display

3. **Show Results**
   - Predictions plot shows good fit to data
   - Residuals show random scatter (good model)
   - Loss curve shows convergence

4. **Interactive**
   - User can change learning rate → see effect on convergence
   - User can change iterations → see if more training helps
   - User can regenerate data → model retrains

---

## 📊 Grading Rubric (If Applicable)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Data Generation** | 10 | Correctly uses synthetic dataset generator |
| **Gradient Descent** | 30 | Correctly implements gradient descent algorithm |
| **Loss Computation** | 10 | MSE loss computed correctly |
| **Evaluation Metrics** | 15 | All metrics (MSE, RMSE, MAE, R²) correct |
| **Visualizations** | 20 | All required plots present and correct |
| **Code Quality** | 10 | Clean, well-commented, organized code |
| **UI/UX** | 5 | Professional, easy to use interface |
| **Total** | **100** | |
| **Bonus** | +10 | Extensions (polynomial, regularization, etc.) |

---

## 💡 Hints

<details>
<summary>Hint 1: Computing Gradients</summary>

For gradient computation, remember:
- Gradient of MSE with respect to weights: `∂L/∂w = -(2/n) X.T @ (y - y_pred)`
- Gradient with respect to bias: `∂L/∂b = -(2/n) * sum(y - y_pred)`
- Use NumPy's matrix multiplication: `@` or `np.dot()`
</details>

<details>
<summary>Hint 2: Feature Scaling</summary>

For better convergence:
- Normalize features: `X_normalized = (X - mean) / std`
- Use StandardScaler or do it manually
- Remember to denormalize predictions!
</details>

<details>
<summary>Hint 3: Checking Your Implementation</summary>

To verify gradients are correct:
1. Generate simple data: y = 2x + 1 (no noise)
2. After training, w should be close to 2, b close to 1
3. If not, check gradient calculations!
</details>

<details>
<summary>Hint 4: R² Score</summary>

```python
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

R² = 1 means perfect fit, R² = 0 means no better than mean.
</details>

<details>
<summary>Hint 5: Streamlit Caching</summary>

Cache expensive operations:
```python
@st.cache_data
def train_model(X, y, lr, iterations):
    # Your training code
    return w, b, losses
```

This prevents retraining on every widget interaction!
</details>

---

## 🔗 Resources

### From This Course
- **Streamlit Tutorial:** `Streamlit-tutorial/README.md`
- **Synthetic Data Generator:** `Topic1-math-gradient-descent/app.py`
- **Example Usage:** `Topic1-math-gradient-descent/example_usage.py`

### External Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [NumPy Documentation](https://numpy.org/doc/)
- [Linear Regression Explained](https://www.youtube.com/watch?v=nk2CQITm_eo) - 3Blue1Brown
- [Gradient Descent Visualization](https://www.youtube.com/watch?v=IHZwWFHWa-w)

### Recommended Reading
- Chapter 3: Linear Regression (Bishop - Pattern Recognition)
- Chapter 5: Machine Learning Basics (Goodfellow - Deep Learning)

---

## 🤝 Collaboration Policy

- **Encouraged:** Discussing concepts with classmates
- **Encouraged:** Asking instructor/TA for help
- **Encouraged:** Using online resources for understanding

- **Not Allowed:** Copying code from classmates
- **Not Allowed:** Sharing your complete solution
- **Not Allowed:** Using AI to write your code

**Goal:** Learn by doing! Use hints and resources, but write your own code.

---

## 📤 Submission

### What to Submit

1. **Your `app.py` file** with completed implementation
2. **`requirements.txt`** file
3. **Screenshots** of your working app showing:
   - Data generation
   - Training progress
   - Final results with visualizations
4. **Brief Report** (1-2 pages) including:
   - Challenges you faced
   - How you solved them
   - What you learned
   - Results analysis

### How to Submit

[To be specified by instructor - could be GitHub, LMS, email, etc.]

### Deadline

[To be specified by instructor]

---

## ❓ FAQ

**Q: Can I use scikit-learn for linear regression?**
A: No! The point is to implement gradient descent yourself to understand it. However, you can use scikit-learn to verify your implementation.

**Q: My loss isn't decreasing. What's wrong?**
A: Check:
1. Learning rate (try smaller values like 0.01, 0.001)
2. Gradient computation (verify the math)
3. Data normalization (features should be similar scales)

**Q: How many iterations should I use?**
A: Start with 1000. If loss hasn't plateaued, increase. If it diverges, decrease learning rate.

**Q: Can I work in groups?**
A: [To be specified by instructor - typically: discuss together, code individually]

**Q: What if I finish early?**
A: Try the extension tasks! Or help a classmate understand (without sharing code).

**Q: Can I use the Normal Equation instead?**
A: For the main activity, no - we want you to implement gradient descent. But you can add it as an extension to compare!

---

## 🎉 Final Thoughts

This activity combines everything from Topic 1:
- Mathematical foundations (calculus, linear algebra)
- Gradient descent optimization
- Loss functions and convergence
- Model evaluation
- Practical implementation

**Most importantly:** You're building a real, working application that you can share and use for future learning!

**Good luck and have fun! 🚀**

---

**Questions?** Ask your instructor, TA, or post in the course forum.

**Need help?** Review the hints section above or consult the provided resources.

**Finished early?** Help a classmate or try the extension challenges!
