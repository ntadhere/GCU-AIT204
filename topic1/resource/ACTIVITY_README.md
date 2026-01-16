# 📚 In-Class Activity Materials - Complete Package

## Overview

This package contains everything needed for the **Linear Regression with Gradient Descent** in-class activity for AIT-204.

---

## 📦 What's Included

### For Students (Share These)

1. **`IN_CLASS_ACTIVITY.md`** ⭐ MAIN ACTIVITY DOCUMENT
   - Complete activity instructions
   - Learning objectives and tasks
   - Step-by-step guide
   - Hints and tips
   - Grading rubric
   - FAQ

2. **`regression_app_starter.py`** ⭐ STARTER CODE
   - Well-structured template
   - TODO sections for students to complete
   - Detailed comments and hints
   - About 40% complete (students implement the core logic)

### For Instructors (Keep Private)

3. **`regression_app_SOLUTION.py`** 🔒 COMPLETE SOLUTION
   - Fully working implementation
   - Reference for grading
   - Help debugging student issues
   - **DO NOT SHARE WITH STUDENTS**

4. **`INSTRUCTOR_NOTES.md`** 🔒 TEACHING GUIDE
   - Time management suggestions
   - Common student mistakes
   - Troubleshooting guide
   - Discussion questions
   - Assessment guidance
   - Extension task ideas

5. **`ACTIVITY_README.md`** 📋 THIS FILE
   - Quick reference for instructors
   - File descriptions
   - Setup instructions

---

## 🚀 Quick Start for Instructors

### Before Class (30 minutes)

1. **Review materials**
   ```bash
   # Read these files:
   - IN_CLASS_ACTIVITY.md (understand student tasks)
   - INSTRUCTOR_NOTES.md (preparation tips)
   - regression_app_SOLUTION.py (see expected result)
   ```

2. **Test the starter code**
   ```bash
   cd /Users/isac/Desktop/AIT-204-code-and-resources
   pip install streamlit numpy pandas plotly
   streamlit run regression_app_starter.py
   ```

3. **Test the solution**
   ```bash
   streamlit run regression_app_SOLUTION.py
   ```

4. **Prepare distribution**
   ```bash
   # Share with students:
   - IN_CLASS_ACTIVITY.md
   - regression_app_starter.py

   # Keep private:
   - regression_app_SOLUTION.py
   - INSTRUCTOR_NOTES.md
   ```

### During Class (90-120 minutes)

1. **Introduction (15 min)**
   - Demo the solution app
   - Explain learning objectives
   - Walk through starter code

2. **Work time (60-80 min)**
   - Students work through TODOs
   - Circulate and help
   - Use hints from INSTRUCTOR_NOTES.md

3. **Wrap-up (15 min)**
   - Discuss challenges
   - Share interesting results
   - Connect to course concepts

### After Class

1. **Share solution** (optional, after deadline)
2. **Grade submissions** using rubric in IN_CLASS_ACTIVITY.md
3. **Collect feedback** for future improvements

---

## 📋 Student Requirements

### Prerequisites

**Knowledge:**
- Basic Python programming
- NumPy basics (arrays, operations)
- Understanding of linear regression (from lectures)
- Familiarity with gradient descent concept

**Tools:**
- Python 3.9 or higher
- pip (for installing packages)
- Code editor (VS Code, PyCharm, etc.)

### Required Packages

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.14.0
```

Students can install with:
```bash
pip install streamlit numpy pandas plotly
```

---

## 🎯 Learning Objectives

By completing this activity, students will:

1. ✅ Implement gradient descent from scratch
2. ✅ Understand matrix operations in ML
3. ✅ Compute gradients manually
4. ✅ Evaluate model performance
5. ✅ Build interactive ML applications
6. ✅ Debug ML algorithms

---

## 📊 What Students Build

A complete Streamlit web application that:

- **Generates** synthetic regression data
- **Trains** a linear regression model using gradient descent
- **Evaluates** performance with MSE, RMSE, MAE, R²
- **Visualizes:**
  - Training progress (loss curve)
  - Predictions vs actual data
  - Residual plots
- **Exports** results to CSV
- **Interactive** parameter tuning

---

## 🔧 Implementation Details

### Starter Code Structure

**Complete (students don't modify):**
- Page configuration
- Data generation function
- Streamlit UI layout
- Session state management

**Incomplete (students implement):**
- `TODO 1`: Initialize weights and bias
- `TODO 2`: Compute predictions (forward pass)
- `TODO 3`: Compute MSE loss
- `TODO 4`: Compute gradients (backpropagation)
- `TODO 5`: Update parameters (gradient descent step)
- `TODO 6`: Compute evaluation metrics
- `TODO 7`: Create visualizations (3 plots)

### Key Concepts Practiced

**Mathematics:**
- Matrix multiplication: `y_pred = X @ w + b`
- Loss function: `MSE = mean((y - y_pred)²)`
- Gradients: `∂L/∂w`, `∂L/∂b`
- Parameter updates: `w = w - α * ∂L/∂w`

**Programming:**
- NumPy operations
- Function implementation
- Debugging matrix shapes
- Session state in Streamlit

---

## ⏱️ Time Estimates

**Total Activity Time: 90-120 minutes**

| Task | Time | Description |
|------|------|-------------|
| Setup & Data Gen | 15 min | Get app running, generate data |
| Gradient Descent | 30 min | Core implementation (hardest part) |
| Evaluation | 20 min | Compute metrics |
| Visualization | 20 min | Create plots |
| Interactive Features | 15 min | Add Streamlit widgets |
| Testing & Debug | 15 min | Make sure everything works |
| Extensions (optional) | 20+ min | Advanced features |

---

## 📝 Grading Rubric

| Component | Points | What to Check |
|-----------|--------|---------------|
| Data Generation | 10 | Uses data generator correctly |
| **Gradient Descent** | 30 | Correct implementation, converges |
| Loss Computation | 10 | MSE formula correct |
| **Evaluation Metrics** | 15 | All 4 metrics (MSE, RMSE, MAE, R²) |
| **Visualizations** | 20 | 3 plots: training, predictions, residuals |
| Code Quality | 10 | Clean, commented, organized |
| UI/UX | 5 | Professional interface |
| **Total** | **100** | |
| Bonus (Extensions) | +10 | Polynomial, regularization, etc. |

---

## 🆘 Common Issues & Solutions

### Issue 1: "Loss is increasing"
**Cause:** Learning rate too high
**Solution:** Reduce to 0.01 or 0.001

### Issue 2: "Loss barely decreasing"
**Cause:** Learning rate too low
**Solution:** Increase to 0.01 or 0.1

### Issue 3: "Shape mismatch error"
**Cause:** Wrong matrix dimensions
**Solution:** Check shapes with `print(X.shape, w.shape)`

### Issue 4: "Gradients are wrong"
**Cause:** Wrong formula or matrix operation
**Solution:** Verify `dw = -(2/n) * X.T @ (y - y_pred)`

### Issue 5: "R² is negative"
**Cause:** Model worse than predicting mean
**Solution:** Check gradient computation, increase iterations

---

## 🎓 Extension Ideas

For students who finish early:

### Level 1: Polynomial Regression (Easy)
- Add polynomial features: `X_poly = np.hstack([X, X**2])`
- Compare linear vs polynomial models

### Level 2: Regularization (Medium)
- Implement L2 regularization (Ridge)
- Add lambda parameter control

### Level 3: Multiple Algorithms (Hard)
- Implement Normal Equation (closed-form solution)
- Compare gradient descent vs analytical solution
- Show when each is better

### Level 4: Advanced Visualization (Medium)
- Animate gradient descent iterations
- Show parameter space with contour plot
- Visualize gradient vectors

---

## 📚 Supporting Materials

Students should have access to:

1. **Streamlit Tutorial**
   - Location: `Streamlit-tutorial/`
   - They should complete this before the activity

2. **Synthetic Dataset Generator**
   - Location: `Topic1-math-gradient-descent/`
   - Can use for more complex data types

3. **Course Lecture Notes**
   - Background on gradient descent
   - Linear regression theory
   - Loss functions

---

## ✅ Pre-Class Checklist

**For Instructors:**

- [ ] Read IN_CLASS_ACTIVITY.md
- [ ] Read INSTRUCTOR_NOTES.md
- [ ] Test regression_app_starter.py
- [ ] Test regression_app_SOLUTION.py
- [ ] Prepare to demo solution
- [ ] Set up projection/screen sharing
- [ ] Prepare to help with debugging
- [ ] Have solution code ready for reference

**For Students (announce ahead):**

- [ ] Complete Streamlit tutorial
- [ ] Install required packages
- [ ] Review gradient descent from lectures
- [ ] Bring laptop with Python installed
- [ ] Be ready to code!

---

## 🔗 File Relationships

```
IN_CLASS_ACTIVITY.md
    ↓ (references)
regression_app_starter.py ← Students work on this
    ↓ (students complete TODOs)
regression_app_SOLUTION.py ← Expected result (instructor reference)
    ↑ (instructor uses for)
INSTRUCTOR_NOTES.md ← Teaching guidance
```

---

## 📤 Distribution

### To Students

**Before class:**
- Email or post on LMS:
  - `IN_CLASS_ACTIVITY.md`
  - `regression_app_starter.py`

**After class (optional):**
- Post solution after submission deadline

### Keep Private

**Never share with students:**
- `regression_app_SOLUTION.py`
- `INSTRUCTOR_NOTES.md`

---

## 🎯 Success Metrics

**The activity is successful if:**

- ✅ 80%+ students complete gradient descent implementation
- ✅ 70%+ students get loss to converge
- ✅ 60%+ students complete all visualizations
- ✅ Students understand gradient descent better
- ✅ Students can debug their own code
- ✅ Students connect theory to practice

---

## 💡 Tips for Success

1. **Start with a demo** - Show the working solution first
2. **Set clear milestones** - "Everyone should have data by 9:30"
3. **Circulate actively** - Don't sit at the front
4. **Use progressive hints** - Don't give answers immediately
5. **Encourage peer learning** - "Discuss concepts, write own code"
6. **Celebrate wins** - "Raise hand when loss converges!"
7. **Connect to theory** - Constantly link to lecture concepts
8. **Have backup plans** - Be ready to live code if needed

---

## 📞 Support

### For Instructors

If you have questions or find issues:
- Check INSTRUCTOR_NOTES.md for detailed guidance
- Test both starter and solution code
- Review common issues section

### For Students (what to tell them)

If stuck:
1. Read the TODO comments carefully
2. Check the hints in IN_CLASS_ACTIVITY.md
3. Compare matrix shapes (print statements)
4. Ask a neighbor (discuss, don't copy)
5. Ask instructor or TA

---

## 🎉 Final Notes

This activity is designed to be:

- **Hands-on** - Students write real code
- **Challenging** - But achievable in 90 minutes
- **Educational** - Connects theory to practice
- **Rewarding** - Students build a real app
- **Flexible** - Extensions for fast finishers

**The goal:** Students leave understanding gradient descent by implementing it themselves!

---

## 📋 Quick Reference

| File | For | Purpose |
|------|-----|---------|
| `IN_CLASS_ACTIVITY.md` | Students | Main instructions |
| `regression_app_starter.py` | Students | Code template |
| `regression_app_SOLUTION.py` | Instructor | Reference solution |
| `INSTRUCTOR_NOTES.md` | Instructor | Teaching guide |
| `ACTIVITY_README.md` | Instructor | This overview |

---

**Ready to go! Your students will build awesome ML apps! 🚀**
