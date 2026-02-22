# Lab 4: Regression Analysis with Regularization Techniques

## Purpose

This lab explores multiple regression techniques and regularization methods using the **Diabetes Dataset** from `sklearn.datasets`. The goal is to understand how different regression models perform for predicting disease progression one year after baseline, and how regularization techniques (Ridge and Lasso) prevent overfitting and improve generalization on unseen data.

### Models Implemented

- **Simple Linear Regression** — single feature (BMI)
- **Multiple Linear Regression** — all 10 features
- **Polynomial Regression** — degrees 1 through 4
- **Ridge Regression** (L2 regularization) — multiple alpha values
- **Lasso Regression** (L1 regularization) — multiple alpha values

## Dataset Overview

The Diabetes dataset contains **442 samples** with **10 baseline health features** (age, sex, BMI, blood pressure, and 6 blood serum measurements). All features are pre-processed (mean-centered and scaled). The target variable is a continuous measure of disease progression. No missing values or duplicates were found.

## Model Performance Summary

| Model                                     | Test MAE  | Test RMSE | Test R²    |
| ----------------------------------------- | --------- | --------- | ---------- |
| Simple Linear Regression (BMI only)       | 52.26     | 63.73     | 0.2334     |
| Multiple Linear Regression (all features) | 42.79     | 53.85     | 0.4526     |
| Polynomial Regression (Degree 2)          | 43.58     | 55.64     | 0.4156     |
| Polynomial Regression (Degree 3)          | 164.85    | 287.13    | -14.56     |
| Ridge Regression (alpha=10.0)             | 39.87     | 51.31     | 0.5030     |
| **Lasso Regression (alpha=1.0)**          | **40.04** | **50.63** | **0.5161** |

**Best model:** Lasso Regression (alpha=1.0) with degree-2 polynomial features achieved the highest test R² of 0.5161 and lowest RMSE of 50.63.

## Key Insights

1. **BMI alone is insufficient:** Simple Linear Regression using only BMI yields R² = 0.23 — a single feature explains less than a quarter of the variance in disease progression. The strongest correlated features with target are `bmi` (0.59), `s5`/ltg (0.57), and `bp` (0.44).

2. **Multiple features improve accuracy significantly:** Using all 10 features nearly doubles R² to 0.45, confirming that disease progression depends on multiple health indicators working together.

3. **Polynomial overfitting is severe:** Degree-2 polynomials expand 10 features to 65 and perform comparably to linear. Degree-3 (285 features) and Degree-4 (1000 features) catastrophically overfit — training R² reaches 1.0 while test R² drops to -14.56 and -32.77 respectively, meaning predictions are worse than predicting the mean.

4. **Regularization effectively controls overfitting:**
   - **Ridge (L2):** Best at alpha=10.0 (R² = 0.503). Shrinks all 65 polynomial coefficients toward zero but retains all of them. Provides smooth regularization.
   - **Lasso (L1):** Best at alpha=1.0 (R² = 0.516). Drives 29 of 65 coefficients exactly to zero, performing automatic feature selection. Only 36 features remain active, yielding a simpler and slightly better model.

5. **Alpha tuning is critical:** Too small an alpha (0.001) provides almost no regularization benefit. Too large (100+) for Lasso eliminates all features (R² drops to -0.01). The sweet spot varies by method: Ridge peaks at alpha=10, Lasso at alpha=1.

6. **Bias-variance trade-off demonstrated clearly:** Simple models underfit (high bias), complex polynomial models overfit (high variance), and regularized models achieve the best balance.

## Challenges and Decisions

- **Feature explosion with polynomials:** Degree-2 polynomial features expand 10 features to 65, and degree-3 to 285. StandardScaler was applied within a pipeline to handle the varying scales and prevent numerical instability.
- **Alpha selection:** A broad range of alpha values (0.001 to 1000) was tested across 8 values to comprehensively show the effect of regularization strength on both Ridge and Lasso.
- **Consistent evaluation:** The same train/test split (`random_state=42`, 80/20 split) was used across all models for fair comparison. All models were evaluated with 4 metrics: MAE, MSE, RMSE, and R².
- **Lasso convergence:** `max_iter=10000` was set for Lasso to ensure convergence at all alpha values, especially lower ones where the optimization problem is harder.
- **Font rendering issue:** A matplotlib font cache issue in the Python 3.14 virtual environment required rebuilding the font manager cache at import time.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Lab4_Regression_Regularization.ipynb
```

## Repository Structure

```
lab4/
├── Lab4_Regression_Regularization.ipynb   # Main notebook
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```
