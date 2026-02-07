# Lab 2: Classification Using KNN and RNN Algorithms

## Overview

This lab compares **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** on the [Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset) (178 samples, 13 features, 3 classes). The goal is to evaluate how each algorithm's key parameter (k vs. radius) affects accuracy, and to understand the impact of feature scaling on distance-based classifiers.

## Lab Structure

| Step                    | Description                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **1. Data Preparation** | Load Wine Dataset, EDA, 80/20 stratified split, StandardScaler for KNN                                                  |
| **2. KNN**              | Train with k = 1, 5, 11, 15, 21 on **scaled** data; evaluate with classification reports, confusion matrices, 5-fold CV |
| **3. RNN**              | Train with radius = 350, 400, 450, 500, 550, 600 on **unscaled** data; track outliers; same evaluation                  |
| **4. Visualization**    | Accuracy line plots, side-by-side comparison, analysis                                                                  |

## Results

| Model | Best Param   | Test Accuracy | CV Accuracy (5-fold) | Macro F1 |
| ----- | ------------ | ------------- | -------------------- | -------- |
| KNN   | k = 11       | **100%**      | 97.19% ± 2.59%       | 1.00     |
| RNN   | radius = 350 | 72.22%        | 66.87% ± 5.49%       | 0.63     |

**Why the gap?** KNN uses StandardScaler-normalized data where all 13 features contribute equally. RNN uses raw data (radius values 350–600 are calibrated for unscaled distances), so high-magnitude features like proline (range 278–1680) dominate the distance metric.

## Key Observations

- **KNN**: Moderate k values (5–15) balance overfitting and underfitting; k = 11 achieves perfect test accuracy with low CV variance
- **RNN**: Smaller radius → fewer neighbors → higher accuracy but more outliers; larger radius dilutes local patterns
- **Outlier handling**: RNN uses `outlier_label='most_frequent'` for samples with no neighbors within the radius
- **Cross-validation**: KNN shows tight CV scores (stable); RNN shows higher variance across folds

## Challenges

1. **Radius–scaling mismatch**: Radius values 350–600 are meaningless on StandardScaler-normalized data (where distances are ~0–8). Solved by running RNN on unscaled data.
2. **RNN outliers**: Small radius values leave some test samples with zero neighbors. Solved with `outlier_label='most_frequent'`.
3. **Parameter selection**: No single k or radius is universally best. Used 5-fold CV to get robust estimates beyond a single train/test split.

## How to Run

```bash
git clone https://github.com/aashishshrestha09/MSCS-634-M20.git
cd MSCS-634-M20/lab2
pip install -r requirements.txt
jupyter notebook Lab2_KNN_RNN_Classification.ipynb
```

Run all cells in sequence ("Cell → Run All"). Expected runtime: ~2 minutes.

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Wine Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset
- K-Nearest Neighbors: https://scikit-learn.org/stable/modules/neighbors.html#classification
