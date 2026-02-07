# Lab 3: Clustering Analysis Using K-Means and K-Medoids Algorithms

## Overview

This lab applies **K-Means** and **K-Medoids** clustering to the [Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset) (178 samples, 13 features, 3 classes). Both algorithms are run with k=3 on z-score standardized data and evaluated using Silhouette Score and Adjusted Rand Index (ARI).

## Lab Structure

| Step                    | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **1. Data Preparation** | Load Wine Dataset, EDA, data quality checks, z-score standardization (StandardScaler)              |
| **2. K-Means**          | Cluster with k=3, compute Silhouette Score and ARI                                                 |
| **3. K-Medoids**        | Cluster with k=3 (alternate PAM), compute Silhouette Score and ARI                                 |
| **4. Visualization**    | PCA 2D scatter plots with centroids/medoids, true label comparison, silhouette analysis, bar chart  |

## Results

| Metric                  | K-Means | K-Medoids |
| ----------------------- | ------- | --------- |
| Silhouette Score        | 0.2849  | 0.2660    |
| Adjusted Rand Index     | 0.8975  | 0.7263    |

## Key Observations

- K-Means outperforms K-Medoids on both metrics — especially ARI (0.90 vs 0.73) — as expected on clean, outlier-free data
- K-Means uses computed centroids (means); K-Medoids uses actual data points (medoids)
- K-Medoids is more robust to outliers but computationally slower
- Feature standardization is essential — raw feature scales range from ~0.13 to ~1680

## Challenges

1. **K-Medoids dependency**: Requires `scikit-learn-extra` package (not in base sklearn). Solved with `pip install scikit-learn-extra`.
2. **Visualization of 13D data**: Used PCA to project to 2D for scatter plots; captures ~55% of total variance.
3. **Cluster-label alignment**: Cluster IDs don't inherently match class IDs — ARI handles this by measuring agreement regardless of label permutation.

## How to Run

```bash
git clone https://github.com/aashishshrestha09/MSCS-634-M20.git
cd MSCS-634-M20/lab3
pip install -r requirements.txt
jupyter notebook Lab3_KMeans_KMedoids_Clustering.ipynb
```

Run all cells in sequence. Expected runtime: ~1 minute.

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Wine Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset
- K-Means: https://scikit-learn.org/stable/modules/clustering.html#k-means
- K-Medoids (scikit-learn-extra): https://scikit-learn-extra.readthedocs.io/
