# Lab 5: Clustering Techniques Using DBSCAN and Hierarchical Clustering

## Overview

This lab applies **Hierarchical (Agglomerative) Clustering** and **DBSCAN** to the [Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset) (178 samples, 13 features, 3 classes). Both algorithms are run on z-score standardized data and evaluated using Silhouette Score, Homogeneity Score, Completeness Score, and Adjusted Rand Index.

## Lab Structure

| Step                           | Description                                                                                                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Preparation**        | Load Wine Dataset, EDA (`.head()`, `.info()`, `.describe()`), data quality checks, feature correlation heatmap, z-score standardization, PCA projection                       |
| **2. Hierarchical Clustering** | Agglomerative clustering with Ward linkage; test n_clusters = 2–5; compare 4 linkage methods; silhouette analysis plot; dendrogram generation                                 |
| **3. DBSCAN Clustering**       | K-distance plot for eps selection; parameter sweep over eps × min_samples; visualize clusters and noise; compute Silhouette, Homogeneity, Completeness, ARI; cross-tabulation |
| **4. Analysis & Insights**     | Side-by-side comparison with true labels, quantitative metric table, cross-tabulations, parameter influence discussion, strengths/weaknesses                                  |

## Key Insights

- **Hierarchical Clustering** with `n_clusters=3` and Ward linkage aligns well with the Wine dataset's three known classes. The dendrogram clearly shows three natural clusters, and the silhouette analysis confirms cohesive groupings with most samples above the mean silhouette line.
- **Linkage method matters**: Ward linkage outperforms complete, average, and single linkage on all metrics for this dataset, confirming that variance-minimization suits compact, similarly-sized clusters.
- **DBSCAN** is sensitive to `eps` and `min_samples` on this dataset. The k-distance plot guided eps selection to the 2.0–2.5 range. Small eps values produce many noise points; large eps merges distinct classes.
- **Feature standardization** is critical, as raw feature values range from approximately ~0.13 to ~1680 and could disproportionately influence distance-based computations.
- **Adjusted Rand Index** provides a chance-corrected agreement measure, offering a more reliable comparison than raw homogeneity or completeness alone.
- Hierarchical clustering is better suited for compact, globular clusters (like in the Wine dataset), while DBSCAN excels at detecting arbitrary-shaped clusters and identifying outliers.

## Challenges & Decisions

1. **Parameter tuning for DBSCAN**: Required a systematic grid search over `eps` × `min_samples` to find configurations that produce meaningful clusters. Many combinations resulted in all points being noise or a single cluster. The k-distance plot provided a principled starting point for eps selection.
2. **Visualization of 13D data**: Used PCA to project to 2D for scatter plots; captures ~55% of total variance.
3. **Noise handling in metrics**: Silhouette Score includes noise points (label = -1), which can affect the score. Metrics are only computed when at least 2 clusters are found.
4. **Linkage method selection**: Tested 4 linkage methods (Ward, complete, average, single) to justify the choice of Ward linkage rather than assuming it.

## How to Run

```bash
git clone https://github.com/aashishshrestha09/MSCS-634-M20
cd MSCS-634-M20/lab5
pip install -r requirements.txt
jupyter notebook Lab5_DBSCAN_Hierarchical_Clustering.ipynb
```

Run all cells in sequence. Expected runtime: under 1 minute.

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Wine Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset
- DBSCAN: Ester et al., "A Density-Based Algorithm for Discovering Clusters" (KDD 1996)
- Ward's Method: Ward, J.H., "Hierarchical Grouping to Optimize an Objective Function" (1963)
