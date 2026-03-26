# Lab 6: Association Rule Mining with Apriori and FP-Growth

## Purpose

This lab explores association rule mining on the **Online Retail Dataset** (UCI Machine Learning Repository). We apply both the **Apriori** and **FP-Growth** algorithms to discover frequent itemsets, generate association rules, and compare the two methods in terms of output and performance.

## Lab Structure

| Step       | Description               | Details                                                                                                                  |
| ---------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Step 1** | Data Preparation          | Load the Online Retail Dataset; drop missing values, cancelled transactions, and duplicates; verify clean data structure |
| **Step 2** | Exploratory Data Analysis | Top-20 item frequency bar plot; co-occurrence heatmap for top-15 items; transaction length statistics                    |
| **Step 3** | Data Transformation       | Group transactions by invoice; one-hot encode with `TransactionEncoder` for algorithm compatibility                      |
| **Step 4** | Apriori Algorithm         | Mine frequent itemsets at 3% minimum support; visualize top-15 itemsets by support                                       |
| **Step 5** | FP-Growth Algorithm       | Mine frequent itemsets with the same threshold; verify output matches Apriori                                            |
| **Step 6** | Association Rule Analysis | Generate rules at 50% confidence threshold; display top rules by lift; scatter plots and heatmap of rule metrics         |
| **Step 7** | Comparative Analysis      | Side-by-side comparison of execution time, itemset counts, and rule counts; discussion of algorithm tradeoffs            |

## Key Insights

- **FP-Growth was faster than Apriori** — it avoids the candidate generation step by compressing transactions into an FP-tree, making it more efficient for this dataset.
- **Both algorithms produced identical results** — the same frequent itemsets and association rules, confirming they solve the same problem with different search strategies.
- **High-lift rules reveal genuine item associations** — rules with lift well above 1.0 indicate items purchased together far more often than random chance would predict, useful for cross-selling and product placement.
- **A 3% support threshold** balanced coverage and interpretability — lower thresholds produced too many noisy itemsets, while higher thresholds missed interesting multi-item patterns.
- **The co-occurrence heatmap** provided an early visual signal of which item pairs to watch for in the formal rule mining stage.

## Challenges and Decisions

1. **Support threshold selection** — Started at 1% which generated too many itemsets. Tested several values and settled on 3%, which provided a manageable number of frequent itemsets while still capturing multi-item patterns.
2. **Memory from one-hot encoding** — With ~3,600+ unique items, the boolean DataFrame was wide but still fit in memory. For larger datasets, sparse matrix representations would be necessary.
3. **Cancelled transactions** — Invoice numbers starting with 'C' represent cancellations and had to be filtered out to avoid inflating item frequencies with non-purchase records.
4. **Frozenset display** — `mlxtend` returns itemsets as `frozenset` objects, which are hard to read. Converted them to sorted comma-separated strings for cleaner display in tables and plot labels.

## How to Run

```bash
git clone https://github.com/aashishshrestha09/MSCS-634-M20.git
cd MSCS-634-M20/lab6
pip install -r requirements.txt
jupyter notebook Lab6_Association_Rule_Mining.ipynb
```

## References

- [mlxtend Documentation — Apriori](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [mlxtend Documentation — FP-Growth](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Agrawal, R. & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
- Han, J., Pei, J., & Yin, Y. (2000). "Mining Frequent Patterns without Candidate Generation"
