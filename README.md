# MSCS 634 — Lab 1

## Data Visualization, Preprocessing & Statistical Analysis Using Python

## Table of Contents

- Project Overview
- Dataset Description
- Methodology & Implementation
- Key Findings & Insights
- Challenges & Design Decisions
- Repository Structure
- How to Reproduce
- Screenshots
- References

## Project Overview

This repository contains a single Jupyter Notebook that demonstrates an end-to-end workflow:

- Load and validate a real dataset
- Create exploratory visualizations
- Preprocess the data (missing values, duplicates, outliers, reduction, scaling, discretization)
- Compute descriptive statistics and correlations

Main notebook: `Lab1_Transfermarkt_Transfers.ipynb`

## Dataset Description

**Dataset:** Transfermarkt (Kaggle) — Football Transfers (`transfers.csv`)

**Source:** https://www.kaggle.com/datasets/davidcariboo/player-scores/data

**Files in this repo:**

- Raw slice (unaltered): `data/transfermarkt_transfers_raw.csv` — **79,646 rows × 10 columns**
- Working copy (for preprocessing): `data/transfermarkt_transfers_working.csv` — **79,846 rows × 16 columns**

### What each row represents

Each row is a player transfer between clubs, typically containing:

- Player name and clubs (from/to)
- Transfer date (and derived transfer year)
- Transfer fee and market value

### Important note about missing values

In real transfer datasets, fees are often missing because they are undisclosed or inconsistently reported. This notebook treats missingness as meaningful information (not just “bad data”).

### Why there are two CSVs

- The **raw** file is kept unchanged for provenance.
- The **working** file is generated to make the lab steps reproducible and easy to demonstrate (it includes derived columns and intentionally injected missingness/duplicates/outliers).

## Methodology & Implementation

All work is implemented in `Lab1_Transfermarkt_Transfers.ipynb`.

### Step 1: Data Collection

- Load the working dataset into a Pandas DataFrame.
- Perform basic verification (preview rows, shape/columns, missingness summary).

### Step 2: Data Visualization

Two visualizations are created:

- Transfer fee distribution using log scaling/log transform (right-skewed data)
- Market value vs transfer fee using a log-log scatter plot

### Step 3: Data Preprocessing

The notebook demonstrates:

- **Missing values:** missingness indicators + simple imputations
- **Duplicates:** detection and removal
- **Outliers:** IQR method applied on `log_fee`
- **Reduction:** sampling and dropping high-cardinality text columns
- **Scaling:** Min–Max scaling for numeric features
- **Discretization:** fee buckets for interpretable categories

### Step 4: Statistical Analysis

The notebook computes:

- `.info()` and `.describe()`
- Central tendency (min/max/mean/median/mode)
- Dispersion (range/IQR/variance/std)
- Correlation matrix (and optional heatmap)

## Key Findings & Insights

- Transfer fees are strongly right-skewed; log scaling makes the distribution interpretable.
- Market value and transfer fee are positively associated, but there is substantial dispersion.
- Preprocessing choices (missing-fee flags, deduplication, and outlier handling) noticeably affect summary statistics and correlation strength.

## Challenges & Design Decisions

- **Undisclosed fees:** keep a missing-fee indicator so “missing” isn’t confused with “zero”.
- **Right-skewed values:** analyze outliers on a log scale using IQR to reduce the impact of extreme deals.
- **Performance/readability:** sampling + dropping text columns keeps the analysis focused on numeric features.

## Repository Structure

```
lab1/
├── Lab1_Transfermarkt_Transfers.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── transfermarkt_transfers_raw.csv
│   └── transfermarkt_transfers_working.csv
├── scripts/
│   └── generate_transfermarkt_transfers.py
└── screenshots/
    └── README.md
```

## How to Reproduce

Note: `.venv/` is a local virtual environment folder and is not committed to the repository.

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Re)generate the working dataset
python scripts/generate_transfermarkt_transfers.py

# 4) Launch Jupyter
python -m jupyter lab
```

Open `Lab1_Transfermarkt_Transfers.ipynb` and run the notebook top-to-bottom.

## Screenshots

Screenshots/artifacts are stored in the `screenshots/` folder.

If you run the notebook top-to-bottom, it can generate the PNG files below automatically (by saving plots/tables to disk).

Expected files (Transfermarkt):

- `screenshots/tm_01_head.png`
- `screenshots/tm_02_fee_hist.png`
- `screenshots/tm_03_value_vs_fee.png`
- `screenshots/tm_04_missing_before.png`
- `screenshots/tm_05_missing_after.png`
- `screenshots/tm_06_duplicates.png`
- `screenshots/tm_07_iqr_bounds.png`
- `screenshots/tm_08_outliers.png`
- `screenshots/tm_09_outliers_removed.png`
- `screenshots/tm_10_reduction.png`
- `screenshots/tm_11_scaling_before.png`
- `screenshots/tm_12_scaling_after.png`
- `screenshots/tm_13_discretization.png`
- `screenshots/tm_14_info.png`
- `screenshots/tm_15_describe.png`
- `screenshots/tm_16_central_tendency.png`
- `screenshots/tm_17_dispersion.png`
- `screenshots/tm_18_correlation.png`
- `screenshots/tm_19_heatmap.png` (optional)

See `screenshots/README.md` for the same list and naming guidance.

## References

- Transfermarkt Kaggle dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores/data
- Pandas documentation: https://pandas.pydata.org/
- Seaborn documentation: https://seaborn.pydata.org/
- scikit-learn preprocessing: https://scikit-learn.org/
