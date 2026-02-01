"""Generate a reproducible working dataset for Lab 1 from the Transfermarkt Kaggle dump.

We keep a *raw* CSV in data/ that is suitable for committing to GitHub:
- data/transfermarkt_transfers_raw.csv

Then we generate a *working* CSV with predictable issues (missing values, duplicates,
outliers) so the lab preprocessing steps have something to fix:
- data/transfermarkt_transfers_working.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "transfermarkt_transfers_raw.csv"
WORKING_PATH = REPO_ROOT / "data" / "transfermarkt_transfers_working.csv"


def load_raw_transfers(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw transfers file at {raw_path}. "
            "Copy it from your Kaggle export (datasets/transfers.csv)."
        )

    df = pd.read_csv(raw_path)

    # Normalize / parse types
    if "transfer_date" in df.columns:
        df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")

    df["transfer_fee_eur"] = pd.to_numeric(df.get("transfer_fee"), errors="coerce")
    df["market_value_eur"] = pd.to_numeric(df.get("market_value_in_eur"), errors="coerce")

    # Clean obviously invalid negatives (shouldn't exist, but be defensive)
    for col in ["transfer_fee_eur", "market_value_eur"]:
        df.loc[df[col] < 0, col] = np.nan

    return df


def create_working_copy(df_raw: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df_raw.copy()

    # Derived columns helpful for analysis
    if "transfer_date" in df.columns:
        df["transfer_year"] = df["transfer_date"].dt.year

    df["fee_missing"] = df["transfer_fee_eur"].isna()
    df["value_missing"] = df["market_value_eur"].isna()

    # Simple discretization target for the lab
    # (bins chosen to create interpretable categories)
    bins = [0, 1_000_000, 5_000_000, 20_000_000, 100_000_000, np.inf]
    labels = ["<=1m", "1-5m", "5-20m", "20-100m", ">100m"]
    df["fee_bucket"] = pd.cut(df["transfer_fee_eur"], bins=bins, labels=labels, include_lowest=True)

    # Inject a small amount of missing values (so missing-data handling is demonstrable)
    # Keep this small so it doesn't dominate the dataset.
    non_missing_value_idx = df.index[~df["market_value_eur"].isna()]
    if len(non_missing_value_idx) > 0:
        miss_n = max(30, int(0.01 * len(non_missing_value_idx)))
        miss_idx = rng.choice(non_missing_value_idx, size=min(miss_n, len(non_missing_value_idx)), replace=False)
        df.loc[miss_idx, "market_value_eur"] = np.nan

    if "from_club_name" in df.columns:
        non_missing_from_idx = df.index[~df["from_club_name"].isna()]
        if len(non_missing_from_idx) > 0:
            miss2_n = max(20, int(0.005 * len(non_missing_from_idx)))
            miss2_idx = rng.choice(non_missing_from_idx, size=min(miss2_n, len(non_missing_from_idx)), replace=False)
            df.loc[miss2_idx, "from_club_name"] = np.nan

    # Inject outliers into transfer_fee_eur (a few extreme spikes)
    fee_candidates = df.index[df["transfer_fee_eur"].notna() & (df["transfer_fee_eur"] > 0)]
    if len(fee_candidates) > 0:
        out_n = min(25, len(fee_candidates))
        out_idx = rng.choice(fee_candidates, size=out_n, replace=False)
        df.loc[out_idx, "transfer_fee_eur"] = df.loc[out_idx, "transfer_fee_eur"] * 8

    # Inject duplicates
    dup_n = min(200, len(df))
    if dup_n > 0:
        dup_rows = df.sample(n=dup_n, random_state=seed)
        df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle rows (stable randomness)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Recompute missing flags after injection
    df["fee_missing"] = df["transfer_fee_eur"].isna()
    df["value_missing"] = df["market_value_eur"].isna()

    return df


def main() -> None:
    df_raw = load_raw_transfers(RAW_PATH)
    df_working = create_working_copy(df_raw, seed=42)

    WORKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_working.to_csv(WORKING_PATH, index=False)

    print(f"Raw transfers: {RAW_PATH} shape={df_raw.shape}")
    print(f"Working transfers: {WORKING_PATH} shape={df_working.shape}")
    print("Missing counts (working):")
    print(df_working[["transfer_fee_eur", "market_value_eur"]].isna().sum())


if __name__ == "__main__":
    main()
