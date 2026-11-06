"""Shrink the raw Kaggle file to the columns we use and save as Parquet.

The BigQuery Sandbox (free, no card) loads files through the console upload,
which has a size limit. The raw CSV is ~650 MB with 394 columns; we only need
~30 of them, and Parquet compresses well, so the output is small.

Usage:
  python src/prep.py train_transaction.csv data/transactions.parquet
"""

import sys
from pathlib import Path

import pandas as pd

COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "P_emaildomain",
    "C1", "C2", "C5", "C6", "C9", "C11", "C13", "C14",
    "D1", "D4", "D10", "D15",
]


def main(src: str, dst: str) -> None:
    df = pd.read_csv(src, usecols=COLS)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    size_mb = Path(dst).stat().st_size / 1e6
    print(f"{len(df):,} rows, {df.shape[1]} cols -> {dst} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
