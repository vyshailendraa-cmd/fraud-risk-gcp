"""Prepare the PaySim CSV for upload to the BigQuery Sandbox (free, no card).

- Adds txn_id (the original row order) so transactions within the same hour
  have a stable order.
- Turns account IDs like "C1231006815" into integers plus a merchant flag,
  which makes the files much smaller.
- Writes Parquet parts under 100 MB each, because the BigQuery console upload
  accepts files up to 100 MB. You upload part 1 to create the table, then
  append the other parts.

Usage:
  python src/prep.py "C:\\path\\to\\PS_20174392719_1491204439457_log.csv" data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_PARTS = 3


def main(src: str, out_dir: str) -> None:
    df = pd.read_csv(src)
    df.insert(0, "txn_id", np.arange(1, len(df) + 1, dtype="int64"))
    for col, flag in [("nameOrig", "orig_is_merchant"), ("nameDest", "dest_is_merchant")]:
        df[flag] = df[col].str[0] == "M"
        df[col] = df[col].str[1:].astype("int64")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(np.array_split(np.arange(len(df)), N_PARTS), start=1):
        path = out / f"paysim_part{i}.parquet"
        df.iloc[part].to_parquet(path, index=False)
        print(f"{path}: {len(part):,} rows, {path.stat().st_size / 1e6:.0f} MB")
    print(f"total {len(df):,} rows; fraud rate {df['isFraud'].mean():.4%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
