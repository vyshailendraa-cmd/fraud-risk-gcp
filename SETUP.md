# Setup (100% free, no credit card)

Uses the **BigQuery Sandbox** (free, no card; tables expire after 60 days) and **Google Colab** (free).

## 1. Get the data

Download **PaySim** from Kaggle: kaggle.com/datasets/ealaxi/paysim1 → **Download** (about 180 MB zipped).
Unzip it. You get `PS_20174392719_1491204439457_log.csv` (about 470 MB, 6.36 million rows).

## 2. Prepare the upload files

In PowerShell, inside this repo folder:

```powershell
pip install pandas pyarrow
python src/prep.py "C:\Users\<you>\Downloads\PS_20174392719_1491204439457_log.csv" data
```

This writes `data\paysim_part1.parquet`, `part2` and `part3` (each about 86 MB). The BigQuery console accepts uploads up to 100 MB, so the data is split into three parts. (`data/` is git-ignored.)

## 3. Load into the BigQuery Sandbox

1. Go to console.cloud.google.com/bigquery and sign in with your Google account.
2. Create a project when prompted (e.g. `fraud-risk-azryl`). **Don't** enable billing.
3. In the Explorer: ⋮ next to your project → **Create dataset** → Dataset ID `fraud`, location `asia-southeast1` → Create.
4. ⋮ next to `fraud` → **Create table**:
   - Create table from: **Upload** → `paysim_part1.parquet`, file format **Parquet**
   - Table: `raw_transactions` → **Create table**
5. Repeat **Create table** for `paysim_part2.parquet`, then for `paysim_part3.parquet`, with the same table name `raw_transactions`. Under **Advanced options**, set **Write preference** to **Append to table**.
6. Check in a query tab:
   ```sql
   SELECT COUNT(*) AS n, SUM(isFraud) AS frauds FROM fraud.raw_transactions
   ```
   Expect 6,362,620 rows and 8,213 frauds.

## 4. Build the features

New query tab → paste the whole of `sql/01_features.sql` → **Run**. It creates `fraud.features` (2,770,409 TRANSFER and CASH_OUT transactions).

## 5. Train in Google Colab

New notebook at colab.research.google.com. Run each block in its own cell:

```python
from google.colab import auth
auth.authenticate_user()   # sign in with the same Google account
```

```python
!git clone https://github.com/vyshailendraa-cmd/fraud-risk-gcp.git
%cd fraud-risk-gcp
!pip install -q -r requirements.txt
!python src/train.py --project fraud-risk-azryl --table fraud.features
```

Use your own project ID if it's different. Training takes a few minutes.

## 6. Save the results

In the Colab file panel (left sidebar), open `fraud-risk-gcp/outputs/` and download `metrics.csv`, `pr_curve.png` and `feature_importance.png`. Put them in this repo's `outputs/` folder, fill in the README results table from `metrics.csv`, then commit and push.
