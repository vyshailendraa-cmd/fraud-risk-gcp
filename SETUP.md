# Setup (100% free, no credit card)

Uses the **BigQuery Sandbox** (free, no card; tables expire after 60 days) and **Google Colab** (free).

## 1. Get the data (Kaggle, free)

1. Sign in at kaggle.com, open the **IEEE-CIS Fraud Detection** competition, go to the Data tab and accept the rules.
2. Download `train_transaction.csv` (about 650 MB).

## 2. Shrink it to the columns we use

In PowerShell, inside this repo folder:

```powershell
pip install pandas pyarrow
python src/prep.py "C:\path\to\train_transaction.csv" data\transactions.parquet
```

Expect about 590,540 rows and a file of roughly 20–40 MB. (`data/` is git-ignored.)

## 3. BigQuery Sandbox (no card)

1. Go to console.cloud.google.com/bigquery and sign in with your Google account.
2. Create a project when prompted (e.g. `fraud-risk-azryl`). **Don't** enable billing. The sandbox banner shows at the top.
3. In the Explorer: ⋮ next to your project → **Create dataset** → ID `fraud`, region `asia-southeast1` → Create.
4. ⋮ next to `fraud` → **Create table**:
   - Create table from: **Upload** → select `data\transactions.parquet`, format **Parquet**
   - Table: `raw_transactions` → Create
5. Check it in a query tab:
   ```sql
   SELECT COUNT(*) n, AVG(isFraud) fraud_rate FROM fraud.raw_transactions
   ```
   Expect ~590,540 rows, fraud rate ~3.5%.

## 4. Build the features

Open a new query tab, paste the contents of `sql/01_features.sql`, and run it. It creates `fraud.features`.

## 5. Train in Google Colab (free)

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

Outputs land in `outputs/`: metrics.json/csv, PR curve and feature importance. Download them from the Colab file panel (left sidebar) and commit `metrics.csv` and the two PNGs to the repo.
