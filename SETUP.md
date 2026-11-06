# Setup: GCP + BigQuery (do this first, ~1 hour)

Run these in **Cloud Shell** (the `>_` icon top-right of console.cloud.google.com). It already has `gcloud`, `bq` and Python.

## 1. Project + APIs

```bash
# Create the project (pick your own unique ID)
gcloud projects create fraud-risk-azryl --name="Fraud Risk GCP"
gcloud config set project fraud-risk-azryl

# Link billing in the console first: Billing > Link a billing account (free trial credits)
gcloud services enable bigquery.googleapis.com storage.googleapis.com aiplatform.googleapis.com

export PROJECT_ID=fraud-risk-azryl
export REGION=asia-southeast1          # Singapore, closest to KL
export BUCKET=gs://${PROJECT_ID}-data
gcloud storage buckets create $BUCKET --location=$REGION
```

## 2. Get the data (Kaggle IEEE-CIS Fraud Detection)

1. On kaggle.com, open the *IEEE-CIS Fraud Detection* competition and click **Join / I understand and accept** the rules (required to download).
2. Kaggle > Settings > API > **Create New Token** → downloads `kaggle.json`.
3. In Cloud Shell, upload `kaggle.json` (⋮ menu > Upload), then:

```bash
pip install --user kaggle
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
~/.local/bin/kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv
unzip train_transaction.csv.zip
gcloud storage cp train_transaction.csv $BUCKET/raw/
```

## 3. Load into BigQuery

```bash
bq --location=$REGION mk --dataset ${PROJECT_ID}:fraud

bq --location=$REGION load --autodetect --source_format=CSV --skip_leading_rows=1 \
  fraud.raw_transactions $BUCKET/raw/train_transaction.csv

bq query --use_legacy_sql=false \
  "SELECT COUNT(*) n, AVG(isFraud) fraud_rate FROM fraud.raw_transactions"
# Expect ~590,540 rows, fraud rate ~3.5%
```

## 4. Build features

```bash
bq query --use_legacy_sql=false < sql/01_features.sql
```

## 5. Train on Vertex AI

Console > **Vertex AI > Colab Enterprise** > New notebook (runtime in `asia-southeast1`). Then in a cell:

```python
!git clone https://github.com/vyshailendraa-cmd/fraud-risk-gcp.git
%cd fraud-risk-gcp
!pip install -q -r requirements.txt
!python src/train.py --project fraud-risk-azryl --table fraud.features
```

Outputs land in `outputs/` (metrics.json, PR curve, feature importance, saved models).

## Cost control (read this)

- BigQuery: this whole project uses well under the free 1 TB/month of queries.
- Colab Enterprise: **stop/delete the runtime** when done. It bills while running.
- Later, Vertex AI endpoints bill per hour while deployed. **Undeploy after demos.**
- Set a budget alert: Billing > Budgets & alerts > e.g. USD 20.
