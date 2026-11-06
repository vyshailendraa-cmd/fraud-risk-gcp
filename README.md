# Fraud Risk Detection on Google Cloud (BigQuery + Vertex AI)

Predicts whether a card transaction is fraudulent, using ~590,000 real e-commerce transactions (Kaggle IEEE-CIS). Features are built in **BigQuery SQL**, and models are trained on **Vertex AI** (a baseline, then XGBoost, then LightGBM). The decision threshold is chosen by **business cost**, not just by accuracy.

> Status: Phase 1 (data, features, models) done. Next: deployed scoring API, drift monitoring, LLM analyst agent.

## Results (test set = the most recent 15% of transactions)

| Model | PR-AUC | Recall @ 1% FPR | Fraud amount caught | False alarms |
|---|---|---|---|---|
| Baseline (logistic regression) | _fill_ | _fill_ | _fill_ | _fill_ |
| XGBoost | _fill_ | _fill_ | _fill_ | _fill_ |
| LightGBM | _fill_ | _fill_ | _fill_ | _fill_ |

_Copy these numbers from `outputs/metrics.csv` after training._

![PR curve](outputs/pr_curve.png) ![Feature importance](outputs/feature_importance.png)

**In plain English:** _fill in after training, e.g. "At a 1% false-alarm rate, the LightGBM model catches X% of fraud, which is Y% of the money lost to fraud. The strongest signals are how many times a card was used in the last hour and how unusual the amount is for that card."_

## Architecture

```
Kaggle CSV → Cloud Storage → BigQuery (raw) → BigQuery SQL features → Vertex AI (train + evaluate) → models + metrics
```

## Key decisions

**1. Preventing data leakage**
- **Features look only at the past.** Velocity features (transactions per card in the last 1h/24h, spend vs the card's own average) use SQL window frames ending at `1 PRECEDING`, so a transaction never "sees" itself or anything after it.
- **No feature uses the fraud label.** A "card's past fraud count" feature would look powerful but wouldn't be available in real time, because fraud labels arrive days later via chargebacks.
- **Time-based split, not random.** Train on the earliest 70%, tune on the next 15%, test on the latest 15%. A random split lets the model learn from the future and inflates scores.
- **Known residual risk:** the same card can appear in both train and test. That's realistic for production (returning customers), but it means scores on brand-new cards would be lower.

**2. Choosing metrics**
- Only ~3.5% of transactions are fraud, so accuracy is meaningless (predicting "legit" every time gets ~96.5%).
- **PR-AUC** measures how well the model ranks fraud above legitimate transactions.
- **Recall at 1% false-positive rate** answers the business question "how much fraud do we catch if we only bother 1 in 100 genuine customers?"

**3. Cost-based threshold**
Each missed fraud costs its transaction amount, and each false alarm costs an assumed $5 (review time, annoyed customer). The threshold that minimises total cost is picked on the validation set, then reported once on the test set.

## Run it

See [SETUP.md](SETUP.md) for the GCP setup. Then:

```bash
pip install -r requirements.txt
python src/train.py --project <your-gcp-project> --table fraud.features
```

## Tech
Python, SQL, BigQuery, Vertex AI (Colab Enterprise), XGBoost, LightGBM, scikit-learn.
