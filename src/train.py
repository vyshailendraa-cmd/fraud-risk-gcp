"""Train and evaluate fraud models: baseline, XGBoost, LightGBM.

Data comes from the BigQuery feature table (sql/01_features.sql) or a local
CSV/Parquet export for testing.

Design choices (the interview talking points):
- Time-based split (train on the past, test on the future), never random.
  A random split lets the model see "future" behaviour and inflates scores.
- Metrics suited to ~3.5% fraud: PR-AUC and recall at a fixed false-positive
  rate. Accuracy is useless here (always predicting "legit" gets ~96.5%).
- Decision threshold chosen by business cost on the validation set, then
  reported once on the untouched test set.

Usage:
  python src/train.py --project <gcp-project> --table fraud.features
  python src/train.py --file data/features.parquet
"""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "isFraud"
TIME_COL = "TransactionDT"
CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain"]
NUMERIC = [
    "TransactionAmt", "log_amt", "hour_of_day", "day_of_week",
    "txn_count_1h", "txn_count_24h", "amt_sum_24h",
    "card_txn_history", "amt_vs_card_avg", "secs_since_prev_txn",
    "C1", "C2", "C5", "C6", "C9", "C11", "C13", "C14",
    "D1", "D4", "D10", "D15",
]
FEATURES = NUMERIC + CATEGORICAL

# Business cost assumptions (state them in the README; tune if you like)
COST_REVIEW = 5.0   # cost of a false alarm: an analyst review / annoyed customer
# Cost of a missed fraud = the transaction amount (money lost)


def load_data(args) -> pd.DataFrame:
    if args.file:
        path = Path(args.file)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        from google.cloud import bigquery

        client = bigquery.Client(project=args.project)
        cols = ", ".join([TARGET, TIME_COL] + FEATURES)
        df = client.query(f"SELECT {cols} FROM `{args.project}.{args.table}`").to_dataframe()
    for c in CATEGORICAL:
        df[c] = df[c].fillna("missing").astype("category")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.sort_values(TIME_COL).reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac=0.7, valid_frac=0.15):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + valid_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def recall_at_fpr(y, p, target_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def best_cost_threshold(y, p, amt):
    """Threshold that minimises total cost = missed fraud amount + review cost."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 99):
        flag = p >= t
        cost = amt[(y == 1) & ~flag].sum() + COST_REVIEW * ((y == 0) & flag).sum()
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t


def evaluate(name, y, p, amt, threshold):
    flag = p >= threshold
    caught_amt = amt[(y == 1) & flag].sum()
    total_fraud_amt = amt[y == 1].sum()
    false_alarms = int(((y == 0) & flag).sum())
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4),
        "threshold": round(threshold, 2),
        "fraud_caught_pct": round(float(((y == 1) & flag).sum() / max((y == 1).sum(), 1)), 4),
        "fraud_amount_caught_pct": round(float(caught_amt / max(total_fraud_amt, 1e-9)), 4),
        "false_alarms": false_alarms,
        "total_cost": round(float(amt[(y == 1) & ~flag].sum() + COST_REVIEW * false_alarms), 2),
    }


def train_baseline(tr, va):
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    model = Pipeline([
        ("prep", ColumnTransformer([("num", num, NUMERIC)])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(tr[NUMERIC], tr[TARGET])
    return model, lambda d: model.predict_proba(d[NUMERIC])[:, 1]


def train_xgb(tr, va):
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", early_stopping_rounds=100, n_jobs=-1,
    )
    model.fit(tr[FEATURES], tr[TARGET], eval_set=[(va[FEATURES], va[TARGET])], verbose=False)
    return model, lambda d: model.predict_proba(d[FEATURES])[:, 1]


def train_lgb(tr, va):
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=64, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )
    model.fit(
        tr[FEATURES], tr[TARGET], eval_set=[(va[FEATURES], va[TARGET])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model, lambda d: model.predict_proba(d[FEATURES])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--table", default="fraud.features")
    ap.add_argument("--file")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    if not args.file and not args.project:
        ap.error("give --project (BigQuery) or --file (local)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args)
    tr, va, te = time_split(df)
    print(f"rows: train={len(tr):,} valid={len(va):,} test={len(te):,}")
    print(f"fraud rate: train={tr[TARGET].mean():.3%} test={te[TARGET].mean():.3%}")

    results, curves = [], {}
    for name, fn in [("baseline_logreg", train_baseline), ("xgboost", train_xgb), ("lightgbm", train_lgb)]:
        model, predict = fn(tr, va)
        p_va, p_te = predict(va), predict(te)
        t = best_cost_threshold(va[TARGET].values, p_va, va["TransactionAmt"].values)
        res = evaluate(name, te[TARGET].values, p_te, te["TransactionAmt"].values, t)
        results.append(res)
        curves[name] = precision_recall_curve(te[TARGET].values, p_te)
        joblib.dump(model, out / f"{name}.joblib")
        print(json.dumps(res))

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)

    # PR curves on the test set
    plt.figure(figsize=(6, 5))
    for name, (prec, rec, _) in curves.items():
        plt.plot(rec, prec, label=name)
    plt.axhline(te[TARGET].mean(), ls="--", c="grey", label="random")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of alerts that are fraud)")
    plt.title("Precision-recall on future (test) transactions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)

    # Feature importance from the best model by PR-AUC
    best = max(results[1:], key=lambda r: r["pr_auc"])["model"]
    m = joblib.load(out / f"{best}.joblib")
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values().tail(15)
    plt.figure(figsize=(6, 5))
    imp.plot.barh()
    plt.title(f"Top features ({best})")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    print(f"done; best model: {best}; outputs in {out}/")


if __name__ == "__main__":
    main()
