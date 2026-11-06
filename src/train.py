"""Train and evaluate fraud models on PaySim features: baseline, XGBoost, LightGBM.

Data comes from the BigQuery feature table (sql/01_features.sql) or a local
Parquet/CSV export for testing.

Design choices (the interview talking points):
- Time-based split by step (hour): train on the past, test on the future.
- Metrics for rare fraud (~0.3%): PR-AUC and recall at a fixed false-positive
  rate. Accuracy is useless (always predicting "legit" scores 99.7%).
- Decision threshold chosen by business cost on the validation set, then
  reported once on the untouched test set.
- A deliberately LEAKY model (using post-transaction balances) is trained too,
  to show how leakage inflates scores. It is reported but never deployed.

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
TIME_COL = "step"
AMOUNT = "amount"
CATEGORICAL = ["type"]
NUMERIC = [
    "amount", "log_amt", "hour_of_day",
    "oldbalanceOrg", "amt_to_orig_balance", "drains_orig_account", "orig_balance_zero",
    "oldbalanceDest", "dest_balance_zero",
    "dest_txn_count_24h", "dest_amt_in_24h", "dest_txn_history", "dest_first_seen",
    "orig_txn_history",
]
FEATURES = NUMERIC + CATEGORICAL
LEAKY = ["leak_newbalanceOrig", "leak_newbalanceDest", "leak_orig_balance_error", "leak_dest_balance_error"]
EXTRA = ["isFlaggedFraud"]


def load_data(args) -> pd.DataFrame:
    if args.file:
        path = Path(args.file)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        from google.cloud import bigquery

        client = bigquery.Client(project=args.project)
        cols = ", ".join([TARGET, TIME_COL] + FEATURES + LEAKY + EXTRA)
        df = client.query(f"SELECT {cols} FROM `{args.project}.{args.table}`").to_dataframe()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in NUMERIC + LEAKY:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac=0.7, valid_frac=0.15):
    """Split on whole hours so no hour is shared between sets."""
    steps = df[TIME_COL].values
    s1 = np.quantile(steps, train_frac)
    s2 = np.quantile(steps, train_frac + valid_frac)
    return df[steps <= s1], df[(steps > s1) & (steps <= s2)], df[steps > s2]


def recall_at_fpr(y, p, target_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def best_cost_threshold(y, p, amt, review_cost):
    """Threshold minimising total cost = missed fraud amount + review cost per false alarm."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 99):
        flag = p >= t
        cost = amt[(y == 1) & ~flag].sum() + review_cost * ((y == 0) & flag).sum()
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t


def evaluate(name, y, p, amt, threshold, review_cost):
    flag = p >= threshold
    false_alarms = int(((y == 0) & flag).sum())
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4),
        "threshold": round(threshold, 2),
        "fraud_caught_pct": round(float(((y == 1) & flag).sum() / max((y == 1).sum(), 1)), 4),
        "fraud_amount_caught_pct": round(float(amt[(y == 1) & flag].sum() / max(amt[y == 1].sum(), 1e-9)), 4),
        "false_alarms": false_alarms,
        "total_cost": round(float(amt[(y == 1) & ~flag].sum() + review_cost * false_alarms), 2),
    }


def train_baseline(tr, va, cols):
    num = [c for c in cols if c in NUMERIC]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    model = Pipeline([
        ("prep", ColumnTransformer([("num", pipe, num)])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(tr[num], tr[TARGET])
    return model, lambda d: model.predict_proba(d[num])[:, 1]


def train_xgb(tr, va, cols):
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", early_stopping_rounds=50, n_jobs=-1,
    )
    model.fit(tr[cols], tr[TARGET], eval_set=[(va[cols], va[TARGET])], verbose=False)
    return model, lambda d: model.predict_proba(d[cols])[:, 1]


def train_lgb(tr, va, cols):
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, importance_type="gain", verbose=-1,
    )
    model.fit(
        tr[cols], tr[TARGET], eval_set=[(va[cols], va[TARGET])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model, lambda d: model.predict_proba(d[cols])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--table", default="fraud.features")
    ap.add_argument("--file")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--review-cost", type=float, default=50.0,
                    help="cost of one false alarm (analyst review / blocked customer)")
    args = ap.parse_args()
    if not args.file and not args.project:
        ap.error("give --project (BigQuery) or --file (local)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args)
    tr, va, te = time_split(df)
    print(f"rows: train={len(tr):,} valid={len(va):,} test={len(te):,}")
    print(f"fraud rate: train={tr[TARGET].mean():.3%} valid={va[TARGET].mean():.3%} test={te[TARGET].mean():.3%}")

    y_te, amt_te = te[TARGET].values, te[AMOUNT].values

    # The existing rule system (isFlaggedFraud) as the "before" benchmark
    rule = te["isFlaggedFraud"].values.astype(float)
    results = [evaluate("existing_rule (isFlaggedFraud)", y_te, rule, amt_te, 0.5, args.review_cost)]

    curves = {}
    runs = [
        ("baseline_logreg", train_baseline, FEATURES),
        ("xgboost", train_xgb, FEATURES),
        ("lightgbm", train_lgb, FEATURES),
        ("LEAKY_xgboost (do not use)", train_xgb, FEATURES + LEAKY),
    ]
    for name, fn, cols in runs:
        model, predict = fn(tr, va, cols)
        t = best_cost_threshold(va[TARGET].values, predict(va), va[AMOUNT].values, args.review_cost)
        p_te = predict(te)
        res = evaluate(name, y_te, p_te, amt_te, t, args.review_cost)
        results.append(res)
        print(json.dumps(res))
        if not name.startswith("LEAKY"):
            curves[name] = precision_recall_curve(y_te, p_te)
            joblib.dump(model, out / f"{name}.joblib")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)

    plt.figure(figsize=(6, 5))
    for name, (prec, rec, _) in curves.items():
        plt.plot(rec, prec, label=name)
    plt.axhline(y_te.mean(), ls="--", c="grey", label="random")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of alerts that are fraud)")
    plt.title("Precision-recall on future (test) transactions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)

    honest = [r for r in results if r["model"] in ("xgboost", "lightgbm")]
    best = max(honest, key=lambda r: r["pr_auc"])["model"]
    m = joblib.load(out / f"{best}.joblib")
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 5))
    imp.plot.barh()
    plt.title(f"Feature importance ({best})")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    print(f"done; best honest model: {best}; outputs in {out}/")


if __name__ == "__main__":
    main()
