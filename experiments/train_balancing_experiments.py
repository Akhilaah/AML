import argparse
import logging
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# --------------------------------------------------
# METRIC
# --------------------------------------------------

def recall_at_precision(y_true, y_prob, target_precision=0.95):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    valid = precision >= target_precision
    if valid.sum() == 0:
        return 0.0
    return recall[valid].max()

# --------------------------------------------------
# DATA LOADING (memory safe)
# --------------------------------------------------

TARGET = "Is Laundering"

def load_split(split, features_dir, sample):

    path = f"{features_dir}/{split}_features.parquet"
    logging.info(f"Scanning {path}")

    lf = pl.scan_parquet(path)

    # -------- memory safe sampling BEFORE collect --------
    if sample < 1:
        step = int(1 / sample)
        logging.info(f"Sampling {sample*100:.2f}% rows")
        lf = lf.filter(pl.int_range(0, pl.len()) % step == 0)

    # select numeric columns only
    schema = lf.collect_schema()
    numeric_cols = [c for c in schema if str(schema[c]).startswith(("Int", "UInt", "Float"))]

    df = lf.select(numeric_cols).collect().to_pandas()

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    logging.info(f"{split} rows loaded: {len(df)} | fraud rate: {y.mean():.6f}")
    return X, y

# --------------------------------------------------
# BALANCING METHODS
# --------------------------------------------------

def apply_balancing(method, X, y):

    # ---------- BASELINE ----------
    if method == "baseline":
        return X, y, None

    # ---------- CLASS WEIGHT ----------
    if method == "class_weight":
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        weight = neg / max(pos, 1)
        sample_weight = np.where(y == 1, weight, 1.0)
        return X, y, sample_weight

    # ---------- UNDERSAMPLING ----------
    if method == "undersample":
        rus = RandomUnderSampler(sampling_strategy=0.05, random_state=42)
        Xr, yr = rus.fit_resample(X, y)
        return Xr, yr, None

    # ---------- SMOTE + ENN ----------
    if method == "smoteenn":

    # ----- Step 1: isolate minority -----
        fraud = X[y == 1]
        legit = X[y == 0]

    # AML trick: keep only limited majority
        max_ratio = 20   # keep 20x legit per fraud
        legit_sample = legit.sample(
        n=min(len(legit), len(fraud) * max_ratio),
        random_state=42
    )

        X_small = pd.concat([fraud, legit_sample])
        y_small = pd.Series([1]*len(fraud) + [0]*len(legit_sample))

    # ----- Step 2: clean numeric -----
        X_small = X_small.replace([np.inf, -np.inf], np.nan).fillna(-999)

    # ----- Step 3: SMOTEENN -----
        from imblearn.combine import SMOTEENN
        sm = SMOTEENN(
        sampling_strategy=0.25,   # IMPORTANT: never 1.0 for fraud
        random_state=42
    )

        Xr, yr = sm.fit_resample(X_small, y_small)

        return Xr, yr, None

    # ---------- FOCAL WEIGHT ----------
    if method == "focal_weight":
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        base = neg / max(pos, 1)
        sample_weight = np.where(y == 1, base * 2.5, 1.0)
        return X, y, sample_weight

    raise ValueError(method)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_and_eval(X, y, method):

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):

        X_train, y_train = X.iloc[tr], y.iloc[tr]
        X_val, y_val = X.iloc[va], y.iloc[va]

        Xb, yb, sw = apply_balancing(method, X_train, y_train)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )

        model.fit(Xb, yb, sample_weight=sw)

        prob = model.predict_proba(X_val)[:,1]

        roc = roc_auc_score(y_val, prob)
        pr = average_precision_score(y_val, prob)
        r95 = recall_at_precision(y_val, prob, 0.95)

        logging.info(f"{method} | Fold {fold} | ROC {roc:.4f} | PR {pr:.4f} | R95 {r95:.4f}")

        results.append([roc, pr, r95])

    return np.mean(results, axis=0)

# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="aml_features")
    parser.add_argument("--sample", type=float, default=0.01)
    args = parser.parse_args()

    X, y = load_split("train", args.features_dir, args.sample)

    methods = ["baseline", "class_weight", "undersample", "smoteenn", "focal_weight"]

    final = {}

    for m in methods:
        logging.info(f"\n===== RUNNING {m.upper()} =====")
        final[m] = train_and_eval(X, y, m)

    print("\nFINAL COMPARISON")
    print("Method | ROC | PR | Recall@95")
    for k,v in final.items():
        print(f"{k:12s} {v[0]:.4f} {v[1]:.5f} {v[2]:.4f}")

if __name__ == "__main__":
    main()