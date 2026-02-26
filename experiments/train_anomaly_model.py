# -*- coding: utf-8 -*-
"""
Memory-safe AML Isolation Forest Training + Streaming Scoring
"""

import logging
import os
import polars as pl
import pickle
import sys
import gc
from pathlib import Path
from typing import List
import mlflow
import dagshub

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- CONSTANTS ----------------
RANDOM_STATE = 42
INPUT_DIR = Path("aml_features")
OUTPUT_DIR = Path("data/processed_with_anomaly")
MODEL_DIR = Path("models")
CONTAMINATION = "auto"

EXCLUDE_COLS = [
    "Is Laundering","is_laundering","Account_HASHED","Account",
    "account_id","transaction_id","tx_id",
    "Timestamp","timestamp","Date","date",
]


# ---------------- SETUP ----------------
def setup_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- LOAD SMALL TRAIN SAMPLE ----------------
def load_sample_for_training(split="train", frac=0.1, batch_size=200_000):
    """
    Memory safe sampler → loads fraction in chunks instead of full RAM load
    """
    path = INPUT_DIR / f"{split}_features.parquet"
    logger.info(f"Sampling {frac*100:.2f}% of {split} for training (streaming)")

    lf = pl.scan_parquet(path)
    lf = lf.filter(pl.int_range(0, pl.len()) % int(1/frac) == 0)

    batches = []
    total = 0

    for batch in lf.collect(streaming=True).iter_slices(batch_size):
        pdf = batch.to_pandas()
        batches.append(pdf)
        total += len(pdf)
        logger.info(f"Loaded {total:,} sampled rows...")

    df = pd.concat(batches, ignore_index=True)
    logger.info(f"Final sampled dataset: {len(df):,} rows")

    return df


# ---------------- FEATURE SELECTION ----------------
def identify_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Keep ONLY numeric columns for Isolation Forest
    """

    # Keep numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    feature_cols = []
    removed_cols = []

    for col in numeric_df.columns:
        if not any(excl.lower() in col.lower() for excl in EXCLUDE_COLS):
            feature_cols.append(col)

    # Log removed columns
    for col in df.columns:
        if col not in feature_cols:
            removed_cols.append(col)

    logger.info(f"Using {len(feature_cols)} numeric feature columns")
    logger.info(f"Removed {len(removed_cols)} non-numeric columns")

    return feature_cols


# ---------------- MEDIAN COMPUTATION ----------------
def compute_median(train_df, feature_cols):
    X = train_df[feature_cols].values.astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    median_values = np.nanmedian(X, axis=0)
    del X
    gc.collect()
    return median_values


# ---------------- PREPROCESS ----------------
def preprocess_split(df, feature_cols, median_values):
    X = df[feature_cols].values.astype(np.float32)
    X[~np.isfinite(X)] = np.nan

    mask = np.isnan(X)
    X[mask] = np.take(median_values, np.where(mask)[1])
    X = np.nan_to_num(X, nan=0.0)

    return X


# ---------------- TRAIN MODEL ----------------
def train_isolation_forest(X_train):
    logger.info("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    model.fit(X_train)
    return model


# ---------------- SAVE MODEL ----------------
def save_model(model):
    model_path = MODEL_DIR / "anomaly_detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {model_path}")


# ================= MEMORY SAFE SCORING =================
def score_split_streaming(name, model, feature_cols, median_values, batch_size=150_000):

    input_path = INPUT_DIR / f"{name}_features.parquet"
    output_path = OUTPUT_DIR / f"{name}_features.parquet"

    logger.info(f"Streaming scoring: {name}")

    first_batch = True
    total_rows = 0

    scanner = pl.scan_parquet(input_path)

    for batch in scanner.collect(streaming=True).iter_slices(batch_size):

        df = batch.to_pandas()

        X = preprocess_split(df, feature_cols, median_values)
        df["anomaly_score"] = -model.score_samples(X)

        total_rows += len(df)

        out = pl.from_pandas(df)

        if first_batch:
            out.write_parquet(output_path)
            first_batch = False
        else:
            with open(output_path, "ab") as f:
                out.write_parquet(f)

        del df, X, out
        gc.collect()

        logger.info(f"{name}: processed {total_rows:,} rows")

    logger.info(f"{name} finished ✓ total rows: {total_rows:,}")


# ---------------- MLflow ----------------
def run_mlflow_tracking(n_features):
    mlflow.set_tracking_uri("https://dagshub.com/virajdeshmukh080818/AML.mlflow")
    dagshub.init(repo_owner='virajdeshmukh080818', repo_name='AML', mlflow=True)

    mlflow.set_experiment("Unsupervised_Anomaly_Feature_Gen")

    with mlflow.start_run():
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("n_features", n_features)
        mlflow.log_artifact(str(MODEL_DIR / "anomaly_detector.pkl"))


# ================= MAIN =================
def main():

    setup_directories()

    # ---- Train ----
    train_df = load_sample_for_training("train", 0.1)
    feature_cols = identify_feature_columns(train_df)
    median_values = compute_median(train_df, feature_cols)

    X_train = preprocess_split(train_df, feature_cols, median_values)
    del train_df
    gc.collect()

    model = train_isolation_forest(X_train)
    del X_train
    gc.collect()

    save_model(model)

    # ---- Stream scoring ----
    for split in ["train", "val", "test"]:
        score_split_streaming(split, model, feature_cols, median_values)

    run_mlflow_tracking(len(feature_cols))

    logger.info("DONE ✓")


if __name__ == "__main__":
    main()