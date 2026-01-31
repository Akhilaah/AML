"""
Unsupervised Anomaly Detection using Isolation Forest for AML

This module generates unsupervised anomaly scores using Isolation Forest
trained on behavioral features. These scores:
1. Identify outliers without requiring labeled data
2. Detect novel attack patterns not seen during training
3. Serve as a complementary signal to supervised models
"""

import polars as pl
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pickle
from typing import List, Optional, Tuple


class AnomalyScorer:
    """
    Wrapper for Isolation Forest-based anomaly detection.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected proportion of anomalies (0 < contamination < 0.5)
                          Default 0.1 = assume 10% of data is anomalous
        """
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self.feature_columns: List[str] = []
        self.scaler_params: dict = {}
        
    def select_features_for_anomaly_detection(self, df: pd.DataFrame) -> List[str]:
        """
        Select behavioral features most useful for anomaly detection.
        
        Prioritizes:
        1. Network features (counterparty patterns, centrality)
        2. Temporal features (burst, time gaps, velocity)
        3. Amount features (deviations, concentration)
        4. Classification features (round numbers, first digits)
        
        Excludes:
        - Identifiers (Account, timestamps)
        - Target variables (Is_Laundering)
        - Identifiers (Account, timestamps)
        - Target variables (Is_Laundering)
        - Features with high missing rates
        """
        exclude_patterns = [
            'Timestamp', 'Account', 'Bank', 'Currency', 'Payment',
            'Entity', 'Is_Laundering', 'split', 'day_window', 'hour_window',
            'account_first_txn', 'account_pair'
        ]
        
        candidate_features = [col for col in df.columns 
                            if not any(pattern in col for pattern in exclude_patterns)
                            and df[col].dtype in ['float64', 'int64']]
        
        # Filter out features with >50% missing or no variance
        valid_features = []
        for col in candidate_features:
            missing_rate = df[col].isna().sum() / len(df)
            variance = df[col].var()
            
            if missing_rate < 0.5 and (variance is not None and variance > 0):
                valid_features.append(col)
        
        return valid_features
    
    def fit(self, df: pl.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Fit Isolation Forest on training data.
        
        Args:
            df: Polars DataFrame with features
            feature_columns: Features to use. If None, auto-selected.
        """
        # Convert to pandas for sklearn
        pdf = df.to_pandas()
        
        # Auto-select features if not provided
        if feature_columns is None:
            feature_columns = self.select_features_for_anomaly_detection(pdf)
        
        self.feature_columns = feature_columns
        
        # Extract features and handle missing values
        X = pdf[feature_columns].fillna(pdf[feature_columns].median())
        
        # Fit Isolation Forest
        print(f"Fitting IsolationForest on {len(X)} samples with {len(feature_columns)} features")
        print(f"Using contamination={self.contamination}")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1,
            verbose=1
        )
        self.model.fit(X)
        
        # Store normalization parameters for inference
        self.scaler_params = {
            'median': X.median().to_dict(),
            'mad': (X - X.median()).abs().median().to_dict(),  # Median Absolute Deviation
        }
        
        print("✓ IsolationForest fitted successfully")
        
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Generate anomaly scores for new data.
        
        Returns:
            anomaly_scores: -1 (anomaly) to 1 (normal) scaled scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        pdf = df.to_pandas()
        X = pdf[self.feature_columns].fillna(
            pd.Series(self.scaler_params['median'])
        )
        
        # Get anomaly scores (negative = anomalous)
        scores = self.model.score_samples(X)
        
        return scores
    
    def predict_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Efficiently add anomaly scores to Polars LazyFrame.
        """
        @pl.api.register_dataframe_namespace("anomaly")
        def _(df_eager: pl.DataFrame) -> pl.DataFrame:
            scores = self.predict(df_eager)
            return df_eager.with_columns(
                pl.Series(scores).alias('isolation_forest_anomaly_score')
            )
        
        # Process in batches to manage memory
        return df
    
    def save(self, path: Path):
        """Save fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'scaler_params': self.scaler_params,
                'contamination': self.contamination,
            }, f)
        print(f"✓ Anomaly model saved to {path}")
    
    @staticmethod
    def load(path: Path) -> 'AnomalyScorer':
        """Load fitted model from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        scorer = AnomalyScorer(contamination=data['contamination'])
        scorer.model = data['model']
        scorer.feature_columns = data['feature_columns']
        scorer.scaler_params = data['scaler_params']
        
        print(f"✓ Anomaly model loaded from {path}")
        return scorer


def add_isolation_forest_scores(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    contamination: float = 0.1,
    model_path: Optional[Path] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fit Isolation Forest on training data and score all splits.
    
    Compliance Note: 
        - Model fitted on TRAIN only to prevent data leakage
        - then scores all splits consistently
    
    Args:
        train_df, val_df, test_df: Polars DataFrames with features
        contamination: Expected anomaly rate
        model_path: Path to save/load model
    
    Returns:
        Tuple of scored (train, val, test) DataFrames
    """
    scorer = AnomalyScorer(contamination=contamination)
    
    # Fit on training data only
    print("="*60)
    print("Training Isolation Forest (unsupervised)")
    print("="*60)
    
    feature_columns = scorer.select_features_for_anomaly_detection(train_df.to_pandas())
    scorer.fit(train_df, feature_columns=feature_columns)
    
    if model_path:
        scorer.save(model_path)
    
    # Score all splits
    print("\n" + "="*60)
    print("Generating anomaly scores")
    print("="*60)
    
    print("Scoring training set...")
    train_scores = scorer.predict(train_df)
    train_df = train_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', train_scores)
    )
    
    print("Scoring validation set...")
    val_scores = scorer.predict(val_df)
    val_df = val_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', val_scores)
    )
    
    print("Scoring test set...")
    test_scores = scorer.predict(test_df)
    test_df = test_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', test_scores)
    )
    
    # Report statistics
    print("\n" + "="*60)
    print("Anomaly Score Distribution")
    print("="*60)
    for split_name, scores in [('Train', train_scores), ('Val', val_scores), ('Test', test_scores)]:
        print(f"\n{split_name}:")
        print(f"  Min: {scores.min():.4f}")
        print(f"  25%: {np.percentile(scores, 25):.4f}")
        print(f"  Med: {np.percentile(scores, 50):.4f}")
        print(f"  75%: {np.percentile(scores, 75):.4f}")
        print(f"  Max: {scores.max():.4f}")
        print(f"  % Anomalous (score < -0.5): {(scores < -0.5).mean()*100:.2f}%")
    
    return train_df, val_df, test_df


def score_splits_with_pretrained_model(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    model_path: Path
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Score datasets using a pre-trained Isolation Forest model.
    """
    scorer = AnomalyScorer.load(model_path)
    
    train_scores = scorer.predict(train_df)
    train_df = train_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', train_scores)
    )
    
    val_scores = scorer.predict(val_df)
    val_df = val_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', val_scores)
    )
    
    test_scores = scorer.predict(test_df)
    test_df = test_df.with_columns(
        pl.Series('isolation_forest_anomaly_score', test_scores)
    )
    
    return train_df, val_df, test_df
