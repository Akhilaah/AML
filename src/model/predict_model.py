"""
AML Model Inference and Batch Prediction

This module provides utilities for:
1. Loading trained models for inference
2. Scoring new transactions in batch or streaming mode
3. Generating risk scores and explanations
4. Handling edge cases and missing data
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

from src.model.train_model import AMLXGBoostModel
from src.features.experimental.isolation_forest_anomaly import AnomalyScorer

logger = logging.getLogger(__name__)


class AMLInferenceEngine:
    """
    Production-ready inference engine for AML risk scoring.
    """
    
    def __init__(self,
                 model_path: Path,
                 anomaly_model_path: Optional[Path] = None):
        """
        Load trained models.
        
        Args:
            model_path: Path to trained XGBoost model
            anomaly_model_path: Path to Isolation Forest model
        """
        self.model = AMLXGBoostModel.load(model_path)
        self.anomaly_scorer = None
        
        if anomaly_model_path:
            self.anomaly_scorer = AnomalyScorer.load(anomaly_model_path)
        
        logger.info("✓ Inference engine initialized")
    
    def predict_batch(self,
                     df: pl.DataFrame,
                     include_anomaly_score: bool = True,
                     include_explanation: bool = False) -> pl.DataFrame:
        """
        Score a batch of transactions.
        
        Args:
            df: Input features (Polars DataFrame)
            include_anomaly_score: Add unsupervised anomaly scores
            include_explanation: Add SHAP values (slower)
        
        Returns:
            DataFrame with risk scores
        """
        # Get model predictions
        probas = self.model.predict(df)
        predictions = (probas >= self.model.threshold).astype(int)
        
        # Add to dataframe
        result = df.clone()
        result = result.with_columns([
            pl.Series('xgboost_probability', probas).alias('risk_score_xgboost'),
            pl.Series('prediction', predictions).alias('is_aml_risk'),
        ])
        
        # Add anomaly scores if available
        if include_anomaly_score and self.anomaly_scorer:
            anomaly_scores = self.anomaly_scorer.predict(df)
            result = result.with_columns(
                pl.Series('isolation_forest_anomaly_score', anomaly_scores)
            )
            
            # Combine scores: geometric mean for dual signal
            result = result.with_columns([
                (pl.col('risk_score_xgboost') * 
                 (1 - 1/(1 + np.exp(-pl.col('isolation_forest_anomaly_score'))))).alias('combined_risk_score')
            ])
        
        # Add explanations if requested (caution: expensive operation)
        if include_explanation and self.model.explainer:
            logger.warning("SHAP explanations requested - this is slow for large batches")
            # Implementation would collect SHAP values
            # Typically done offline for batch scoring
        
        return result
    
    def score_with_thresholds(self,
                             df: pl.DataFrame,
                             thresholds: Dict[str, float]) -> pl.DataFrame:
        """
        Apply multiple decision thresholds to generate risk categories.
        
        Args:
            df: Input features
            thresholds: Dict mapping risk levels to probability thresholds
                       e.g. {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.85}
        
        Returns:
            DataFrame with risk category
        """
        result = self.predict_batch(df, include_anomaly_score=True)
        
        # Assign risk category
        risk_category = pl.lit('low')
        for category in sorted(thresholds.keys(), key=lambda x: thresholds[x]):
            risk_category = pl.when(
                pl.col('risk_score_xgboost') >= thresholds[category]
            ).then(pl.lit(category)).otherwise(risk_category)
        
        result = result.with_columns(
            risk_category.alias('risk_category')
        )
        
        return result
    
    def generate_alerts(self,
                       df: pl.DataFrame,
                       alert_threshold: float = 0.7,
                       min_confidence: float = 0.6) -> pl.DataFrame:
        """
        Generate AML alerts for high-risk transactions.
        
        Args:
            df: Input features
            alert_threshold: Probability threshold for alerting
            min_confidence: Minimum model confidence
        
        Returns:
            DataFrame with only flagged transactions
        """
        scored = self.predict_batch(df, include_anomaly_score=True)
        
        # Filter alerts
        alerts = scored.filter(
            (pl.col('risk_score_xgboost') >= alert_threshold) &
            (pl.col('is_aml_risk') == 1)
        )
        
        # Sort by risk score
        alerts = alerts.sort('risk_score_xgboost', descending=True)
        
        logger.info(f"Generated {len(alerts)} AML alerts out of {len(df)} transactions")
        
        return alerts
    
    def detect_anomalies(self,
                        df: pl.DataFrame,
                        anomaly_threshold: float = -0.5) -> pl.DataFrame:
        """
        Detect behavioral anomalies using Isolation Forest.
        
        Args:
            df: Input features
            anomaly_threshold: Anomaly score threshold (lower = more anomalous)
        
        Returns:
            DataFrame with anomalies
        """
        if not self.anomaly_scorer:
            raise ValueError("Anomaly scorer not loaded")
        
        scores = self.anomaly_scorer.predict(df)
        
        result = df.clone()
        result = result.with_columns(
            pl.Series('isolation_forest_anomaly_score', scores)
        )
        
        anomalies = result.filter(
            pl.col('isolation_forest_anomaly_score') < anomaly_threshold
        )
        
        logger.info(f"Detected {len(anomalies)} anomalies out of {len(df)} transactions")
        
        return anomalies
    
    def score_entity(self,
                    entity_id: str,
                    df: pl.DataFrame,
                    aggregation_method: str = 'max') -> Dict:
        """
        Generate risk profile for an entity (aggregate all associated transactions).
        
        Args:
            entity_id: Entity ID to score
            df: Dataset containing entity transactions
            aggregation_method: 'max', 'mean', 'weighted'
        
        Returns:
            Entity risk profile
        """
        entity_txns = df.filter(pl.col('Entity_ID') == entity_id)
        
        if len(entity_txns) == 0:
            logger.warning(f"No transactions found for entity {entity_id}")
            return {'entity_id': entity_id, 'risk_score': None}
        
        # Score transactions
        scored = self.predict_batch(entity_txns, include_anomaly_score=True)
        
        # Aggregate
        if aggregation_method == 'max':
            entity_risk = float(scored.select(pl.col('risk_score_xgboost').max()).item())
            
        elif aggregation_method == 'mean':
            entity_risk = float(scored.select(pl.col('risk_score_xgboost').mean()).item())
            
        elif aggregation_method == 'weighted':
            weights = scored.select('Amount Paid').to_series()
            weights = weights / weights.sum()
            entity_risk = float((scored.select('risk_score_xgboost').to_series() * weights).sum())
        
        # Profile
        profile = {
            'entity_id': entity_id,
            'num_transactions': len(scored),
            'risk_score': entity_risk,
            'num_high_risk_txns': int((scored.select(pl.col('is_aml_risk') == 1).sum())),
            'avg_transaction_amount': float(scored.select(pl.col('Amount Paid').mean()).item()),
            'total_volume': float(scored.select(pl.col('Amount Paid').sum()).item()),
            'anomaly_rate': float(
                (scored.select(pl.col('isolation_forest_anomaly_score') < -0.5).sum()) 
                / len(scored)
            ),
        }
        
        return profile
    
    def export_results(self,
                      scored_df: pl.DataFrame,
                      output_path: Path,
                      format: str = 'parquet'):
        """
        Export scored data to various formats.
        
        Args:
            scored_df: Scored predictions DataFrame
            output_path: Output file path
            format: 'parquet', 'csv', 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            scored_df.write_parquet(output_path)
        elif format == 'csv':
            scored_df.write_csv(output_path)
        elif format == 'json':
            scored_df.write_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✓ Results exported to {output_path}")


def batch_score_transactions(
    feature_parquet_path: Path,
    model_path: Path,
    anomaly_model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    alert_threshold: float = 0.7
) -> pl.DataFrame:
    """
    End-to-end batch scoring workflow.
    """
    logger.info("="*60)
    logger.info("Starting Batch Inference")
    logger.info("="*60)
    
    # Load features
    logger.info(f"Loading features from {feature_parquet_path}")
    df = pl.read_parquet(feature_parquet_path)
    
    # Initialize engine
    engine = AMLInferenceEngine(model_path, anomaly_model_path)
    
    # Score
    logger.info(f"Scoring {len(df)} transactions...")
    scored = engine.predict_batch(df, include_anomaly_score=True)
    
    # Generate alerts
    alerts = engine.generate_alerts(df, alert_threshold=alert_threshold)
    
    # Export
    if output_path:
        output_path = Path(output_path)
        engine.export_results(scored, output_path / 'all_scores.parquet')
        engine.export_results(alerts, output_path / 'alerts.parquet')
        
        logger.info(f"Results saved to {output_path}")
    
    logger.info("="*60)
    logger.info("Batch Inference Complete")
    logger.info("="*60)
    
    return scored, alerts
