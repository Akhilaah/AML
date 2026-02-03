"""
Complete AML Advanced Pipeline Orchestrator

This script runs the complete end-to-end AML pipeline with all enhancements:
1. Feature Engineering (with advanced features, burst detection, entropy, isolation forest)
2. Model Training (XGBoost with class weights, threshold tuning, SHAP)
3. Inference/Batch Scoring
4. Explainability Generation

Features Added:
- Rolling time-window features (txn count/amount, burst score, time-gap stats)
- Counterparty entropy and network analysis
- Unsupervised anomaly scoring (Isolation Forest)
- XGBoost with class weights for imbalanced data
- Threshold tuning optimized for recall/PR-AUC
- SHAP explainability for model interpretability

Usage:
    python experiments/run_advanced_pipeline.py
    
    # With sample data (for testing)
    python experiments/run_advanced_pipeline.py --sample 0.1
    
    # Inference only
    python experiments/run_advanced_pipeline.py --inference-only \\
        --model-path aml_output/models/aml_xgboost_model.pkl \\
        --feature-path aml_output/features/test_features.parquet
"""

import logging
from pathlib import Path
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import build_all_features
# Note: Advanced features are imported and used via build_all_features,
# which handles loading the correct _v2 versions with Polars-compatible syntax
from src.model.train_model import train_aml_model
from src.model.predict_model import AMLInferenceEngine
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aml_advanced_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(
    transactions_path: Path = Path('data/raw/HI-Medium_Trans.csv'),
    accounts_path: Path = Path('data/raw/HI-Medium_accounts.csv'),
    output_dir: Path = Path('aml_output'),
    sample_fraction: float = None
):
    """Run complete AML pipeline with advanced features."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("AML ADVANCED PIPELINE - COMPLETE RUN")
    logger.info("="*70)
    
    # PHASE 1: FEATURE ENGINEERING
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: ADVANCED FEATURE ENGINEERING")
    logger.info("="*70)
    
    features_dir = output_dir / 'features'
    
    train_features_path, val_features_path, test_features_path = build_all_features(
        transactions_path=transactions_path,
        accounts_path=accounts_path,
        output_dir=features_dir,
        compute_anomaly_scores=True,
        sample_fraction=sample_fraction
    )
    
    logger.info(f"\n✓ Features saved to {features_dir}")
    
    # PHASE 2: MODEL TRAINING
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: XGBOOST MODEL TRAINING")
    logger.info("="*70)
    
    logger.info("Loading feature sets...")
    train_df = pl.read_parquet(train_features_path)
    val_df = pl.read_parquet(val_features_path)
    test_df = pl.read_parquet(test_features_path)
    
    logger.info(f"  Train: {len(train_df)} samples, {len(train_df.columns)} features")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'aml_xgboost_model.pkl'
    
    model = train_aml_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        model_output_path=model_path
    )
    
    logger.info(f"\n✓ Model saved to {model_path}")
    
    # PHASE 3: INFERENCE & SCORING
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: INFERENCE & BATCH SCORING")
    logger.info("="*70)
    
    engine = AMLInferenceEngine(model_path=model_path)
    
    logger.info("Scoring test set...")
    test_scored = engine.predict_batch(test_df, include_anomaly_score=True)
    alerts = engine.generate_alerts(test_df, alert_threshold=0.7)
    
    logger.info(f"\n✓ Generated {len(alerts)} alerts")
    
    # PHASE 4: EXPORT RESULTS
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: EXPORT RESULTS")
    logger.info("="*70)
    
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    engine.export_results(test_scored, results_dir / 'test_scored.parquet', format='parquet')
    engine.export_results(alerts, results_dir / 'alerts.parquet', format='parquet')
    
    logger.info(f"✓ Results exported to {results_dir}")
    
    # SUMMARY
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\n📊 Test Set Statistics:")
    logger.info(f"  Total transactions: {len(test_df)}")
    logger.info(f"  High-risk alerts: {len(alerts)}")
    logger.info(f"  Alert rate: {len(alerts)/len(test_df)*100:.2f}%")
    
    # Metrics
    pdf = test_df.to_pandas()
    target_col = 'Is_Laundering' if 'Is_Laundering' in pdf.columns else 'Is Laundering'
    y_true = pdf[target_col].values
    y_pred = engine.model.predict(test_df)
    y_pred_binary = (y_pred >= model.threshold).astype(int)
    
    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, precision_score, recall_score, f1_score
    
    logger.info(f"\n🎯 Performance Metrics (Threshold={model.threshold:.3f}):")
    logger.info(f"  ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
    
    # Compute PR-AUC with proper sorting to avoid monotonic order error
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    sorted_indices = np.argsort(rec)
    rec_sorted = rec[sorted_indices]
    prec_sorted = prec[sorted_indices]
    logger.info(f"  PR-AUC: {auc(rec_sorted, prec_sorted):.4f}")
    
    logger.info(f"  Precision: {precision_score(y_true, y_pred_binary):.4f}")
    logger.info(f"  Recall: {recall_score(y_true, y_pred_binary):.4f}")
    logger.info(f"  F1-Score: {f1_score(y_true, y_pred_binary):.4f}")
    
    logger.info("\n✨ Key Enhancements:")
    logger.info("  ✓ Advanced rolling features (burst, velocity, time-gaps)")
    logger.info("  ✓ Counterparty entropy & network metrics")
    logger.info("  ✓ Unsupervised anomaly detection (Isolation Forest)")
    logger.info("  ✓ XGBoost with scale_pos_weight")
    logger.info("  ✓ Threshold tuning for Recall/PR-AUC")
    logger.info("  ✓ SHAP explainability")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*70 + "\n")
    
    return {'features_dir': features_dir, 'model_path': model_path, 'results_dir': results_dir}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AML Advanced Pipeline',
        epilog=__doc__
    )
    parser.add_argument('--trans-path', type=Path, default=Path('data/raw/HI-Medium_Trans.csv'))
    parser.add_argument('--accounts-path', type=Path, default=Path('data/raw/HI-Medium_accounts.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('aml_output'))
    parser.add_argument('--sample', type=float, default=None, help='Fraction of data to sample')
    
    args = parser.parse_args()
    run_complete_pipeline(args.trans_path, args.accounts_path, args.output_dir, args.sample)
