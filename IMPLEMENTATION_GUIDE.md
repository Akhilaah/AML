"""
AML PIPELINE ENHANCEMENT - IMPLEMENTATION GUIDE

This document describes the extended AML pipeline with advanced features,
XGBoost modeling, and explainability capabilities.

==================================================================================
1. OVERVIEW OF ENHANCEMENTS
==================================================================================

The AML pipeline has been extended with the following improvements:

A. ADVANCED FEATURE ENGINEERING
   ✓ Rolling time-window features with burst detection
   ✓ Time-gap statistics and regularity analysis
   ✓ Transaction velocity and acceleration metrics
   ✓ Counterparty entropy and network analysis
   ✓ Unsupervised anomaly detection (Isolation Forest)

B. ENHANCED MODEL TRAINING
   ✓ XGBoost classifier with scale_pos_weight for class imbalance
   ✓ Threshold tuning optimized for Recall/PR-AUC
   ✓ Early stopping and cross-validation
   ✓ SHAP explainability for model interpretation

C. PRODUCTION INFERENCE
   ✓ Batch scoring with anomaly signals
   ✓ Risk categorization and alerting
   ✓ Entity-level scoring and profiling
   ✓ Results export in multiple formats

==================================================================================
2. NEW FEATURE MODULES
==================================================================================

2.1 ADVANCED ROLLING FEATURES
    File: src/features/experimental/advanced_rolling_features.py
    
    Features generated:
    - burst_score_1h: Intensity of transaction clustering (0-10 scale)
    - burst_count_24h: Number of burst windows in 24 hours
    - minutes_since_last_txn: Time interval between consecutive txns
    - avg_timegap_minutes_28d: Average inter-transaction time
    - timegap_consistency_28d: Regularity of transactions (0=automated, high=erratic)
    - timegap_acceleration: Change in time gaps (wind-down detection)
    - daily_amount_paid: Daily transaction volume
    - amount_velocity_daily_pct_change: % change in daily volume
    - amount_acceleration_2nd_order: Rate of volume change
    - amount_upper_quartile_deviation_28d: Upper tail spread
    - mean_median_ratio_28d: Skewness indicator
    - unique_counterparties_*d: Counterparty diversity metrics
    - amount_concentration_cv_28d: Volume concentration ratio
    - round_*_ratio_*txns: Round number structuring indicators
    - anomaly_cascade_score: Concurrent anomaly flags
    
    Use Case:
    Detects smurfing/structuring (small regular txns), transaction bursts,
    and wind-down patterns indicating account closure/transition.

2.2 COUNTERPARTY ENTROPY & NETWORK FEATURES
    File: src/features/experimental/counterparty_entropy_features.py
    
    Features generated:
    - counterparty_concentration_ratio_28d: Volume focus (mule detector)
    - sender_receiver_imbalance_28d: Asymmetrical flows
    - receiver_diversity_score_28d: Distribution spreading (0-1)
    - sender_diversity_score_28d: Source spreading (0-1)
    - counterparty_switch_frequency_20txn: Switching rate
    - sender_diversity_switching_rate_20txn: Inbound switching
    - counterparty_recycling_ratio_28d: Round-robin pattern detection
    - inflow_outflow_balance_28d: Pass-through likelihood
    - net_flow_28d: Accumulated funds
    - passthrough_likelihood_28d: Mule vs pass-through score
    - balance_volatility_daily: Flow consistency
    - hours_since_last_counterparty_switch: Switching intervals
    - switch_interval_regularity_std: Automation detection
    - end_of_day_txn_count_7d: Clearing house pattern
    - total_unique_receivers_all_time: Lifetime network size
    - total_unique_senders_all_time: Lifetime source diversity
    - is_high_degree_sender: Hub status (>100 unique receivers)
    - is_high_degree_receiver: Collector status
    
    Use Case:
    Identifies money mules (high inflow), pass-through networks,
    hub-and-spoke architectures, and shell corporations.

2.3 UNSUPERVISED ANOMALY DETECTION
    File: src/features/experimental/isolation_forest_anomaly.py
    
    Algorithm:
    - Isolation Forest trained on behavioral features
    - Unsupervised learning (no labeled data required)
    - Detects statistical outliers in feature space
    - Complementary to supervised XGBoost model
    
    Output:
    - isolation_forest_anomaly_score: -1 (anomalous) to 1 (normal)
    
    Use Case:
    Catches novel attack patterns not in training data,
    identifies unusual account behaviors regardless of known patterns.

==================================================================================
3. ENHANCED MODEL TRAINING
==================================================================================

3.1 XGBoost WITH CLASS WEIGHTS
    File: src/model/train_model.py
    
    Key Features:
    - scale_pos_weight = N_negative / N_positive
      (Automatically computed from training data)
    - Penalizes false negatives more than false positives
    - Targets recall optimization (fewer missed frauds)
    
    Example Scenario:
    If dataset has 999 legitimate and 1 fraudulent transaction:
    - scale_pos_weight = 999 / 1 = 999
    - Model penalizes each false negative by 999x loss
    - Result: High recall, potentially lower precision (acceptable in AML)

3.2 THRESHOLD TUNING
    File: src/model/train_model.py :: tune_threshold()
    
    Optimization Options:
    - 'f2': F2-score (β=2, recall-weighted) [DEFAULT FOR AML]
    - 'f1': F1-score (balanced precision/recall)
    - 'recall': Maximize recall with min precision threshold
    - 'precision': Maximize precision
    
    Process:
    1. Generates probability predictions on validation set
    2. Tests thresholds from 0.1 to 0.95 (step 0.01)
    3. Computes metrics for each threshold
    4. Selects threshold maximizing target metric
    
    Example Output:
        Optimal threshold: 0.45
        - Precision: 0.82
        - Recall: 0.91
        - F2: 0.88

3.3 EARLY STOPPING & CROSS-VALIDATION
    - Early stopping: Stops boosting if validation metric plateaus (50 rounds)
    - Reduces overfitting and training time
    - Uses PR-AUC as validation metric (better for imbalanced data)

==================================================================================
4. EXPLAINABILITY WITH SHAP
==================================================================================

File: src/model/train_model.py :: explain_predictions()

SHAP (SHapley Additive exPlanations) provides:

1. Global Importance:
   - Which features matter most for predictions overall
   - Mean |SHAP value| per feature
   
2. Local Explanations:
   - For each transaction: which features pushed prediction up/down
   - SHAP values show impact direction and magnitude
   
3. Interpretability Benefits:
   - Regulators: Explain why transaction was flagged
   - Investigators: Understand risk factors
   - Model improvement: Identify important features

Example Interpretation:
    "Transaction flagged as risky because:
     - Counterparty is high-degree receiver (SHAP: +0.45)
     - Burst score elevated (SHAP: +0.32)
     - Time gap abnormally large (SHAP: +0.18)
     - Overall risk: 0.78 (HIGH)"

==================================================================================
5. INFERENCE & BATCH SCORING
==================================================================================

File: src/model/predict_model.py

5.1 BATCH SCORING
    engine.predict_batch(df, include_anomaly_score=True)
    
    Returns:
    - risk_score_xgboost: XGBoost probability (0-1)
    - is_aml_risk: Binary prediction (0=clean, 1=risky)
    - isolation_forest_anomaly_score: Unsupervised score
    - combined_risk_score: Weighted combination of signals

5.2 ALERT GENERATION
    engine.generate_alerts(df, alert_threshold=0.7)
    
    Flags transactions with:
    - risk_score_xgboost >= threshold
    - is_aml_risk == 1
    
    Returns sorted by risk (highest first)

5.3 RISK CATEGORIZATION
    engine.score_with_thresholds(
        df,
        thresholds={'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.85}
    )
    
    Assigns categories for routing/action levels

5.4 ENTITY-LEVEL SCORING
    engine.score_entity(entity_id, df, aggregation_method='max')
    
    Aggregates transaction scores to entity level
    Methods: 'max' (worst), 'mean' (average), 'weighted' (by amount)

==================================================================================
6. FILE STRUCTURE & MODULARITY
==================================================================================

Core Pipeline:
├── src/features/
│   ├── build_features.py ........................ Main feature orchestrator
│   └── experimental/
│       ├── rolling_features.py ................. Standard rolling features
│       ├── ratio_features.py ................... Ratio/derived features
│       ├── advanced_rolling_features.py ........ [NEW] Burst, velocity, gaps
│       ├── counterparty_entropy_features.py ... [NEW] Entropy & networks
│       └── isolation_forest_anomaly.py ........ [NEW] Unsupervised scoring
│
├── src/model/
│   ├── train_model.py .......................... [ENHANCED] XGBoost + SHAP
│   └── predict_model.py ........................ [NEW] Inference engine
│
└── experiments/
    └── run_advanced_pipeline.py ............... [NEW] End-to-end orchestrator

Key Design Principles:
✓ Separation of concerns: Each module has single responsibility
✓ Lazy evaluation: Features computed on demand, memory efficient
✓ Modular composition: Mix-and-match feature generations
✓ Testability: Each function independently verifiable
✓ Scalability: Polars for lazy/streaming evaluation

==================================================================================
7. EXECUTION FLOW
==================================================================================

7.1 COMPLETE PIPELINE
    Command: python experiments/run_advanced_pipeline.py
    
    Steps:
    1. Load transactions and accounts
    2. Create temporal splits (train/val/test)
    3. Generate all features (base + advanced + entropy + anomaly)
    4. Train XGBoost with class weights
    5. Tune threshold for recall optimization
    6. Evaluate on test set
    7. Generate SHAP explanations
    8. Batch score and generate alerts
    9. Export results

7.2 WITH SAMPLING (for testing)
    Command: python experiments/run_advanced_pipeline.py --sample 0.1
    
    Uses 10% of data for faster iteration (same feature pipeline)

7.3 INFERENCE ONLY (pre-trained model)
    Command: python experiments/run_advanced_pipeline.py --inference-only \\
                 --model-path path/to/model.pkl \\
                 --feature-path path/to/features.parquet

==================================================================================
8. PERFORMANCE OPTIMIZATION STRATEGIES
==================================================================================

Memory Management:
- Lazy evaluation: Polars doesn't load data until needed
- Checkpoint system: Intermediate results saved to disk
- Batch processing: Handles large datasets incrementally

Speed Optimization:
- GPU support: XGBoost configured for CUDA (if available)
- Parallelization: Isolation Forest uses n_jobs=-1
- Feature selection: Only relevant features for anomaly detection

================================================================================
 9. DEPLOYMENT CHECKLIST
==================================================================================

Pre-Deployment:
☐ Test pipeline with sample data (--sample 0.1)
☐ Verify feature statistics (check for NaNs, infinities)
☐ Review SHAP explanations for business logic
☐ Validate recall/precision trade-off with stakeholders
☐ Test inference engine with new data

Deployment:
☐ Save trained model (pickle format)
☐ Version model and features metadata
☐ Set up logging and monitoring
☐ Configure alert thresholds for production
☐ Establish feedback loop for model retraining

Monitoring:
☐ Track alert volume over time
☐ Monitor recall on discovered fraud
☐ Check for concept drift in feature distributions
☐ Measure alert precision (% true positives among alerts)
☐ Schedule monthly model retraining

==================================================================================
10. TROUBLESHOOTING & FAQ
==================================================================================

Q: Why low precision but high recall?
A: Intentional for AML. False negatives (missed fraud) are more costly
   than false positives (manual review of clean transactions).

Q: Model is slow during training.
A: Check GPU availability (NVIDIA/CUDA). CPU training is slower.
   Consider using --sample flag for large datasets.

Q: Feature engineering fails with memory errors.
A: Use checkpoint system: Pipeline automatically saves intermediate results.
   Increase system RAM or use smaller data samples.

Q: How to interpret SHAP values?
A: Positive SHAP = feature pushes toward fraud prediction.
   Negative SHAP = feature pushes toward clean prediction.
   Magnitude = importance of that feature for that transaction.

Q: Can I customize the feature set?
A: Yes! build_features.py is modular. Add/remove feature modules in
   the build_training_features() function.

==================================================================================
11. FUTURE ENHANCEMENTS
==================================================================================

Potential Improvements:
+ Temporal modeling (LSTMs/Transformers for sequence patterns)
+ Graph neural networks for network analysis
+ Federated learning for privacy
+ Active learning for label efficiency
+ Multi-model ensemble (XGBoost + LightGBM + CatBoost)
+ Online learning for concept drift adaptation
+ Risk-based sampling for imbalanced data

==================================================================================
"""

OUTPUT_GUIDE = __doc__

if __name__ == "__main__":
    print(OUTPUT_GUIDE)
