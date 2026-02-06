# AI Agent Instructions for AML Codebase

## Project Overview
Advanced Anti-Money Laundering (AML) detection pipeline using XGBoost and explainable AI. This is a **feature-engineering-first system** that transforms raw transaction data into 79+ behavioral features, trains imbalanced classification models, and provides regulatory audit trails via SHAP explanations.

**Key characteristic**: Pipeline is modular—features can be generated standalone without model training, enabling rapid feature experimentation.

---

## Architecture & Data Flow

### 1. Data Pipeline Stages (Sequential Processing)

```
Raw Data (CSV)
    ↓
[Load & Encode] → accounts, transactions with Account_HASHED (PII hashing)
    ↓
[Sort by Account & Timestamp] → Enables efficient grouping
    ↓
[Feature Engineering Stages]:
    Stage 1: Base features (temporal: day_of_week, hour; Benford checks)
    Stage 2: Entity stats join (account-level reference data)
    Stage 3: Standard rolling features (v2 modules use integer windows)
    Stage 4: Ratio/derived features
    Stage 5: Advanced rolling features (burst, time-gaps, velocity)
    Stage 6: Counterparty entropy (network analysis)
    Stage 7: Network topology (hub detection)
    Stage 8: Toxic corridors (regulator-defined risk pairs)
    Stage 9: Isolation Forest anomaly scoring (unsupervised)
    ↓
[Train/Val/Test Split] → Stratified by label, saved as Parquet
    ↓
[Model Training] (optional):
    XGBoost with scale_pos_weight, threshold tuning, SHAP
```

### 2. Module Organization

**Feature Generation**: `src/features/experimental/`
- `*_v2.py` modules = Polars-refactored versions (always use these)
- Old modules without v2 = deprecated, don't modify
- Each module is a pure function: `LazyFrame → LazyFrame`

**Model Training**: `src/model/`
- `train_model.py`: XGBoost + threshold optimization + SHAP explainer
- `predict_model.py`: Batch scoring, risk categorization

**Data**: `src/data/make_dataset.py` (CSV loading, entity encoding)

**Utils**: `src/utils/hashing.py` (PII anonymization via SHA256)

---

## Critical Developer Workflows

### Running the Pipeline

```bash
# 1. Feature engineering only (fastest, no ML training):
python experiments/run_feature_pipeline.py \
  --trans-path data/raw/HI-Medium_Trans.csv \
  --accounts-path data/raw/HI-Medium_accounts.csv \
  --output-dir aml_features --sample 0.1

# 2. Full pipeline (features + model training):
python experiments/run_advanced_pipeline.py --sample 0.1

# 3. Development iteration (lint check):
make lint  # flake8 check on src/
make clean # remove __pycache__
```

### Entry Points

- **Feature-only**: [experiments/run_feature_pipeline.py](experiments/run_feature_pipeline.py) → calls `build_all_features()`
- **Full pipeline**: `experiments/run_advanced_pipeline.py` (if available)
- **Orchestrator**: [src/features/build_features.py](src/features/build_features.py) (imports all feature stages)

---

## Key Conventions & Patterns

### 1. Polars Instead of Pandas (Critical!)
- **All new code MUST use Polars**, not Pandas
- v2 modules use `groupby_dynamic()`, integer rolling windows, proper `fill_null()` type safety
- **Never use**:
  - `.rolling_sum_by(window_size='28d')` → use `.rolling_sum(window_size=500)` instead
  - `.clip(min_value=X, max_value=Y)` → use `.clip(X, Y)` or manual logic
  - `concat_str(sep=)` → use `concat_str(separator=)`

### 2. Column Naming Conventions

**PII Columns** (always hashed via SHA256):
- Input: `Account Number`, `From Bank`, `To Bank` → Output: `Account_HASHED`, `From_Bank_HASHED`
- Hash function: [src/utils/hashing.py](src/utils/hashing.py) `hash_pii_column()`

**Feature Output Columns**:
- Temporal windows: `{metric}_{window}d` (e.g., `txn_count_28d`)
- Entropy/network: `{metric}_ratio`, `{metric}_score`
- Boolean flags: cast to `Int8` when using `.fill_null(0)` to avoid type mismatch
- All features are `Float32` or `Float64` (except categorical encoded as Int8/Int32)

### 3. Class Imbalance Handling

**XGBoost parameter**: `scale_pos_weight = N_negative / N_positive`
- Auto-computed in [src/model/train_model.py](src/model/train_model.py) `prepare_data()`
- For 999:1 ratio → `scale_pos_weight = 999` (penalize each false negative 999x)
- Targets **recall optimization** (fewer missed AML cases) over precision

**Threshold tuning**: Grid search 0.1–0.95, optimized for F2-score (default, recall-weighted)

### 4. Data Validation Pattern

[src/features/build_features.py](src/features/build_features.py) validates features before output:
- Check no all-NaN columns
- Check numeric dtype (`Float32`, `Float64`, `Int*`)
- Raise if validation fails (prevents silent data corruption)

---

## Feature Engineering Specifics

### Understanding 79 Features (Organized by Module)

**Advanced Rolling** (burst, velocity, time-gaps):
- `burst_score_1h`: Transaction clustering intensity (0–10)
- `timegap_consistency_28d`: Regularity (low=automated, high=erratic)
- `amount_velocity_daily_pct_change`: Daily volume % change
- Files: [src/features/experimental/advanced_rolling_features_v2.py](src/features/experimental/advanced_rolling_features_v2.py)

**Counterparty Entropy** (network analysis, mule/pass-through detection):
- `counterparty_concentration_ratio_28d`: Volume focus (high=mule risk)
- `passthrough_likelihood_28d`: Inflow−outflow balance
- `is_high_degree_sender`: Hub status (>100 unique receivers)
- Files: [src/features/experimental/counterparty_entropy_features_v2.py](src/features/experimental/counterparty_entropy_features_v2.py)

**Isolation Forest**:
- `isolation_forest_anomaly_score`: Unsupervised outlier score (−1 anomalous, +1 normal)
- Files: [src/features/experimental/isolation_forest_anomaly.py](src/features/experimental/isolation_forest_anomaly.py)

### When Adding New Features

1. Create `new_feature_v2.py` in `src/features/experimental/`
2. Function signature: `def add_my_features(df: pl.LazyFrame) -> pl.LazyFrame`
3. Return **lazy** DataFrame to preserve streaming optimization
4. Import & call in [src/features/build_features.py](src/features/build_features.py) build_training_features()
5. Test on `--sample 0.01` (1% data) before full pipeline
6. Validate numeric output: use [src/features/build_features.py](src/features/build_features.py) `validate_features()`

---

## Testing & Debugging

### Debugging Feature Issues

```python
# In build_features.py, check intermediate outputs:
logger.info(f"  Columns after stage 5: {df.collect().columns}")  # Materialize to inspect
logger.info(f"  Shape: {df.collect().shape}")
logger.info(f"  Sample: {df.head(5).collect()}")
```

### Environment & Dependencies

- **Python 3.8+**, uses `uv` for package management
- Key packages: `polars`, `xgboost>=2.0`, `shap>=0.49.1`, `scikit-learn`
- Virtual env: `atlas/` (Miniconda-based)

### CI/Testing

- **No automated tests in repo** (feature-driven pipeline doesn't have fixtures)
- Manual validation: `make features --sample 0.1` must complete without errors
- Lint: `make lint` runs `flake8 src/`

---

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| `AttributeError: rolling_sum_by` | Using old API—switch to v2 module |
| Type mismatch in `.fill_null()` | Cast boolean to Int8 first: `.cast(pl.Int8).fill_null(0)` |
| OOM on full dataset | Use `--sample 0.1` for testing; batch processing in build_features works for 28-day windows |
| Feature column missing in output | Check it's not in `exclude_cols` list in [src/model/train_model.py](src/model/train_model.py) `prepare_data()` |
| Account not found after hashing | Ensure input uses exact column name; check [src/utils/hashing.py](src/utils/hashing.py) |

---

## Reference Documentation

- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Technical deep-dive on each feature module
- **[QUICK_START.md](QUICK_START.md)**: Command cheat sheet
- **[POLARS_REFACTORING_COMPLETE.md](POLARS_REFACTORING_COMPLETE.md)**: Polars v0.20+ migration reference
- **Makefile**: `make features`, `make data`, `make lint`

---

## Recent Changes & Migration Notes

✅ **Polars Refactoring Complete**: All modules now use v0.20+ API (groupby_dynamic, proper type safety)  
✅ **79 Features Enabled**: All 7 feature stages integrated in build_features.py  
✅ **XGBoost + SHAP**: Model training with class weights and regulatory explainability  

When modifying feature modules:
- Always preserve lazy evaluation (return `pl.LazyFrame`)
- Use v2 module patterns as template
- Test with small samples first (`--sample 0.01`)
