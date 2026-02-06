# Polars Sorting Fix Summary

## Issue
The pipeline was encountering `polars.exceptions.ComputeError: input data is not sorted` when running `experiments/run_feature_pipeline.py`.

## Root Cause
Polars requires data to be sorted before using certain time-based operations:
- `group_by_dynamic()` - requires sorting by the time column
- `rolling_sum_by()` - requires sorting by the time column  
- `rolling_max_by()` - requires sorting by the time column

The issue occurred because:
1. In `build_features.py`, data was sorted on a LazyFrame: `df = df.sort(['Account_HASHED', 'Timestamp'])`
2. LazyFrames don't execute operations immediately - the sort was deferred
3. When feature functions called `group_by_dynamic` or time-based rolling operations, the data wasn't actually sorted yet
4. This caused Polars to throw the "input data is not sorted" error

## Solution
Added explicit sorting at the beginning of functions that use time-based operations:

### Files Modified

1. **src/features/experimental/rolling_features_v2.py**
   - Added `df = df.sort(['Account_HASHED', 'Timestamp'])` at the start of `compute_rolling_features_batch1()`
   - This ensures data is sorted before `group_by_dynamic()` calls

2. **src/features/experimental/ratio_features.py**
   - Added `df = df.sort(['Account_HASHED', 'Timestamp'])` at the start of `compute_advanced_features()`
   - This ensures data is sorted before `rolling_sum_by()` calls

3. **src/features/experimental/network_features.py**
   - Added `df = df.sort(['Account_HASHED', 'Timestamp'])` at the start of `compute_account_network_features()`
   - This ensures data is sorted before `rolling_max_by()` calls

### Files Already Correct
The following files already had proper sorting in place:
- `src/features/experimental/advanced_rolling_features_v2.py` - sorts at the start of each function
- `src/features/experimental/counterparty_entropy_features_v2.py` - sorts at the start of each function

## Technical Details

### Why Sorting is Required
Polars' time-based operations like `group_by_dynamic` and `rolling_*_by` need sorted data to:
1. Efficiently create time windows
2. Ensure correct temporal ordering for rolling calculations
3. Optimize memory usage during streaming execution

### Sorting Strategy
- Sort by `['Account_HASHED', 'Timestamp']` to ensure:
  - All transactions for the same account are grouped together
  - Within each account, transactions are in chronological order
- This enables efficient window operations per account

## Testing
To verify the fix works:
```bash
python experiments/run_feature_pipeline.py --trans-path data/raw/HI-Medium_Trans.csv --accounts-path data/raw/HI-Medium_accounts.csv --output-dir aml_features --sample 0.1
```

## Impact
- **Performance**: Minimal - sorting is a one-time operation per function, and Polars is highly optimized for sorting
- **Correctness**: Critical - ensures all time-based rolling operations work correctly
- **Compatibility**: Maintains full compatibility with existing pipeline
