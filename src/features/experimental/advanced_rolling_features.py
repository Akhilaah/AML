"""
Advanced Rolling Time-Window Features for AML Detection

This module computes sophisticated temporal features that capture:
1. Burst detection and intensity scoring
2. Time-gap statistics (inter-transaction intervals)
3. Advanced volume metrics with acceleration/deceleration
4. Account activity lifecycle patterns
"""

import polars as pl
from typing import Dict, List
import numpy as np


def compute_burst_score(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute burst score - measures the intensity of transaction clustering.
    
    Burst Score Algorithm:
    - Divides time into 1-hour windows
    - Counts transactions per window
    - Compares actual count to baseline (moving average)
    - High burst >= 2x baseline transactions in a window
    
    Returns: DataFrame with burst_score_1h, burst_count_24h columns
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Count transactions per hour per account
    df = df.with_columns([
        pl.col('Timestamp')
            .dt.truncate('1h')
            .alias('hour_window')
    ])
    
    # 2. Get hourly transaction count
    df = df.with_columns([
        pl.col('Timestamp')
            .count()
            .over(['Account_HASHED', 'hour_window'])
            .alias('txn_in_hour')
    ])
    
    # 3. Compute 24-hour baseline (moving average of hourly counts)
    df = df.with_columns([
        pl.col('txn_in_hour')
            .rolling_mean(window_size=24)
            .over('Account_HASHED')
            .fill_null(0)
            .alias('baseline_txn_per_hour_24h')
    ])
    
    # 4. Calculate burst score (ratio to baseline, clamped at 10)
    df = df.with_columns([
        ((pl.col('txn_in_hour') / (pl.col('baseline_txn_per_hour_24h') + 1)) - 1)
            .clip(min_value=0, max_value=10)
            .alias('burst_score_1h'),
        
        # Count bursts in 24-hour window (high burst = score > 1.0)
        (pl.col('txn_in_hour') > (pl.col('baseline_txn_per_hour_24h') * 2))
            .cast(pl.Int8)
            .rolling_sum(window_size=24)
            .over('Account_HASHED')
            .fill_null(0)
            .alias('burst_count_24h')
    ])
    
    return df


def compute_timegap_statistics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute inter-transaction time statistics.
    
    Metrics:
    - Avg/min/max time between consecutive transactions (in minutes)
    - Consistency metric: std / mean (lower = more automated)
    - Gap acceleration: increasing intervals suggest account winding down
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Calculate time gap to previous transaction (in minutes)
    df = df.with_columns([
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 60)
        .fill_null(0)
        .alias('minutes_since_last_txn')
    ])
    
    # 2. 28-day rolling statistics on time gaps
    df = df.with_columns([
        pl.col('minutes_since_last_txn')
            .rolling_mean(window_size='28d', min_periods=5)
            .over('Account_HASHED')
            .alias('avg_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_min(window_size='28d')
            .over('Account_HASHED')
            .alias('min_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_max(window_size='28d')
            .over('Account_HASHED')
            .alias('max_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_std(window_size='28d')
            .over('Account_HASHED')
            .alias('std_timegap_minutes_28d'),
    ])
    
    # 3. Consistency metric: 0 = perfectly regular, high = erratic
    df = df.with_columns([
        (pl.col('std_timegap_minutes_28d') / (pl.col('avg_timegap_minutes_28d') + 1))
            .fill_null(0)
            .alias('timegap_consistency_28d')
    ])
    
    # 4. Detect gap acceleration (increasing time between txns = wind-down pattern)
    df = df.with_columns([
        ((pl.col('minutes_since_last_txn') - pl.col('avg_timegap_minutes_28d')) 
         / (pl.col('avg_timegap_minutes_28d') + 1))
        .alias('timegap_acceleration')
    ])
    
    return df


def compute_velocity_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute transaction velocity and volume acceleration metrics.
    
    Captures:
    - Daily transaction velocity
    - Amount velocity (daily amount change)
    - Acceleration patterns (increasing or decreasing volumes)
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Daily counts and amounts
    df = df.with_columns([
        pl.col('Timestamp')
            .dt.truncate('1d')
            .alias('day_window')
    ])
    
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_sum(window_size=1, by='day_window')
            .over('Account_HASHED')
            .alias('daily_amount_paid')
    ])
    
    # 2. Velocity metrics (rate of change)
    df = df.with_columns([
        # Transaction count velocity (txns/day change)
        (pl.col('txn_count_24h') - pl.col('txn_count_24h').shift(1).over('Account_HASHED'))
            .fill_null(0)
            .alias('txn_velocity_daily_delta'),
        
        # Amount velocity (% change in daily amount)
        ((pl.col('daily_amount_paid') - pl.col('daily_amount_paid').shift(1).over('Account_HASHED'))
         / (pl.col('daily_amount_paid').shift(1).over('Account_HASHED') + 1e-6))
        .fill_null(0)
        .clip(min_value=-2, max_value=2)  # Clip extreme values
        .alias('amount_velocity_daily_pct_change'),
    ])
    
    # 3. Acceleration (second derivative - change in velocity)
    df = df.with_columns([
        (pl.col('amount_velocity_daily_pct_change') 
         - pl.col('amount_velocity_daily_pct_change').shift(1).over('Account_HASHED'))
        .fill_null(0)
        .alias('amount_acceleration_2nd_order'),
    ])
    
    return df


def compute_rolling_percentiles(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute rolling percentile-based features (without using percentile function).
    
    Uses quartile approximations based on min/max/median for better memory efficiency.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # Median already captured in rolling_features.py
    # Here we add min/max ratios as proxy for distribution spread
    
    df = df.with_columns([
        # Amount range as % of median (captures variance without expensive percentile calc)
        ((pl.col('max_amount_paid_28d') - pl.col('mean_amount_paid_28d'))
         / (pl.col('median_amount_paid_28d') + 1e-6))
        .fill_null(0)
        .alias('amount_upper_quartile_deviation_28d'),
        
        # Min-mean spread
        (pl.col('mean_amount_paid_28d') 
         / (pl.col('median_amount_paid_28d') + 1e-6))
        .alias('mean_median_ratio_28d'),
    ])
    
    return df


def compute_concentration_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute metrics that detect concentration/focalization in transaction patterns.
    
    High concentration = money flowing to/from few counterparties (classic smurfing)
    Low concentration = dispersed legitimate behavior
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Count unique counterparties per account in rolling window
    df = df.with_columns([
        pl.col('Account_duplicated_0')
            .n_unique()
            .over(['Account_HASHED', pl.col('Timestamp').dt.truncate('7d')])
            .alias('unique_counterparties_7d'),
        
        pl.col('Account_duplicated_0')
            .n_unique()
            .over(['Account_HASHED', pl.col('Timestamp').dt.truncate('28d')])
            .alias('unique_counterparties_28d'),
    ])
    
    # 2. Herfindahl-Hirschman Index (HHI) approximation
    # Measures volume concentration: 1/N <= HHI <= 1
    # HHI = 1 means all volume to one counterparty
    # HHI = 1/N means uniform distribution across N counterparties
    
    df = df.with_columns([
        (pl.col('unique_counterparties_28d') + 1).pow(-1).alias('perfect_hhi_28d')
    ])
    
    # Concentration via volume variance
    df = df.with_columns([
        (pl.col('std_amount_paid_28d') / (pl.col('mean_amount_paid_28d') + 1e-6))
        .fill_null(0)
        .alias('amount_concentration_cv_28d')
    ])
    
    return df


def compute_round_number_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect patterns in round number usage - automated systems vs legitimate behavior.
    
    Metrics:
    - Proportion of round amounts (100, 500, 1000)
    - Frequency of specific round thresholds
    - Round amount sequences (sign of structuring)
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Check for round number multiples
    df = df.with_columns([
        (pl.col('Amount Paid') % 100 == 0).cast(pl.Int8).alias('is_multiple_100'),
        (pl.col('Amount Paid') % 500 == 0).cast(pl.Int8).alias('is_multiple_500'),
        (pl.col('Amount Paid') % 1000 == 0).cast(pl.Int8).alias('is_multiple_1000'),
    ])
    
    # 2. Rolling proportions (structuring indicator)
    df = df.with_columns([
        pl.col('is_multiple_100')
            .rolling_mean(window_size=50, min_periods=10)
            .over('Account_HASHED')
            .alias('round_100_ratio_50txns'),
        
        pl.col('is_multiple_1000')
            .rolling_mean(window_size=50, min_periods=10)
            .over('Account_HASHED')
            .alias('round_1000_ratio_50txns'),
        
        # Count consecutive round amounts (structuring)
        (pl.col('is_multiple_100')
         .rolling_sum(window_size=5)
         / 5)  # Avg of last 5
        .over('Account_HASHED')
        .alias('consecutive_round_density_5txn'),
    ])
    
    return df


def compute_anomaly_cascade_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect cascading anomalies - when multiple anomaly signals appear together.
    
    Single anomalies are common; cascades are rare and highly suspicious.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Create binary anomaly flags from existing features
    df = df.with_columns([
        # High burst activity
        (pl.col('burst_score_1h') > 2.0).cast(pl.Int8).alias('flag_high_burst'),
        
        # Extreme time gaps (possible gap injection attack)
        (pl.col('minutes_since_last_txn') > 1440).cast(pl.Int8).alias('flag_large_gap'),
        
        # Extreme consistency (robot-like)
        (pl.col('timegap_consistency_28d') < 0.2).cast(pl.Int8).alias('flag_extreme_consistency'),
        
        # High concentration (mule-like)
        (pl.col('amount_concentration_cv_28d') > 1.5).cast(pl.Int8).alias('flag_high_concentration'),
        
        # High round number usage
        (pl.col('round_1000_ratio_50txns') > 0.7).cast(pl.Int8).alias('flag_heavy_structuring'),
    ])
    
    # 2. Cascade score = sum of concurrent anomalies
    df = df.with_columns([
        (pl.col('flag_high_burst') 
         + pl.col('flag_large_gap')
         + pl.col('flag_extreme_consistency')
         + pl.col('flag_high_concentration')
         + pl.col('flag_heavy_structuring'))
        .alias('anomaly_cascade_score')
    ])
    
    # 3. Cascade frequency (how often cascades occur)
    df = df.with_columns([
        (pl.col('anomaly_cascade_score') >= 2)
            .cast(pl.Int8)
            .rolling_mean(window_size=28, min_periods=1)
            .over('Account_HASHED')
            .alias('cascade_frequency_28d')
    ])
    
    return df


# Main composition function
def add_advanced_rolling_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply all advanced rolling time-window features.
    """
    print("  Adding burst scores...")
    df = compute_burst_score(df)
    
    print("  Adding time-gap statistics...")
    df = compute_timegap_statistics(df)
    
    print("  Adding velocity metrics...")
    df = compute_velocity_metrics(df)
    
    print("  Adding percentile features...")
    df = compute_rolling_percentiles(df)
    
    print("  Adding concentration metrics...")
    df = compute_concentration_metrics(df)
    
    print("  Adding round number patterns...")
    df = compute_round_number_patterns(df)
    
    print("  Adding anomaly cascade features...")
    df = compute_anomaly_cascade_features(df)
    
    return df
