"""
Advanced rolling features for AML detection - Polars-compatible version.

Features:
- Burst detection (clustering intensity)
- Time-gap statistics (inter-transaction intervals)
- Velocity metrics (rate of change)
- Concentration metrics (volume focus)
- Anomaly cascade (multi-signal detection)

All using row-based rolling windows (Polars-compatible).
"""

import polars as pl
import numpy as np
from typing import List


def compute_burst_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect transaction clustering (smurfing indicator).
    
    Uses simple hourly transaction counts compared to recent average.
    Scores range 0-10 (clamped).
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Hourly transaction counts
    df = df.with_columns([
        pl.col('Timestamp').dt.truncate('1h').alias('hour_window')
    ])
    
    df = df.with_columns([
        pl.col('Timestamp')
            .count()
            .over(['Account_HASHED', 'hour_window'])
            .alias('txn_in_hour')
    ])
    
    # 2. Rolling average (48-hour baseline)
    df = df.with_columns([
        pl.col('txn_in_hour')
            .rolling_mean(window_size=48)
            .over('Account_HASHED')
            .fill_null(1.0)
            .alias('baseline_txn_per_hour_24h')
    ])
    
    # 3. Burst score (ratio to baseline, clamped 0-10)
    df = df.with_columns([
        ((pl.col('txn_in_hour').cast(pl.Float64) / 
          (pl.col('baseline_txn_per_hour_24h') + 1.0)) - 1.0)
        .cast(pl.Float64)
        .alias('burst_score_1h'),
        
        # Count bursts in 24-hour rolling window
        (pl.col('txn_in_hour') > (pl.col('baseline_txn_per_hour_24h') * 2.0))
        .cast(pl.Int8)
        .rolling_sum(window_size=24)
        .over('Account_HASHED')
        .fill_null(0)
        .alias('burst_count_24h')
    ])
    
    return df


def compute_timegap_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute inter-transaction time statistics.
    
    Metrics:
    - Avg/min/max time between transactions (in minutes)
    - Consistency metric (std/mean - lower = more regular)
    - Gap acceleration (increasing intervals = wind-down)
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Time gap to previous transaction (minutes)
    df = df.with_columns([
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 60.0)
        .fill_null(0.0)
        .alias('minutes_since_last_txn')
    ])
    
    # 2. Rolling statistics (500-row window ≈ 28 days)
    df = df.with_columns([
        pl.col('minutes_since_last_txn')
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .alias('avg_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_min(window_size=500)
            .over('Account_HASHED')
            .alias('min_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_max(window_size=500)
            .over('Account_HASHED')
            .alias('max_timegap_minutes_28d'),
        
        pl.col('minutes_since_last_txn')
            .rolling_std(window_size=500)
            .over('Account_HASHED')
            .alias('std_timegap_minutes_28d'),
    ])
    
    # 3. Consistency metric
    df = df.with_columns([
        (pl.col('std_timegap_minutes_28d') / 
         (pl.col('avg_timegap_minutes_28d') + 1.0))
        .fill_null(0.0)
        .alias('timegap_consistency_28d')
    ])
    
    # 4. Gap acceleration
    df = df.with_columns([
        ((pl.col('minutes_since_last_txn') - pl.col('avg_timegap_minutes_28d')) /
         (pl.col('avg_timegap_minutes_28d') + 1.0))
        .alias('timegap_acceleration')
    ])
    
    return df


def compute_velocity_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute transaction velocity (rate of change).
    
    Detects sudden activity spikes or slowdowns.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Daily amount patterns (via rolling sum)
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_sum(window_size=100)
            .over('Account_HASHED')
            .alias('daily_amount_paid')
    ])
    
    # 2. Velocity metrics
    df = df.with_columns([
        # Transaction count change
        (pl.col('daily_amount_paid') - 
         pl.col('daily_amount_paid').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .alias('txn_velocity_daily_delta'),
        
        # Percentage change in amounts
        ((pl.col('daily_amount_paid') - 
          pl.col('daily_amount_paid').shift(1).over('Account_HASHED')) /
         (pl.col('daily_amount_paid').shift(1).over('Account_HASHED') + 0.000001))
        .fill_null(0.0)
        .alias('amount_velocity_daily_pct_change'),
    ])
    
    # 3. Acceleration (second derivative)
    df = df.with_columns([
        (pl.col('amount_velocity_daily_pct_change') -
         pl.col('amount_velocity_daily_pct_change').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .alias('amount_acceleration_2nd_order'),
    ])
    
    return df


def compute_concentration_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect volume concentration (money mule indicator).
    
    Mules have flow to/from few counterparties.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Unique counterparty counts
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
    
    # 2. HHI (Herfindahl-Hirschman Index approximation)
    df = df.with_columns([
        ((pl.col('unique_counterparties_28d') + 1) ** (-1.0))
        .alias('perfect_hhi_28d')
    ])
    
    # 3. Distribution statistics
    df = df.with_columns([
        pl.col('Amount Paid')
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .alias('mean_amount_paid_28d'),
        
        pl.col('Amount Paid')
            .rolling_std(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .alias('std_amount_paid_28d'),
    ])
    
    # 4. Concentration CV (coeff. of variation)
    df = df.with_columns([
        (pl.col('std_amount_paid_28d') / 
         (pl.col('mean_amount_paid_28d') + 0.000001))
        .fill_null(0.0)
        .alias('amount_concentration_cv_28d')
    ])
    
    return df


def compute_round_number_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect structuring (using round numbers like $1000, $500).
    
    AML indicator: many transactions in round amounts.
    """
    df = df.with_columns([
        # Identify round amounts
        ((pl.col('Amount Paid') % 100.0) == 0.0).cast(pl.Int8).alias('is_multiple_100'),
        ((pl.col('Amount Paid') % 500.0) == 0.0).cast(pl.Int8).alias('is_multiple_500'),
        ((pl.col('Amount Paid') % 1000.0) == 0.0).cast(pl.Int8).alias('is_multiple_1000'),
    ])
    
    # 2. Rolling ratios in 50-txn windows
    df = df.with_columns([
        pl.col('is_multiple_100')
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .alias('round_100_ratio_50txns'),
        
        pl.col('is_multiple_1000')
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .alias('round_1000_ratio_50txns'),
        
        (pl.col('is_multiple_100').rolling_sum(window_size=5) / 5.0)
            .over('Account_HASHED')
            .alias('consecutive_round_density_5txn'),
    ])
    
    return df


def compute_anomaly_cascade_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Multi-signal anomaly detection.
    
    Combines multiple fraud signals:
    - High burst score
    - Large time gaps
    - Extreme consistency
    - High volume concentration
    - Heavy structuring
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Define anomaly flags
    df = df.with_columns([
        (pl.col('burst_score_1h') > 2.0).cast(pl.Int8).alias('flag_high_burst'),
        (pl.col('minutes_since_last_txn') > 1440.0).cast(pl.Int8).alias('flag_large_gap'),
        (pl.col('timegap_consistency_28d') < 0.2).cast(pl.Int8).alias('flag_extreme_consistency'),
        (pl.col('amount_concentration_cv_28d') > 1.5).cast(pl.Int8).alias('flag_high_concentration'),
        (pl.col('round_1000_ratio_50txns') > 0.7).cast(pl.Int8).alias('flag_heavy_structuring'),
    ])
    
    # 2. Cascade score (sum of all signals)
    df = df.with_columns([
        (pl.col('flag_high_burst') +
         pl.col('flag_large_gap') +
         pl.col('flag_extreme_consistency') +
         pl.col('flag_high_concentration') +
         pl.col('flag_heavy_structuring'))
        .alias('anomaly_cascade_score')
    ])
    
    # 3. Cascade frequency (rolling mean)
    df = df.with_columns([
        (pl.col('anomaly_cascade_score') >= 2).cast(pl.Int8)
            .rolling_mean(window_size=500)
            .over('Account_HASHED')
            .alias('cascade_frequency_28d')
    ])
    
    return df


def add_advanced_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add all advanced rolling features to DataFrame.
    
    Applies transformations in order and returns enhanced DataFrame.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("  Adding burst scores...")
    df = compute_burst_score(df)
    
    logger.info("  Adding time-gap statistics...")
    df = compute_timegap_statistics(df)
    
    logger.info("  Adding velocity metrics...")
    df = compute_velocity_metrics(df)
    
    logger.info("  Adding concentration metrics...")
    df = compute_concentration_metrics(df)
    
    logger.info("  Adding round number patterns...")
    df = compute_round_number_patterns(df)
    
    logger.info("  Adding anomaly cascade features...")
    df = compute_anomaly_cascade_features(df)
    
    return df
