"""
Rolling window features - Polars-compatible version using integer windows.

Replaced all time-based rolling (rolling_sum_by, etc.) with:
- Integer-window rolling operations
- Or groupby_dynamic for time-based aggregation
"""

import polars as pl


def compute_rolling_features_batch1(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 1: Transaction counts using groupby_dynamic for time-windows.
    """
    # Use groupby_dynamic to create time-based windows, then join back
    hourly = df.group_by_dynamic('Timestamp', every='1h', by=['Account_HASHED']).agg(
        pl.count().alias('txn_count_1h')
    )
    daily = df.group_by_dynamic('Timestamp', every='24h', by=['Account_HASHED']).agg(
        pl.count().alias('txn_count_24h')
    )
    weekly = df.group_by_dynamic('Timestamp', every='7d', by=['Account_HASHED']).agg(
        pl.count().alias('txn_count_7d')
    )
    monthly = df.group_by_dynamic('Timestamp', every='28d', by=['Account_HASHED']).agg(
        pl.count().alias('txn_count_28d')
    )
    
    # Join back to original (forward-fill the counts for all rows in that window)
    df = df.join_asof(
        hourly, by=['Account_HASHED'], on='Timestamp', strategy='backward'
    ).fill_null(0)
    
    df = df.join_asof(
        daily, by=['Account_HASHED'], on='Timestamp', strategy='backward'
    ).fill_null(0)
    
    df = df.join_asof(
        weekly, by=['Account_HASHED'], on='Timestamp', strategy='backward'
    ).fill_null(0)
    
    df = df.join_asof(
        monthly, by=['Account_HASHED'], on='Timestamp', strategy='backward'
    ).fill_null(0)
    
    return df.with_columns([
        pl.col('txn_count_1h').shift(1).fill_null(0),
        pl.col('txn_count_24h').shift(1).fill_null(0),
        pl.col('txn_count_7d').shift(1).fill_null(0),
        pl.col('txn_count_28d').shift(1).fill_null(0),
    ])


def compute_rolling_features_batch2(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 2: Volume statistics using rolling windows.
    
    Using integer window sizes:
    - 500 rows ≈ 28 days
    - 200 rows ≈ 7 days
    """
    return df.with_columns([
        pl.col('Amount Paid')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('total_amount_paid_28d'),

        pl.col('Amount Received')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .fill_null(0.0)
            .alias('total_amount_received_28d'),
    ])


def compute_rolling_features_batch3(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Batch 3: Statistical aggregations using 500-row windows.
    """
    return df.with_columns([
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

        pl.col('Amount Paid')
            .rolling_quantile(window_size=500, quantile=0.5)
            .over('Account_HASHED')
            .shift(1)
            .alias('median_amount_paid_28d'),

        pl.col('Amount Paid')
            .rolling_max(window_size=500)
            .over('Account_HASHED')
            .shift(1)
            .alias('max_amount_paid_28d'),
    ])
