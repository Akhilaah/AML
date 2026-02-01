"""
Counterparty entropy and network analysis features - Polars-compatible version.

Detects:
- Money mule behavior (high inflow, low outflow)
- Network concentration (few counterparties)
- Pass-through laundering (balanced flow)
- Hub-and-spoke networks (central node)

All using Polars-compatible syntax (no time-based rolling).
"""

import polars as pl
from typing import Tuple


def compute_counterparty_entropy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Shannon entropy-like diversity metrics for counterparties.
    
    Lower entropy = fewer unique counterparties (mule indicator).
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # Count transactions per counterparty
    txn_per_counterparty = df.group_by(['Account_HASHED', 'Account_duplicated_0']).agg(
        pl.count().alias('txn_count')
    )
    
    # Compute entropy
    txn_per_counterparty = txn_per_counterparty.with_columns([
        pl.col('txn_count').sum().over('Account_HASHED').alias('total_txns'),
        (pl.col('txn_count') / pl.col('txn_count').sum().over('Account_HASHED'))
        .alias('probability')
    ])
    
    txn_per_counterparty = txn_per_counterparty.with_columns([
        (pl.col('probability') * pl.col('probability').log())
        .fill_null(0.0)
        .sum()
        .over('Account_HASHED')
        .alias('entropy_value')
    ])
    
    entropy = txn_per_counterparty.select(['Account_HASHED', 'entropy_value']).unique()
    
    # Merge back to original
    df = df.join(entropy, on='Account_HASHED', how='left')
    
    df = df.with_columns([
        pl.col('entropy_value').fill_null(0.0).alias('counterparty_entropy_28d')
    ])
    
    return df.drop('entropy_value')


def compute_counterparty_switching_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect round-robin money laundering (switching between counterparties).
    
    If A->X->A->Y->A pattern, that's suspicious.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Counterparty switches
    df = df.with_columns([
        (pl.col('Account_duplicated_0').ne(
            pl.col('Account_duplicated_0').shift(1).over('Account_HASHED')
        ))
        .cast(pl.Int8)
        .fill_null(0)
        .alias('is_counterparty_switch')
    ])
    
    # 2. Switching rate (per 50 txns and 200 txns)
    df = df.with_columns([
        (pl.col('is_counterparty_switch') 
            .rolling_mean(window_size=50)
            .over('Account_HASHED')
            .alias('counterparty_switch_rate_50txns')),
        
        (pl.col('is_counterparty_switch')
            .rolling_sum(window_size=500)
            .over('Account_HASHED')
            .alias('total_counterparty_switches_28d')),
    ])
    
    # 3. Unique receiver diversity
    df = df.with_columns([
        pl.col('Account_duplicated_0')
        .n_unique()
        .over('Account_HASHED')
        .alias('num_unique_receivers_28d'),
    ])
    
    # 4. Recycling ratio (do we reuse the same N counterparties?)
    df = df.with_columns([
        (pl.col('num_unique_receivers_28d') / 
         (pl.col('total_counterparty_switches_28d') + 1.0))
        .alias('counterparty_recycling_ratio_28d')
    ])
    
    return df


def compute_network_balance_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Inflow vs outflow balance (money mule and pass-through detection).
    
    - Mule: High inflow, low outflow
    - Pass-through: inflow ≈ outflow
    - Collector: Many inflows, few outflows
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Balance metrics
    df = df.with_columns([
        pl.col('total_amount_received_28d') / 
        (pl.col('total_amount_paid_28d') + 0.000001)
        .alias('inflow_outflow_balance_28d'),
        
        (pl.col('total_amount_received_28d') - pl.col('total_amount_paid_28d'))
        .alias('net_flow_28d'),
        
        ((pl.col('total_amount_received_28d') + pl.col('total_amount_paid_28d')) / 2.0)
        .alias('average_flow_magnitude_28d'),
    ])
    
    # 2. Transaction count balance
    df = df.with_columns([
        (pl.col('txn_count_28d') / 
         (pl.col('txn_count_28d') + 0.000001))
        .alias('inflow_outflow_txn_ratio_28d')
    ])
    
    # 3. Pass-through detection (ratio close to 1)
    df = df.with_columns([
        (1.0 - 
         ((pl.col('inflow_outflow_balance_28d') - 1.0).abs() /
          (pl.col('inflow_outflow_balance_28d') + 1.0)))
        .alias('passthrough_likelihood_28d'),
        
        (pl.col('inflow_outflow_balance_28d') -
         pl.col('inflow_outflow_balance_28d').shift(1).over('Account_HASHED'))
        .fill_null(0.0)
        .abs()
        .alias('balance_volatility_daily'),
    ])
    
    return df


def compute_temporal_counterparty_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Time-of-day patterns in counterparty selection.
    
    Automated systems have consistent patterns (suspicious).
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Hour of day patterns
    df = df.with_columns([
        pl.col('Timestamp').dt.hour().alias('hour_of_day'),
        (pl.col('Timestamp').dt.hour() >= 17).cast(pl.Int8).alias('is_end_of_day_txn'),
    ])
    
    # 2. End-of-day clearing pattern (smurfing)
    df = df.with_columns([
        pl.col('is_end_of_day_txn')
            .rolling_sum(window_size=200)
            .over('Account_HASHED')
            .alias('end_of_day_txn_count_7d')
    ])
    
    return df


def compute_relationship_asymmetry(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect one-way relationships (A→B but B doesn't send to A).
    
    Typical of money mules and launderers.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Create directed pairs
    df = df.with_columns([
        pl.concat_str([
            pl.col('Account_HASHED'),
            pl.col('Account_duplicated_0')
        ], separator="|").alias('account_pair_directed'),
        
        # Create undirected pair (for reverse lookup)
        pl.concat_str([
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_HASHED'))
                .otherwise(pl.col('Account_duplicated_0')),
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_duplicated_0'))
                .otherwise(pl.col('Account_HASHED')),
        ], separator="|").alias('account_pair_undirected')
    ])
    
    # 2. Count transactions in each direction
    df = df.with_columns([
        pl.col('Timestamp')
        .count()
        .over('account_pair_directed')
        .alias('txns_in_directed_pair')
    ])
    
    # 3. Detect one-way relationships
    df = df.with_columns([
        (pl.col('txns_in_directed_pair') > 5).cast(pl.Int8)
            .rolling_sum(window_size=500)
            .over('account_pair_directed')
            .alias('asymmetric_pair_evidence_28d')
    ])
    
    # 4. Volume asymmetry
    df = df.with_columns([
        (pl.col('Amount Paid') / 
         (pl.col('Amount Paid').shift(-1).over('account_pair_directed') + 0.000001))
        .fill_null(1.0)
        .alias('amount_asymmetry_with_counterparty')
    ])
    
    return df


def compute_network_centrality_proxy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Proxy for hub-and-spoke network detection.
    
    Hubs (mules) have many connections as in/out nodes.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. In-degree and out-degree approximations
    df = df.with_columns([
        pl.col('Account_duplicated_0').n_unique()
        .over('Account_HASHED')
        .alias('out_degree_approximation'),
        
        pl.col('Account_HASHED').n_unique()
        .over('Account_duplicated_0')
        .alias('in_degree_approximation'),
    ])
    
    # 2. Hub score (high both in and out)
    df = df.with_columns([
        (pl.col('out_degree_approximation').cast(pl.Float64) * 
         pl.col('in_degree_approximation').cast(pl.Float64)).sqrt()
        .alias('betweenness_centrality_proxy')
    ])
    
    return df


def add_counterparty_entropy_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add all counterparty-based network features.
    
    Applies transformations in order and returns enhanced DataFrame.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("  Computing counterparty entropy...")
    df = compute_counterparty_entropy(df)
    
    logger.info("  Computing counterparty switching metrics...")
    df = compute_counterparty_switching_metrics(df)
    
    logger.info("  Computing network balance ratios...")
    df = compute_network_balance_ratios(df)
    
    logger.info("  Computing temporal counterparty patterns...")
    df = compute_temporal_counterparty_patterns(df)
    
    logger.info("  Computing relationship asymmetry...")
    df = compute_relationship_asymmetry(df)
    
    logger.info("  Computing network centrality proxy...")
    df = compute_network_centrality_proxy(df)
    
    return df
