"""
Counterparty Entropy and Network Ratio Features for AML Detection

This module computes sophisticated network and relationship features:
1. Shannon entropy of counterparty distributions (measures diversity)
2. Information gain from counterparty switching
3. Network reciprocity and balance metrics
4. Pass-through and mule detection ratios
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple


def compute_counterparty_entropy(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute Shannon entropy of counterparty distributions.
    
    Shannon Entropy: H = -sum(p_i * log(p_i))
    - H = 0: All volume to single counterparty (mule/collector)
    - H = log(N): Uniform distribution across N counterparties (diversified)
    - Legitimate users have moderate H; criminals have extreme values
    
    We approximate entropy using concentration metrics on lazy frame.
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Get unique counterparty counts per window
    df = df.with_columns([
        pl.col('Account_duplicated_0')
            .n_unique()
            .over(['Account_HASHED', pl.col('Timestamp').dt.truncate('7d')])
            .alias('num_unique_receivers_7d'),
        
        pl.col('Account_duplicated_0')
            .n_unique()
            .over(['Account_HASHED', pl.col('Timestamp').dt.truncate('28d')])
            .alias('num_unique_receivers_28d'),
        
        pl.col('Account')
            .n_unique()
            .over(['Account_duplicated_0', pl.col('Timestamp').dt.truncate('7d')])
            .alias('num_unique_senders_7d'),
        
        pl.col('Account')
            .n_unique()
            .over(['Account_duplicated_0', pl.col('Timestamp').dt.truncate('28d')])
            .alias('num_unique_senders_28d'),
    ])
    
    # 2. Approximate entropy indicators
    # For lazy evaluation, we use max counterparty (receiver with most txns)
    # as proxy for entropy extremity
    
    df = df.with_columns([
        # Concentration ratio: txns to top receiver / total txns
        # Computed as 1/num_unique (perfect diversity) to num_unique (perfect concentration)
        (1.0 / (pl.col('num_unique_receivers_28d') + 1))
            .alias('counterparty_concentration_ratio_28d'),
        
        # Maximum likelihood principle: if N senders but only 1 receiver,
        # suspect mule or collector (entropy approaching 0)
        ((pl.col('num_unique_senders_28d') - 1) 
         / (pl.col('num_unique_receivers_28d') + 1))
        .alias('sender_receiver_imbalance_28d'),
    ])
    
    # 3. Entropy-like "diversity score" (0 = concentrated, 1 = dispersed)
    # Using a sigmoid-like transformation: D = N_unique / (1 + N_unique)
    df = df.with_columns([
        (pl.col('num_unique_receivers_28d') / (1.0 + pl.col('num_unique_receivers_28d')))
            .alias('receiver_diversity_score_28d'),
        
        (pl.col('num_unique_senders_28d') / (1.0 + pl.col('num_unique_senders_28d')))
            .alias('sender_diversity_score_28d'),
    ])
    
    return df


def compute_counterparty_switching_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Measure counterparty switching patterns.
    
    Metrics:
    - How often account changes receiver (low = collector, high = distributor)
    - Switching entropy (randomness vs cyclic patterns)
    - Payoff structure recovery (detecting round-robin mule network)
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Count counterparty switches
    df = df.with_columns([
        (pl.col('Account_duplicated_0') 
         != pl.col('Account_duplicated_0').shift(1).over('Account_HASHED'))
        .fill_null(False)
        .cast(pl.Int8)
        .alias('is_counterparty_switch'),
        
        # Also track sender switches from counterparty perspective
        (pl.col('Account') 
         != pl.col('Account').shift(1).over('Account_duplicated_0'))
        .fill_null(False)
        .cast(pl.Int8)
        .alias('is_sender_switch'),
    ])
    
    # 2. Rolling switch frequency
    df = df.with_columns([
        # For sending account: how often they change receivers
        pl.col('is_counterparty_switch')
            .rolling_mean(window_size=20, min_periods=5)
            .over('Account_HASHED')
            .alias('counterparty_switch_frequency_20txn'),
        
        # For receiving account: how often they get new senders
        pl.col('is_sender_switch')
            .rolling_mean(window_size=20, min_periods=5)
            .over('Account_duplicated_0')
            .alias('sender_diversity_switching_rate_20txn'),
        
        # Cumulative switches (network maturity indicator)
        pl.col('is_counterparty_switch')
            .rolling_sum(window_size='28d')
            .over('Account_HASHED')
            .alias('total_counterparty_switches_28d'),
    ])
    
    # 3. Cyclic pattern detection
    # If account cycles through same N counterparties, it's round-robin mule
    df = df.with_columns([
        (pl.col('num_unique_receivers_28d') 
         / (pl.col('total_counterparty_switches_28d') + 1))
        .alias('counterparty_recycling_ratio_28d'),
    ])
    
    return df


def compute_network_balance_ratios(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute metrics for detecting pass-through and mule behaviors.
    
    Key insight: 
    - Mule incoming high, outgoing low (accumulator)
    - Pass-through: incoming ~= outgoing (transit)
    - Collector: many small incoming, few large outgoing
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Inflow-Outflow balance metrics
    df = df.with_columns([
        # Simple balance ratio
        (pl.col('total_amount_received_28d') / (pl.col('total_amount_paid_28d') + 1e-6))
            .clip(min_value=0.01, max_value=100)  # Clip extremes
            .alias('inflow_outflow_balance_28d'),
        
        # Net flow (amount accumulated)
        (pl.col('total_amount_received_28d') - pl.col('total_amount_paid_28d'))
            .alias('net_flow_28d'),
        
        # Flow intensity (total absolute flow)
        ((pl.col('total_amount_received_28d') + pl.col('total_amount_paid_28d')) / 2)
            .alias('average_flow_magnitude_28d'),
    ])
    
    # 2. Transaction count balance
    df = df.with_columns([
        # Inflow vs Outflow transaction ratio
        (pl.col('txn_count_28d') / (pl.col('txn_count_28d') + 1e-6))
            .clip(min_value=0.05, max_value=20)
            .alias('inflow_outflow_txn_ratio_28d'),
    ])
    
    # 3. Pass-through detection
    # Pass-through: B_in ≈ B_out (ratio close to 1) AND balance persists
    df = df.with_columns([
        # Proximity to 1:1 balance
        (1.0 - (pl.col('inflow_outflow_balance_28d') - 1.0).abs() / 
         (pl.col('inflow_outflow_balance_28d') + 1.0))
        .clip(min_value=0, max_value=1)
        .alias('passthrough_likelihood_28d'),
        
        # Volatility in balance (stable pass-through vs erratic)
        (pl.col('inflow_outflow_balance_28d') 
         - pl.col('inflow_outflow_balance_28d').shift(1).over('Account_HASHED'))
        .fill_null(0)
        .abs()
        .alias('balance_volatility_daily'),
    ])
    
    return df


def compute_temporal_counterparty_patterns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect temporal patterns in counterparty relationships.
    
    Patterns:
    - Temporal clustering (all txns to A on Monday, to B on Tuesday) => possible schedule
    - Fixed intervals between counterparty rotations
    - Morning vs evening switching
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Day-of-week patterns for each counterparty
    df = df.with_columns([
        pl.col('Timestamp').dt.weekday().alias('day_of_week'),
        pl.col('Timestamp').dt.hour().alias('hour_of_day'),
    ])
    
    # 2. Temporal switching patterns
    # Does account switch receivers at same time each day? (sign of automation)
    df = df.with_columns([
        # Time since last counterparty switch (in hours)
        ((pl.col('Timestamp') - pl.col('Timestamp').shift(1).over('Account_HASHED'))
         .dt.total_seconds() / 3600)
        .fill_null(0)
        .alias('hours_since_last_counterparty_switch'),
        
        # Regularity of switching (low std = clockwork)
        pl.col('hours_since_last_counterparty_switch')
            .rolling_std(window_size=20, min_periods=5)
            .over('Account_HASHED')
            .alias('switch_interval_regularity_std'),
    ])
    
    # 3. End-of-day clearing (smurfing indicator)
    df = df.with_columns([
        (pl.col('hour_of_day') >= 17).cast(pl.Int8).alias('is_end_of_day_txn'),
        
        # Count end-of-day txns (clearing house pattern)
        pl.col('is_end_of_day_txn')
            .rolling_sum(window_size='7d')
            .over('Account_HASHED')
            .alias('end_of_day_txn_count_7d'),
    ])
    
    return df


def compute_relationship_asymmetry(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Detect asymmetries in sender-receiver relationships.
    
    Asymmetry patterns:
    - A sends to B, but B never sends back (possible mule or collector)
    - Asymmetric amounts: A sends $100k to B, B sends $1k to A
    - One-directional volume patterns
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. For each account pair, track directionality
    df = df.with_columns([
        pl.concat_str([
            pl.col('Account_HASHED'),
            pl.col('Account_duplicated_0')
        ], sep="|").alias('account_pair_directed'),
        
        pl.concat_str([
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_HASHED'))
                .otherwise(pl.col('Account_duplicated_0')),
            pl.when(pl.col('Account_HASHED') < pl.col('Account_duplicated_0'))
                .then(pl.col('Account_duplicated_0'))
                .otherwise(pl.col('Account_HASHED'))
        ], sep="|").alias('account_pair_undirected')
    ])
    
    # 2. Count bidirectional relationships
    df = df.with_columns([
        pl.col('Account_HASHED')
            .count()
            .over('account_pair_directed')
            .alias('txns_in_directed_pair'),
    ])
    
    # 3. Detect one-way relationships in rolling window
    df = df.with_columns([
        # Heuristic: if pair X->Y has many txns but Y->X has 0, flag as asymmetric
        (pl.col('txns_in_directed_pair') > 5)
            .cast(pl.Int8)
            .rolling_sum(window_size='28d')
            .over('account_pair_directed')
            .alias('asymmetric_pair_evidence_28d'),
    ])
    
    # 4. Volume asymmetry
    df = df.with_columns([
        (pl.col('Amount Paid') / (pl.col('Amount Paid').shift(-1).over('account_pair_directed') + 1e-6))
            .fill_null(1.0)
            .clip(min_value=0.01, max_value=100)
            .alias('amount_asymmetry_with_counterparty'),
    ])
    
    return df


def compute_network_centrality_proxy(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Approximate network centrality metrics without expensive graph computation.
    
    Metrics:
    - Degree centrality proxy: counting unique neighbors
    - Betweenness proxy: appears between many other pairs
    - Closeness proxy: average distance to all neighbors
    """
    df = df.sort(['Account_HASHED', 'Timestamp'])
    
    # 1. Degree centrality
    df = df.with_columns([
        pl.col('Account_duplicated_0')
            .n_unique()
            .over('Account_HASHED')
            .alias('total_unique_receivers_all_time'),
        
        pl.col('Account')
            .n_unique()
            .over('Account_duplicated_0')
            .alias('total_unique_senders_all_time'),
    ])
    
    # 2. Degree rank (top 1%, top 5%, etc)
    df = df.with_columns([
        pl.col('total_unique_receivers_all_time')
            .rank(method='min')
            .over(pl.lit(1))
            .alias('receiver_degree_rank'),
    ])
    
    # 3. High-degree flag (hub/mule detection)
    df = df.with_columns([
        (pl.col('total_unique_receivers_all_time') > 100).cast(pl.Int8).alias('is_high_degree_sender'),
        (pl.col('total_unique_senders_all_time') > 100).cast(pl.Int8).alias('is_high_degree_receiver'),
    ])
    
    return df


# Main composition function
def add_counterparty_entropy_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply all counterparty entropy and network features.
    """
    print("  Computing counterparty entropy...")
    df = compute_counterparty_entropy(df)
    
    print("  Computing counterparty switching metrics...")
    df = compute_counterparty_switching_metrics(df)
    
    print("  Computing network balance ratios...")
    df = compute_network_balance_ratios(df)
    
    print("  Computing temporal counterparty patterns...")
    df = compute_temporal_counterparty_patterns(df)
    
    print("  Computing relationship asymmetry...")
    df = compute_relationship_asymmetry(df)
    
    print("  Computing network centrality proxies...")
    df = compute_network_centrality_proxy(df)
    
    return df
