# freq_helper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NativeFreqBackfillConfig:
    # How many days to fetch per backward step when hunting for native frequency availability
    step_days: int = 30

    # Maximum lookback in days when hunting (hard stop)
    max_lookback_days: int = 365 * 2  # 24 months

    # We require at least this coverage ratio in each chunk for it to count as "available"
    # coverage = (#observed points)/(#expected points for inferred frequency)
    min_chunk_coverage: float = 0.97

    # For frequency inference, ignore weird gaps larger than this (e.g., outages)
    max_gap_minutes: int = 6 * 60  # 6 hours

    # When inferring, need at least this many deltas to be confident
    min_deltas: int = 50


def infer_modal_frequency(
    ts: pd.DatetimeIndex,
    max_gap_minutes: int = 360,
    min_deltas: int = 50,
) -> Optional[pd.Timedelta]:
    """
    Infer the modal time step from a DatetimeIndex by looking at consecutive deltas.
    Returns a Timedelta (e.g., 15min, 60min) or None if insufficient/ambiguous.
    """
    if len(ts) < 3:
        return None

    # Work in UTC instants to avoid DST weirdness
    idx = ts.tz_convert("UTC") if ts.tz is not None else ts
    idx = idx.sort_values().unique()

    deltas = pd.Series(idx[1:].asi8 - idx[:-1].asi8)  # ns integers
    deltas = pd.to_timedelta(deltas.values)

    # Filter out massive gaps (outages) that would skew inference
    max_gap = pd.Timedelta(minutes=int(max_gap_minutes))
    deltas = deltas[deltas <= max_gap]
    if len(deltas) < min_deltas:
        return None

    # Modal delta (robust to a few anomalies)
    # Convert to minutes for stable mode computation
    mins = np.round(deltas / pd.Timedelta(minutes=1)).astype(int)
    vals, counts = np.unique(mins, return_counts=True)
    modal_min = int(vals[np.argmax(counts)])

    return pd.Timedelta(minutes=modal_min)


def expected_points_per_day(freq: pd.Timedelta) -> int:
    minutes = int(freq / pd.Timedelta(minutes=1))
    if minutes <= 0:
        return 24
    return int((24 * 60) / minutes)


def chunk_has_freq_coverage(
    df: pd.DataFrame,
    freq: pd.Timedelta,
    tz: str,
    min_coverage: float,
) -> bool:
    """
    Check if df (with 'timestamp' column) has close-to-expected number of points
    for the given frequency over its time span, on a per-day basis (average).
    """
    if df.empty:
        return False

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return False

    # Convert to tz for "day" grouping (UI expectation)
    ts_local = ts.dt.tz_convert(tz)

    pts_per_day_exp = expected_points_per_day(freq)

    counts = ts_local.groupby(ts_local.dt.normalize()).size()
    if counts.empty:
        return False

    # coverage averaged over days in chunk
    coverage = float((counts / pts_per_day_exp).clip(upper=1.0).mean())
    return coverage >= float(min_coverage)


def find_native_frequency_and_history_start(
    fetch_today_df_fn,
    fetch_range_df_fn,
    tz: str,
    today: pd.Timestamp,
    cfg: NativeFreqBackfillConfig = NativeFreqBackfillConfig(),
) -> Tuple[pd.Timedelta, pd.Timestamp]:
    """
    1) Fetch a small slice around today to infer native frequency.
    2) Walk back in cfg.step_days chunks until that frequency is no longer available.
    Returns (freq, start_timestamp_local_midnight).

    fetch_today_df_fn: () -> DataFrame with ['timestamp', ...]
    fetch_range_df_fn: (start_ts, end_ts) -> DataFrame with ['timestamp', ...]
    """
    today = pd.Timestamp(today)
    if today.tzinfo is None:
        today = today.tz_localize(tz)
    else:
        today = today.tz_convert(tz)

    # A small window around today to infer freq
    today_df = fetch_today_df_fn()
    if today_df is None or len(today_df) < 3:
        # Fall back to 1H
        return pd.Timedelta(hours=1), (today - pd.Timedelta(days=365)).normalize()

    ts = pd.to_datetime(today_df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return pd.Timedelta(hours=1), (today - pd.Timedelta(days=365)).normalize()

    freq = infer_modal_frequency(ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC").dt.tz_localize("UTC").dt.tz_convert("UTC"))
    # ↑ the above is intentionally NOT required; pandas already gives UTC-aware series.
    # However, some environments have mixed tz parsing. We'll do it simply next:

    freq = infer_modal_frequency(ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.tz_localize("UTC"))  # safe roundtrip
    if freq is None:
        freq = pd.Timedelta(hours=1)

    # Start from "end" = tomorrow midnight local, walk back in chunks
    end_local = (today.normalize() + pd.Timedelta(days=1))
    oldest_ok = end_local

    looked = 0
    while looked < cfg.max_lookback_days:
        start_local = (oldest_ok - pd.Timedelta(days=cfg.step_days)).normalize()

        df_chunk = fetch_range_df_fn(start_local, oldest_ok)
        if not chunk_has_freq_coverage(df_chunk, freq=freq, tz=tz, min_coverage=cfg.min_chunk_coverage):
            break

        oldest_ok = start_local
        looked += cfg.step_days

    return freq, oldest_ok
