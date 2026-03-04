# lear_helper.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LearPrepConfig:
    freq: str = "1h"
    fill_method: str = "time"
    max_gap_hours: int = 6
    fallback_fill: str = "ffill_bfill"


DEFAULT_PREP = LearPrepConfig()


def _dedupe_and_sort_utc(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    df = df.dropna(subset=["price_eur_per_mwh"])

    df = (
        df.groupby("timestamp", as_index=False)["price_eur_per_mwh"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


def build_lear_hourly_df_from_prices(
    prices: pd.DataFrame,
    tz_for_display: str,
    config: LearPrepConfig = DEFAULT_PREP,
    force_freq: str = "1h",
    index_tz: str = "UTC",
    return_naive_utc_index: bool = True,
) -> pd.DataFrame:
    """
    Build LEAR-ready DataFrame with column 'Price' and an hourly index.

    IMPORTANT: epftoolbox expects tz-naive datetimes for slicing.
    We therefore build in UTC and drop tz info (UTC clock time) if requested.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(columns=["Price"])

    df = _dedupe_and_sort_utc(prices)

    ts = df["timestamp"]
    if index_tz.upper() == "UTC":
        ts = ts.dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert(index_tz)

    s = pd.Series(df["price_eur_per_mwh"].to_numpy(), index=ts).sort_index()

    # enforce hourly for LEAR
    s = s.resample(force_freq).mean()
    out = s.to_frame("Price")

    # full continuous grid (hourly)
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=config.freq, tz=out.index.tz)
    out = out.reindex(full_idx)

    # Fill gaps conservatively
    if out["Price"].isna().any():
        is_na = out["Price"].isna().astype(int)
        grp = (is_na.diff(1) != 0).cumsum()
        run_lengths = is_na.groupby(grp).sum()
        long_gap_groups = set(run_lengths[run_lengths > config.max_gap_hours].index)

        out["Price"] = out["Price"].interpolate(method=config.fill_method, limit_direction="both")

        if long_gap_groups:
            mask_long = grp.isin(long_gap_groups) & is_na.astype(bool)
            out.loc[mask_long.values, "Price"] = np.nan

        if config.fallback_fill == "ffill_bfill":
            out["Price"] = out["Price"].ffill().bfill()
        elif config.fallback_fill == "ffill":
            out["Price"] = out["Price"].ffill()
        elif config.fallback_fill == "bfill":
            out["Price"] = out["Price"].bfill()

    out.index.name = "timestamp"

    if return_naive_utc_index:
        out.index = out.index.tz_convert("UTC").tz_localize(None)

    return out


def day_start_utc_naive(day_in_display_tz: pd.Timestamp, display_tz: str) -> pd.Timestamp:
    d = pd.Timestamp(day_in_display_tz)
    if d.tzinfo is None:
        d = d.tz_localize(display_tz)
    else:
        d = d.tz_convert(display_tz)

    local_midnight = d.normalize()
    utc_midnight = local_midnight.tz_convert("UTC")
    return utc_midnight.tz_localize(None)


def pred_index_for_day(day_in_display_tz: pd.Timestamp, display_tz: str) -> pd.DatetimeIndex:
    d = pd.Timestamp(day_in_display_tz)
    if d.tzinfo is None:
        d = d.tz_localize(display_tz)
    else:
        d = d.tz_convert(display_tz)
    start = d.normalize()
    return pd.date_range(start, start + pd.Timedelta(hours=23), freq="1h", tz=display_tz)


def actual_day_series_from_utc_df(
    lear_df_utc_or_naive: pd.DataFrame, day_in_display_tz: pd.Timestamp, display_tz: str
) -> pd.Series:
    d = pd.Timestamp(day_in_display_tz)
    if d.tzinfo is None:
        d = d.tz_localize(display_tz)
    else:
        d = d.tz_convert(display_tz)

    start_local = d.normalize()
    end_local = start_local + pd.Timedelta(days=1)

    start_utc_aware = start_local.tz_convert("UTC")
    end_utc_aware = end_local.tz_convert("UTC")

    idx = lear_df_utc_or_naive.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is None:
        start_key = start_utc_aware.tz_localize(None)
        end_key = end_utc_aware.tz_localize(None)
        sub = lear_df_utc_or_naive.loc[(idx >= start_key) & (idx < end_key), "Price"].copy()
        if sub.empty:
            return pd.Series(dtype=float)
        sub.index = sub.index.tz_localize("UTC").tz_convert(display_tz)
        return sub

    sub = lear_df_utc_or_naive.loc[(idx >= start_utc_aware) & (idx < end_utc_aware), "Price"].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub.index = sub.index.tz_convert(display_tz)
    return sub


def validate_lear_coverage(
    lear_df_utc_naive: pd.DataFrame,
    next_day_date_utc_naive: pd.Timestamp,
    require_full_24h: bool = True,
) -> dict:
    if lear_df_utc_naive is None or lear_df_utc_naive.empty:
        return dict(ok=False, reason="lear_df is empty")

    df = lear_df_utc_naive.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        return dict(ok=False, reason="lear_df index is not a DatetimeIndex")

    if df.index.tz is not None:
        return dict(ok=False, reason="lear_df index must be tz-naive UTC for epftoolbox")

    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    target_start = pd.Timestamp(next_day_date_utc_naive).normalize()
    target_end = target_start + pd.Timedelta(days=1)

    full_target_idx = pd.date_range(target_start, target_end, freq="1h", inclusive="left")

    present = df.index.intersection(full_target_idx)
    missing = full_target_idx.difference(df.index)

    out = dict(
        ok=True,
        reason="ok",
        missing_hours=[t.isoformat() for t in missing],
        df_start=df.index.min().isoformat(),
        df_end=df.index.max().isoformat(),
        target_start=target_start.isoformat(),
        target_end=target_end.isoformat(),
        n_target_rows=int(len(present)),
    )

    if require_full_24h and len(present) < 24:
        out["ok"] = False
        out["reason"] = f"Next-day coverage incomplete: have {len(present)}/24 hourly points"

    return out


def ensure_full_target_day_for_lear(
    lear_df_utc_naive: pd.DataFrame,
    next_day_date_utc_naive: pd.Timestamp,
    max_missing_hours_to_fill: int = 2,
    fill_strategy: str = "ffill",
) -> tuple[pd.DataFrame, dict]:
    df = lear_df_utc_naive.copy()
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    target_start = pd.Timestamp(next_day_date_utc_naive).normalize()
    target_end = target_start + pd.Timedelta(days=1)
    full_target_idx = pd.date_range(target_start, target_end, freq="1h", inclusive="left")

    missing = full_target_idx.difference(df.index)

    info = dict(
        ok=True,
        target_start=target_start.isoformat(),
        target_end=target_end.isoformat(),
        missing_hours=[t.isoformat() for t in missing],
        filled_hours=[],
        reason="ok",
    )

    if len(missing) == 0:
        return df, info

    if len(missing) > max_missing_hours_to_fill:
        info["ok"] = False
        info["reason"] = f"Too many missing target hours ({len(missing)}) to auto-fill"
        return df, info

    add = pd.DataFrame(index=missing, columns=["Price"], dtype=float)
    df2 = pd.concat([df[["Price"]], add]).sort_index()

    if fill_strategy == "ffill":
        df2["Price"] = df2["Price"].ffill()
    elif fill_strategy == "bfill":
        df2["Price"] = df2["Price"].bfill()
    else:
        df2["Price"] = df2["Price"].ffill().bfill()

    info["filled_hours"] = [t.isoformat() for t in missing]
    info["reason"] = f"Filled {len(missing)} missing target-hour(s) using {fill_strategy}"
    return df2, info


def flatten_lear_prediction(pred) -> np.ndarray:
    """
    Normalize epftoolbox LEAR prediction to a strict 1D float array.

    - If pred is None (forecast skipped), returns empty array.
    - Handles shapes like (24,), (1,24), (24,1), etc.
    """
    if pred is None:
        return np.array([], dtype=float)

    arr = np.asarray(pred)

    # Special case: object scalar None-like
    if arr.shape == () and arr.dtype == object:
        if arr.item() is None:
            return np.array([], dtype=float)

    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    # If still scalar, reshape to (1,)
    if arr.shape == ():
        arr = np.array([arr], dtype=float)

    return arr.astype(float, copy=False)
