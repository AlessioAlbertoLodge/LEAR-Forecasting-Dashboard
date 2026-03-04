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


# -----------------------------------------------------------------------------
# Core preprocessing (hourly UTC-naive series for your plotting + "normal" LEAR use)
# -----------------------------------------------------------------------------
def _dedupe_and_sort_utc(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "price_eur_per_mwh" not in df.columns and "Price" in df.columns:
        df["price_eur_per_mwh"] = df["Price"]

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
    Returns DataFrame indexed hourly with column ['Price'].

    Safe for DST: index is UTC and (optionally) tz-naive.
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
    s = s.resample(force_freq).mean()

    out = s.to_frame("Price")
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=config.freq, tz=out.index.tz)
    out = out.reindex(full_idx)

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
    else:
        sub = lear_df_utc_or_naive.loc[(idx >= start_utc_aware) & (idx < end_utc_aware), "Price"].copy()
        if sub.empty:
            return pd.Series(dtype=float)
        sub.index = sub.index.tz_convert(display_tz)
        return sub


def flatten_lear_prediction(pred) -> np.ndarray:
    if pred is None:
        return np.array([], dtype=float)
    arr = np.asarray(pred)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape == (1, 24) or arr.shape == (24, 1):
            return arr.reshape(-1).astype(float)
        if arr.shape[0] == 24:
            return arr[:, 0].astype(float)
        if arr.shape[1] == 24:
            return arr[0, :].astype(float)
    return arr.reshape(-1).astype(float)


# -----------------------------------------------------------------------------
# Coverage validation + patching (hourly)
# -----------------------------------------------------------------------------
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
        return dict(ok=False, reason="lear_df index must be tz-naive UTC")

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
    info = dict(ok=True, target_start=None, target_end=None, missing_hours=[], filled_hours=[], reason="ok")
    if lear_df_utc_naive is None or lear_df_utc_naive.empty:
        info.update(ok=False, reason="lear_df is empty")
        return lear_df_utc_naive, info

    df = lear_df_utc_naive.sort_index().copy()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    target_start = pd.Timestamp(next_day_date_utc_naive).normalize()
    target_end = target_start + pd.Timedelta(days=1)
    info["target_start"] = target_start.isoformat()
    info["target_end"] = target_end.isoformat()

    full_target_idx = pd.date_range(target_start, target_end, freq="1h", inclusive="left")
    missing = full_target_idx.difference(df.index)
    info["missing_hours"] = [t.isoformat() for t in missing]

    if len(missing) == 0:
        return df, info
    if len(missing) > int(max_missing_hours_to_fill):
        info["ok"] = False
        info["reason"] = f"Too many missing hours in target day ({len(missing)}); not filling."
        return df, info

    add = pd.DataFrame(index=missing, columns=df.columns, dtype=float)
    df2 = pd.concat([df, add], axis=0).sort_index()

    if fill_strategy == "ffill":
        df2["Price"] = df2["Price"].ffill()
    elif fill_strategy == "bfill":
        df2["Price"] = df2["Price"].bfill()
    elif fill_strategy == "ffill_bfill":
        df2["Price"] = df2["Price"].ffill().bfill()
    else:
        raise ValueError("fill_strategy must be one of: ffill, bfill, ffill_bfill")

    info["filled_hours"] = info["missing_hours"]
    info["reason"] = f"Filled {len(missing)} missing target-hour(s) using {fill_strategy}"
    return df2, info


# -----------------------------------------------------------------------------
# ✅ THE IMPORTANT NEW PART: daily dataframe for epftoolbox internal splits
# -----------------------------------------------------------------------------
def hourly_to_epftoolbox_daily_df(
    lear_df_hourly_utc_naive: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert hourly UTC-naive df with column 'Price' to epftoolbox daily format:

      index: daily (00:00) tz-naive
      columns: Price_1 ... Price_24

    Requires that for each day we have 24 hourly points.
    """
    if lear_df_hourly_utc_naive is None or lear_df_hourly_utc_naive.empty:
        return pd.DataFrame()

    df = lear_df_hourly_utc_naive.copy()
    if "Price" not in df.columns:
        raise ValueError("Expected column 'Price' in hourly LEAR df.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Hourly df index must be DatetimeIndex.")
    if df.index.tz is not None:
        raise ValueError("Hourly df index must be tz-naive UTC (no tzinfo).")

    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # day key = midnight of each timestamp
    day = df.index.normalize()
    hour = df.index.hour

    tmp = pd.DataFrame({"day": day, "hour": hour, "Price": df["Price"].values})
    # pivot -> columns 0..23
    piv = tmp.pivot_table(index="day", columns="hour", values="Price", aggfunc="mean")

    # ensure 24 columns
    missing_cols = [h for h in range(24) if h not in piv.columns]
    for h in missing_cols:
        piv[h] = np.nan
    piv = piv[[h for h in range(24)]].sort_index()

    # rename to Price_1..Price_24
    piv.columns = [f"Price_{h+1}" for h in range(24)]
    piv.index.name = "timestamp"
    return piv


def build_epftoolbox_splits(
    model,
    lear_df_hourly_utc_naive: pd.DataFrame,
    next_day_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Xtrain, Ytrain, Xtest using epftoolbox private helper,
    BUT we must feed it the DAILY-format dataframe.

    This avoids the KeyErrors you saw (daily dates not found in hourly index).
    """
    if not hasattr(model, "_build_and_split_XYs"):
        raise AttributeError("LEAR model does not expose _build_and_split_XYs; epftoolbox version mismatch.")

    daily = hourly_to_epftoolbox_daily_df(lear_df_hourly_utc_naive)
    if daily.empty:
        raise ValueError("Daily dataframe conversion produced empty df (no data).")

    # epftoolbox expects date_test as python datetime (naive)
    date_test = pd.Timestamp(next_day_date).normalize()
    date_py = date_test.to_pydatetime()

    # split: train up to day-1, test includes day and prior days
    df_train = daily.loc[: date_test - pd.Timedelta(days=1)].copy()
    df_test = daily.loc[: date_test].copy()

    if df_train.empty:
        raise ValueError("df_train is empty after daily split. Need more history before next_day_date.")
    if df_test.empty:
        raise ValueError("df_test is empty after daily split.")

    Xtrain, Ytrain, Xtest = model._build_and_split_XYs(df_train=df_train, df_test=df_test, date_test=date_py)
    return Xtrain, Ytrain, Xtest
