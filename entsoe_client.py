# entsoe_client.py
from __future__ import annotations

from typing import Iterable

import pandas as pd

import entsoe
from entsoe.utils import extract_records, add_timestamps
from entsoe.Market import EnergyPrices


# Common EICs (extend as needed)
ZONES: dict[str, str] = {
    "ES (Spain)": "10YES-REE------0",
    "FR (France)": "10YFR-RTE------C",
    "DE-LU": "10Y1001A1001A82H",
    "NL (Netherlands)": "10YNL----------L",
    "BE (Belgium)": "10YBE----------2",
    "IT (Italy)": "10YIT-GRTN-----B",
    "SE (Sweden)": "10YSE-1--------K",
}


def configure_entsoe(api_key: str | None = None) -> None:
    """
    entsoe-apy reads ENTSOE_API from env automatically, but we allow explicit set (Streamlit secrets).
    """
    if api_key:
        entsoe.config.set_config(security_token=api_key)


def _ensure_tz(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    """
    Return a tz-aware Timestamp in tz.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def _to_entsoe_period_yyyymmddhhmm(ts: pd.Timestamp, tz: str) -> int:
    """
    ENTSO-E expects period_start/period_end as int YYYYMMDDHHMM in UTC.
    """
    ts = _ensure_tz(ts, tz).tz_convert("UTC")
    return int(ts.strftime("%Y%m%d%H%M"))


def _pick_column_by_suffix(columns: Iterable[str], preferred_suffixes: list[str]) -> str:
    cols = list(columns)

    # exact match first
    for s in preferred_suffixes:
        if s in cols:
            return s

    # suffix match
    for s in preferred_suffixes:
        for c in cols:
            if c.endswith(s):
                return c

    raise KeyError(
        f"Could not locate expected column. Tried {preferred_suffixes}. Available sample={cols[:40]}"
    )


def _normalize_timeseries_df(result, tz: str) -> pd.DataFrame:
    """
    Pydantic -> records -> timestamps -> DataFrame with tz-aware 'timestamp' column in tz.
    """
    records = extract_records(result)
    records = add_timestamps(records)
    df = pd.DataFrame(records)
    if "timestamp" not in df.columns:
        raise KeyError("add_timestamps did not add 'timestamp'.")

    # make it tz-aware and convert
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    return df


def query_day_ahead_prices_raw(eic: str, start: pd.Timestamp, end: pd.Timestamp, tz: str) -> pd.DataFrame:
    """
    Day-ahead prices (12.1.D) raw points.
    Output:
      timestamp (tz-aware in tz), price_eur_per_mwh
    """
    p0 = _to_entsoe_period_yyyymmddhhmm(start, tz)
    p1 = _to_entsoe_period_yyyymmddhhmm(end, tz)

    result = EnergyPrices(
        in_domain=eic,
        out_domain=eic,
        period_start=p0,
        period_end=p1,
    ).query_api()

    df = _normalize_timeseries_df(result, tz=tz)

    price_col = _pick_column_by_suffix(
        df.columns,
        preferred_suffixes=[
            "price_amount",
            "period.point.price_amount",
            "time_series.period.point.price_amount",
        ],
    )

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "price_eur_per_mwh": pd.to_numeric(df[price_col], errors="coerce"),
        }
    ).dropna(subset=["price_eur_per_mwh"])

    # Critical: de-duplicate identical instants (prevents duplicate index downstream)
    out = (
        out.groupby("timestamp", as_index=False)["price_eur_per_mwh"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def resample_prices(prices: pd.DataFrame, tz: str, freq: str = "1H", how: str = "mean") -> pd.DataFrame:
    """
    Resample raw prices to freq (used for LEAR which expects hourly).
    """
    if prices.empty:
        return prices

    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

    s = df.set_index("timestamp")["price_eur_per_mwh"].sort_index()

    if how == "last":
        s2 = s.resample(freq).last()
    else:
        s2 = s.resample(freq).mean()

    out = s2.reset_index().rename(columns={"index": "timestamp"})
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out
