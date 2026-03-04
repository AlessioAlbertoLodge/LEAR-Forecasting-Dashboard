# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from config import CONFIG, DEFAULT_TZ, get_api_key
from entsoe_client import (
    configure_entsoe,
    ZONES,
    query_day_ahead_prices_raw,
    resample_prices,
)
from epftoolbox.models import LEAR

from lear_helper import (
    build_lear_hourly_df_from_prices,
    day_start_utc_naive,
    pred_index_for_day,
    actual_day_series_from_utc_df,
    validate_lear_coverage,
    ensure_full_target_day_for_lear,
    flatten_lear_prediction,
)

st.set_page_config(page_title="ENTSO-E + LEAR Forecasting", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ ENTSO-E + LEAR")

zone_label = st.sidebar.selectbox(
    "Zone",
    options=list(ZONES.keys()) + ["Custom EIC…"],
    index=(list(ZONES.keys()).index(CONFIG.default_zone_label) if CONFIG.default_zone_label in ZONES else 0),
)
eic = st.sidebar.text_input("EIC (if custom)", value="") if zone_label == "Custom EIC…" else ZONES[zone_label]

tz = st.sidebar.selectbox("Display timezone", options=[DEFAULT_TZ, "UTC"], index=0)
months_back = st.sidebar.slider("Max lookback (months)", 1, 36, CONFIG.default_months_back)

freq_mode = st.sidebar.radio(
    "Training data frequency",
    options=[
        "Fixed 1H (max history)",
        "Use today's native resolution to decide history (may be shorter)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest anchor (\"today\")")

now = pd.Timestamp.now(tz=tz)
auto_today = now.normalize()

use_custom_today = st.sidebar.checkbox("Override 'today'", value=False)
if use_custom_today:
    chosen_today_date = st.sidebar.date_input("Choose 'today' date", value=auto_today.date())
    today = pd.Timestamp(chosen_today_date, tz=tz).normalize()
else:
    today = auto_today

st.sidebar.caption(f"Chosen 'today' = {today.date()} ({tz})")

# -----------------------------------------------------------------------------
# API Key
# -----------------------------------------------------------------------------
api_key = st.secrets.get("ENTSOE_API", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = get_api_key()

configure_entsoe(api_key=api_key)

if not api_key:
    st.sidebar.error("Missing ENTSOE_API token. Set env var ENTSOE_API or Streamlit secret ENTSOE_API.")
    st.stop()

if not eic:
    st.error("Pick a zone or provide a valid EIC.")
    st.stop()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("ENTSO-E Day-Ahead Prices — LEAR Forecasting")
st.caption(
    "LEAR is trained on an hourly UTC-naive index to avoid DST ambiguity; plots can still be shown in your chosen timezone. "
    "Training uses only data strictly before the chosen 'today'."
)

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
# Day X is "tomorrow" relative to the chosen 'today'
day_x = today + pd.Timedelta(days=1)
day_x1 = today + pd.Timedelta(days=2)

# IMPORTANT: LEAR needs the target day to exist in the dataframe for Xtest features.
# We'll fetch up to (day_x + 1 day) to enable Day X forecast.
# Day X+1 forecast: we want it always, but ENTSO-E likely won't publish it, so we "extend" the df by carrying forward Day X prices.
# We'll fetch until end of Day X (exclusive end = day_x + 1 day). That ensures we can validate/fill Day X properly.
end_fetch = day_x + pd.Timedelta(days=1)  # exclusive end

# -----------------------------------------------------------------------------
# Native-resolution helpers (used only to decide history window)
# -----------------------------------------------------------------------------
def infer_modal_delta_minutes(ts: pd.Series) -> int | None:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values().unique()
    if len(t) < 10:
        return None
    deltas = pd.Series(t[1:] - t[:-1])
    deltas = deltas[deltas <= pd.Timedelta(hours=6)]
    if len(deltas) < 10:
        return None
    mins = (deltas / pd.Timedelta(minutes=1)).round().astype(int)
    return int(mins.mode().iloc[0])


def coverage_ok(df: pd.DataFrame, tz: str, delta_min: int, min_cov: float = 0.97) -> bool:
    if df.empty:
        return False
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return False
    ts_local = ts.dt.tz_convert(tz)
    pts_per_day = int((24 * 60) / delta_min)
    counts = ts_local.groupby(ts_local.dt.normalize()).size()
    if counts.empty:
        return False
    cov = float((counts / pts_per_day).clip(upper=1.0).mean())
    return cov >= float(min_cov)


def determine_start_fetch_native(eic: str, tz: str, end_fetch: pd.Timestamp, months_back: int) -> tuple[pd.Timestamp, int]:
    # infer native resolution using recent 2 days before 'today'
    s0 = today - pd.Timedelta(days=2)
    e0 = today + pd.Timedelta(days=1)
    recent = query_day_ahead_prices_raw(eic, s0, e0, tz=tz)

    dm = infer_modal_delta_minutes(recent["timestamp"]) if not recent.empty else None
    if dm is None:
        dm = 60

    step_days = 30
    max_days = int(months_back * 31)

    end_local = end_fetch.normalize()
    oldest_ok = end_local
    looked = 0

    while looked < max_days:
        start_local = (oldest_ok - pd.Timedelta(days=step_days)).normalize()
        chunk = query_day_ahead_prices_raw(eic, start_local, oldest_ok, tz=tz)
        if not coverage_ok(chunk, tz=tz, delta_min=dm, min_cov=0.97):
            break
        oldest_ok = start_local
        looked += step_days

    return oldest_ok, dm


# -----------------------------------------------------------------------------
# Fetch window (history strictly before chosen "today", but includes Day X to forecast)
# -----------------------------------------------------------------------------
# Training must be strictly before 'today' => we will later filter the hourly series.
# But for forecasting Day X, we still need day_x hours in df (ENTSO-E published tomorrow).
if freq_mode.startswith("Fixed 1H"):
    start_fetch = (end_fetch - pd.DateOffset(months=int(months_back))).normalize()
    native_delta = 60
else:
    start_fetch, native_delta = determine_start_fetch_native(eic, tz=tz, end_fetch=end_fetch, months_back=int(months_back))

st.sidebar.caption(
    f"Fetch start: {start_fetch.date()} | Fetch end (exclusive): {end_fetch.date()} | Native step: {native_delta} min"
)

# -----------------------------------------------------------------------------
# Fetch raw prices
# -----------------------------------------------------------------------------
with st.spinner("Fetching ENTSO-E day-ahead prices…"):
    prices_raw = query_day_ahead_prices_raw(eic, start_fetch, end_fetch, tz=tz)

if prices_raw.empty:
    st.warning("No price data returned for this range.")
    st.stop()

# Always resample to 1h for LEAR
prices_1h = resample_prices(prices_raw, tz=tz, freq="1h", how="mean")

# Build UTC-naive hourly df
lear_df_utc = build_lear_hourly_df_from_prices(
    prices_1h,
    tz_for_display=tz,
    force_freq="1h",
    index_tz="UTC",
    return_naive_utc_index=True,
)

# -----------------------------------------------------------------------------
# Enforce: training strictly before chosen 'today' (UTC-naive cutoff)
# -----------------------------------------------------------------------------
today_utc_naive = day_start_utc_naive(today, display_tz=tz)  # UTC-naive midnight of chosen local today
# Keep all hours strictly before today_utc_naive, BUT we also need Day X for Xtest,
# so we'll keep a separate df_forecast that includes future hours (today..day_x).
lear_df_train_only = lear_df_utc.loc[lear_df_utc.index < today_utc_naive].copy()
lear_df_forecast = lear_df_utc.copy()

# -----------------------------------------------------------------------------
# Target dates for epftoolbox (UTC-naive midnight of local target day)
# -----------------------------------------------------------------------------
next_day_x_utc_naive = day_start_utc_naive(day_x, display_tz=tz)
next_day_x1_utc_naive = day_start_utc_naive(day_x1, display_tz=tz)

# -----------------------------------------------------------------------------
# Patch Day X if 1–2 hours missing (common when publication is incomplete)
# -----------------------------------------------------------------------------
lear_df_forecast, patch_info_x = ensure_full_target_day_for_lear(
    lear_df_forecast,
    next_day_date_utc_naive=next_day_x_utc_naive,
    max_missing_hours_to_fill=2,
    fill_strategy="ffill",
)
if patch_info_x.get("filled_hours"):
    st.warning(
        f"Day X had missing hourly points; filled {len(patch_info_x['filled_hours'])} hour(s) for LEAR input.",
        icon="⚠️",
    )
    with st.expander("Day X fill details"):
        st.write(patch_info_x)

diag_x = validate_lear_coverage(lear_df_forecast, next_day_x_utc_naive, require_full_24h=True)
if not diag_x.get("ok", False):
    st.error("LEAR input data is not sufficient to forecast Day X (tomorrow).")
    st.write(diag_x)
    st.stop()

# -----------------------------------------------------------------------------
# ALWAYS produce Day X+1 forecast:
# If ENTSO-E has not published Day X+1, we extend the dataframe by carrying Day X forward.
# This is only to satisfy epftoolbox's need for Xtest features for the next day.
# -----------------------------------------------------------------------------
diag_x1 = validate_lear_coverage(lear_df_forecast, next_day_x1_utc_naive, require_full_24h=True)
if not diag_x1.get("ok", False):
    # Create Day X+1 24h by copying Day X 24h (naive persistence extension)
    start_x = pd.Timestamp(next_day_x_utc_naive).normalize()
    end_x = start_x + pd.Timedelta(days=1)
    idx_x_utc = pd.date_range(start_x, end_x, freq="1h", inclusive="left")

    start_x1 = pd.Timestamp(next_day_x1_utc_naive).normalize()
    end_x1 = start_x1 + pd.Timedelta(days=1)
    idx_x1_utc = pd.date_range(start_x1, end_x1, freq="1h", inclusive="left")

    # Extract Day X prices from forecast df (should exist after patch)
    x_prices = lear_df_forecast.reindex(idx_x_utc)["Price"].copy()

    # If some are still missing, fill from last available (should be rare after diag_x ok)
    if x_prices.isna().any():
        x_prices = x_prices.ffill().bfill()

    # Build extension frame for Day X+1
    ext = pd.DataFrame({"Price": x_prices.values}, index=idx_x1_utc)
    ext.index.name = "timestamp"

    # Merge (do not overwrite existing if any)
    base = lear_df_forecast.copy()
    if base.index.has_duplicates:
        base = base[~base.index.duplicated(keep="last")]

    combined = pd.concat([base, ext], axis=0)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    lear_df_forecast = combined

    st.info(
        "ENTSO-E has not published Day X+1 yet. "
        "For Day X+1 forecasting we extend inputs by carrying Day X prices forward (to avoid empty Xtest)."
    )

    diag_x1_after = validate_lear_coverage(lear_df_forecast, next_day_x1_utc_naive, require_full_24h=True)
    with st.expander("Day X+1 diagnostics (before/after extension)"):
        st.write({"before": diag_x1, "after": diag_x1_after})

    if not diag_x1_after.get("ok", False):
        st.error("Could not construct a full 24h input grid for Day X+1 even after extension.")
        st.write(diag_x1_after)
        st.stop()

# -----------------------------------------------------------------------------
# Build a final dataframe to pass into epftoolbox:
# - Training should be strictly before chosen 'today'
# - Forecasting requires Day X and Day X+1 hours for Xtest construction
# We satisfy both by:
#   - taking training history from lear_df_train_only
#   - appending ONLY the required future hours (today..day_x1) from lear_df_forecast
# -----------------------------------------------------------------------------
# include future hours from (today_utc_naive) up to end of day_x1 (exclusive)
end_x1_excl = pd.Timestamp(next_day_x1_utc_naive).normalize() + pd.Timedelta(days=1)
future_slice = lear_df_forecast.loc[(lear_df_forecast.index >= today_utc_naive) & (lear_df_forecast.index < end_x1_excl)].copy()

lear_df_for_epf = pd.concat([lear_df_train_only, future_slice], axis=0).sort_index()
lear_df_for_epf = lear_df_for_epf[~lear_df_for_epf.index.duplicated(keep="last")]

# -----------------------------------------------------------------------------
# Calibration window
# -----------------------------------------------------------------------------
if lear_df_train_only.empty:
    st.error("Training dataframe is empty after applying 'today' cutoff. Choose an earlier 'today' or increase lookback.")
    st.stop()

history_days = int((lear_df_train_only.index.max() - lear_df_train_only.index.min()).days)
calibration_window_days = int(min(1092, max(30, history_days - 3)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calibration window (days)", f"{calibration_window_days}")
c2.metric("Training start (UTC)", f"{lear_df_train_only.index.min().date()}")
c3.metric("Training end (UTC)", f"{lear_df_train_only.index.max().date()}")
c4.metric("Raw points fetched", f"{len(prices_raw)}")

# -----------------------------------------------------------------------------
# Model + Forecasts
# -----------------------------------------------------------------------------
model = LEAR(calibration_window=calibration_window_days)

with st.spinner("Training LEAR and forecasting day X (tomorrow)…"):
    pred_day_x = model.recalibrate_and_forecast_next_day(
        df=lear_df_for_epf,
        calibration_window=calibration_window_days,
        next_day_date=next_day_x_utc_naive.to_pydatetime(),
    )

with st.spinner("Training LEAR and forecasting day X+1…"):
    pred_day_x1 = model.recalibrate_and_forecast_next_day(
        df=lear_df_for_epf,
        calibration_window=calibration_window_days,
        next_day_date=next_day_x1_utc_naive.to_pydatetime(),
    )

# -----------------------------------------------------------------------------
# Plot indices (display tz)
# -----------------------------------------------------------------------------
idx_x = pred_index_for_day(day_x, display_tz=tz)
idx_x1 = pred_index_for_day(day_x1, display_tz=tz)

pred_vec_x = flatten_lear_prediction(pred_day_x)
if len(pred_vec_x) != 24:
    st.error(f"LEAR returned unexpected prediction length for Day X: {len(pred_vec_x)} (expected 24).")
    st.write({"raw_shape": str(getattr(pred_day_x, "shape", None)), "type": str(type(pred_day_x))})
    st.stop()
pred_x = pd.Series(pred_vec_x, index=idx_x)

pred_vec_x1 = flatten_lear_prediction(pred_day_x1)
if len(pred_vec_x1) != 24:
    st.error(f"LEAR returned unexpected prediction length for Day X+1: {len(pred_vec_x1)} (expected 24).")
    st.write({"raw_shape": str(getattr(pred_day_x1, "shape", None)), "type": str(type(pred_day_x1))})
    st.stop()
pred_x1 = pd.Series(pred_vec_x1, index=idx_x1)

# -----------------------------------------------------------------------------
# Actual series for Day X from the true ENTSO-E-based hourly df (not the extended one)
# -----------------------------------------------------------------------------
actual_local_x = actual_day_series_from_utc_df(lear_df_utc, day_x, display_tz=tz)

# Align actual to 24h plotting grid
actual_on_grid_x = pd.Series(index=idx_x, dtype=float)
if not actual_local_x.empty:
    common = actual_local_x.index.intersection(idx_x)
    actual_on_grid_x.loc[common] = actual_local_x.loc[common].values

have_x = actual_on_grid_x.dropna()
if len(have_x) >= 20:
    aligned = pd.concat([have_x.rename("actual"), pred_x.rename("pred")], axis=1).dropna()
    mae = (aligned["pred"] - aligned["actual"]).abs().mean()
    rmse = ((aligned["pred"] - aligned["actual"]) ** 2).mean() ** 0.5
    st.success(f"Day X comparison available ({len(aligned)} hours). MAE={mae:.2f} €/MWh, RMSE={rmse:.2f} €/MWh")
else:
    st.info("Day X comparison not fully available yet (ENTSO-E may not have all hours). Showing forecast only.")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
st.subheader("Day X (tomorrow): forecast vs actual (if available)")
df_plot_x = pd.DataFrame({"timestamp": idx_x, "forecast": pred_x.values, "actual": actual_on_grid_x.values})
fig_x = px.line(df_plot_x, x="timestamp", y=["forecast", "actual"], title="Day X")
fig_x.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
st.plotly_chart(fig_x, use_container_width=True)

st.subheader("Day X+1 (day after tomorrow): forecast only")
df_plot_x1 = pd.DataFrame({"timestamp": idx_x1, "forecast": pred_x1.values})
fig_x1 = px.line(df_plot_x1, x="timestamp", y="forecast", title="Day X+1")
fig_x1.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
st.plotly_chart(fig_x1, use_container_width=True)

# -----------------------------------------------------------------------------
# Raw tables
# -----------------------------------------------------------------------------
with st.expander("Raw ENTSO-E (raw points)"):
    st.dataframe(prices_raw, use_container_width=True)

with st.expander("Hourly series used for LEAR (UTC-naive, as-fetched)"):
    tmp = lear_df_utc.copy().reset_index().rename(columns={"timestamp": "timestamp_utc_naive"})
    st.dataframe(tmp, use_container_width=True)

with st.expander("Hourly series passed to epftoolbox (UTC-naive, includes future inputs)"):
    tmp2 = lear_df_for_epf.copy().reset_index().rename(columns={"timestamp": "timestamp_utc_naive"})
    st.dataframe(tmp2, use_container_width=True)
