import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date

st.set_page_config(page_title="InBalance Cycle Predictor", layout="centered")
st.title("InBalance — Period Range Predictor (P10 / P50 / P90)")

MODEL_DIR = "models"

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    q10 = joblib.load(os.path.join(MODEL_DIR, "q10.joblib"))
    q50 = joblib.load(os.path.join(MODEL_DIR, "q50.joblib"))
    q90 = joblib.load(os.path.join(MODEL_DIR, "q90.joblib"))
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    return q10, q50, q90, cfg

# ---------- Helpers ----------
def calc_cycle_lengths(starts):
    """Cycle lengths from consecutive period starts."""
    return [(starts[i+1] - starts[i]).days for i in range(len(starts)-1)]

def range_mm_last6(lengths):
    if len(lengths) < 3:
        return np.nan
    last = lengths[-6:] if len(lengths) >= 6 else lengths
    return float(max(last) - min(last))

def is_irregular_last6(lengths, range_thresh=9, min_len=21, max_len=35):
    """Recent irregularity from last up to 6 cycle lengths."""
    r = range_mm_last6(lengths)
    if np.isnan(r):
        return False
    last = lengths[-6:] if len(lengths) >= 6 else lengths
    out_of = any((L < min_len or L > max_len) for L in last)
    return (r > range_thresh) or out_of

def build_feature_row(lengths, irregular_flag, feature_cols):
    len_lag1 = lengths[-1] if len(lengths) >= 1 else np.nan
    len_lag2 = lengths[-2] if len(lengths) >= 2 else np.nan
    len_lag3 = lengths[-3] if len(lengths) >= 3 else np.nan

    s = pd.Series(lengths, dtype="float")
    last3 = s.iloc[-3:] if len(s) >= 3 else s
    med3 = float(last3.median()) if len(last3) else np.nan
    mean3 = float(last3.mean()) if len(last3) else np.nan
    rmm = range_mm_last6(lengths)

    X = pd.DataFrame([{
        "len_lag1": len_lag1,
        "len_lag2": len_lag2,
        "len_lag3": len_lag3,
        "len_roll_median3": med3,
        "len_roll_mean3": mean3,
        "IsIrregular": bool(irregular_flag),
        "range_mm_last6": rmm
    }])

    # enforce exact ordering expected by model
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]

def predict_range_from_starts(starts, q10, q50, q90, cfg):
    """
    Predict next cycle length range and dates from period start dates (datetime list).
    """
    starts = sorted(pd.to_datetime(starts))
    lengths = calc_cycle_lengths(starts) if len(starts) >= 2 else []
    cycles_seen = len(starts)

    cold_max = cfg["cold_start_max_cycles"]
    cold_width = cfg["cold_width_days"]
    pop_med = float(cfg["pop_median_next_len"])
    alpha = float(cfg["alpha_irregular"])
    gap_max = float(cfg["gap_max_days"])
    feature_cols = cfg["feature_cols"]

    last_start = starts[-1].to_pydatetime()

    if cycles_seen <= cold_max or len(lengths) < 1:
        p10d, p50d, p90d = pop_med - cold_width, pop_med, pop_med + cold_width
        irr = False
    else:
        irr = is_irregular_last6(lengths)
        X = build_feature_row(lengths, irr, feature_cols)

        p10d = float(q10.predict(X)[0])
        p50d = float(q50.predict(X)[0])
        p90d = float(q90.predict(X)[0])
        p10d, p90d = min(p10d, p90d), max(p10d, p90d)

        # gap safeguard
        if lengths[-1] > gap_max:
            irr = True
            p10d = min(p10d, p50d - 10)
            p90d = max(p90d, p50d + 10)

        # irregular calibration widening
        if irr and alpha != 1.0:
            p10d = p50d - alpha * (p50d - p10d)
            p90d = p50d + alpha * (p90d - p50d)

    out = {
        "cycles_seen": cycles_seen,
        "cycle_lengths": lengths,
        "is_irregular": irr,
        "p10_days": p10d,
        "p50_days": p50d,
        "p90_days": p90d,
        "p10_date": (last_start + timedelta(days=int(round(p10d)))).date(),
        "p50_date": (last_start + timedelta(days=int(round(p50d)))).date(),
        "p90_date": (last_start + timedelta(days=int(round(p90d)))).date(),
        "last_start": last_start.date(),
    }
    return out

def calc_menses_lengths(period_ranges):
    """Inclusive menses length in days from (start_dt, end_dt)."""
    lengths = []
    for s, e in period_ranges:
        if s is None or e is None:
            lengths.append(None)
            continue
        days = (e - s).days + 1
        lengths.append(days if days > 0 else None)
    return lengths

def luteal_range(is_irregular: bool):
    """Luteal length prior (days)."""
    return (10, 16) if is_irregular else (11, 15)

def ovulation_and_fertile_from_next_period(pred_next_date: date, is_irregular: bool):
    """
    Estimate ovulation and fertile window from predicted next period date
    using luteal length range prior.
    """
    ll_min, ll_max = luteal_range(is_irregular)
    ll_mid = int(round((ll_min + ll_max) / 2))

    # Ovulation occurs ~luteal length days before next period
    ov_mid = pred_next_date - timedelta(days=ll_mid)
    ov_earliest = pred_next_date - timedelta(days=ll_max)  # earlier ovulation
    ov_latest = pred_next_date - timedelta(days=ll_min)    # later ovulation

    # Fertile window: 5 days before ovulation through 1 day after
    fertile_start = ov_earliest - timedelta(days=5)
    fertile_end = ov_latest + timedelta(days=1)

    return {
        "luteal_min": ll_min, "luteal_max": ll_max,
        "ov_mid": ov_mid,
        "ov_earliest": ov_earliest,
        "ov_latest": ov_latest,
        "fertile_start": fertile_start,
        "fertile_end": fertile_end
    }

# ---------- UI ----------
q10, q50, q90, cfg = load_artifacts()

st.caption("Log periods (start + end). We compute menses length + cycle lengths, then predict next period range and estimate ovulation & fertile window.")

n_periods = st.number_input("How many periods to log?", min_value=1, max_value=12, value=3, step=1)

period_ranges = []
for i in range(int(n_periods)):
    with st.expander(f"Period #{i+1}", expanded=(i == 0)):
        s = st.date_input("Start date", key=f"start_{i}")
        e = st.date_input("End date", key=f"end_{i}")

        start_dt = datetime.combine(s, datetime.min.time())
        end_dt = datetime.combine(e, datetime.min.time())

        if end_dt >= start_dt:
            period_ranges.append((start_dt, end_dt))
        else:
            st.warning("End date must be on/after start date.")

# sort + derive starts
period_ranges = sorted(period_ranges, key=lambda x: x[0])
starts = [s for s, _ in period_ranges]

# --- Derived “recording” section (always shown) ---
st.subheader("📊 Recorded metrics")

# Menses length
menses_lengths = calc_menses_lengths(period_ranges)
clean_menses = [x for x in menses_lengths if x is not None]
if clean_menses:
    st.write("Menses lengths (days):", clean_menses)
    st.write("Last menses length:", clean_menses[-1])
    st.write("Average menses length:", round(float(np.mean(clean_menses)), 2))
else:
    st.info("Enter valid start/end dates to compute menses length.")

# Cycle lengths + cycle day today
starts_sorted = sorted(pd.to_datetime(starts))
cycle_lengths = calc_cycle_lengths(starts_sorted) if len(starts_sorted) >= 2 else []
if cycle_lengths:
    st.write("Cycle lengths (days):", cycle_lengths)
    st.write("Last cycle length:", cycle_lengths[-1])
    st.write("Average cycle length:", round(float(np.mean(cycle_lengths)), 2))
else:
    st.info("Log at least 2 period starts to compute cycle length.")

if starts_sorted:
    last_start = starts_sorted[-1].to_pydatetime()
    cd = (datetime.today() - last_start).days + 1
    if cd < 1:
        st.warning("Today is before the last logged period start (check dates).")
    else:
        st.write("Cycle day today:", cd)

# --- Prediction ---
st.subheader("🧠 Predictions")

if st.button("Predict next period + ovulation + fertile window"):
    if len(starts) < 1:
        st.warning("Please log at least 1 period.")
    else:
        pred = predict_range_from_starts(starts, q10, q50, q90, cfg)

        st.write("Computed cycle lengths:", pred["cycle_lengths"] if pred["cycle_lengths"] else "Need ≥2 starts to compute lengths.")
        st.write("Irregular (recent):", pred["is_irregular"])

        st.success(f"Next period start (median/P50): {pred['p50_date'].isoformat()}")
        st.markdown(f"**Next period range (P10–P90):** {pred['p10_date'].isoformat()} → {pred['p90_date'].isoformat()}")
        st.caption(f"Predicted next length: P10={pred['p10_days']:.1f}d, P50={pred['p50_days']:.1f}d, P90={pred['p90_days']:.1f}d")

        # Ovulation + fertile window derived from predicted next period date (use P50 as center, P10/P90 to build range)
        ov_fert = ovulation_and_fertile_from_next_period(pred["p50_date"], pred["is_irregular"])

        # We can also compute a range based on next period P10/P90:
        ov_from_p10 = ovulation_and_fertile_from_next_period(pred["p10_date"], pred["is_irregular"])
        ov_from_p90 = ovulation_and_fertile_from_next_period(pred["p90_date"], pred["is_irregular"])

        ov_earliest = min(ov_from_p10["ov_earliest"], ov_from_p90["ov_earliest"])
        ov_latest   = max(ov_from_p10["ov_latest"], ov_from_p90["ov_latest"])

        fertile_start = ov_earliest - timedelta(days=5)
        fertile_end   = ov_latest + timedelta(days=1)

        st.markdown("### 🌼 Ovulation (estimated)")
        st.write("Most likely:", ov_fert["ov_mid"].isoformat())
        st.write("Range:", ov_earliest.isoformat(), "→", ov_latest.isoformat())
        st.caption(f"Luteal prior used: {ov_fert['luteal_min']}–{ov_fert['luteal_max']} days (wider if irregular)")

        st.markdown("### 🌿 Fertile window (estimated)")
        st.write(f"{fertile_start.isoformat()} → {fertile_end.isoformat()}")
