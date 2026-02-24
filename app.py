import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="InBalance Cycle Predictor", layout="centered")
st.title("InBalance — Period Range Predictor (P10 / P50 / P90)")

MODEL_DIR = "models"

@st.cache_resource
def load_artifacts():
    q10 = joblib.load(os.path.join(MODEL_DIR, "q10.joblib"))
    q50 = joblib.load(os.path.join(MODEL_DIR, "q50.joblib"))
    q90 = joblib.load(os.path.join(MODEL_DIR, "q90.joblib"))
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    return q10, q50, q90, cfg

def calc_cycle_lengths(starts):
    return [(starts[i+1] - starts[i]).days for i in range(len(starts)-1)]

def range_mm_last6(lengths):
    if len(lengths) < 3: return np.nan
    last = lengths[-6:] if len(lengths) >= 6 else lengths
    return float(max(last) - min(last))

def is_irregular_last6(lengths, range_thresh=9, min_len=21, max_len=35):
    r = range_mm_last6(lengths)
    if np.isnan(r): return False
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

def predict_range(starts, q10, q50, q90, cfg):
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
        "lengths": lengths,
        "is_irregular": irr,
        "p10_days": p10d,
        "p50_days": p50d,
        "p90_days": p90d,
        "p10_date": (last_start + timedelta(days=int(round(p10d)))).date().isoformat(),
        "p50_date": (last_start + timedelta(days=int(round(p50d)))).date().isoformat(),
        "p90_date": (last_start + timedelta(days=int(round(p90d)))).date().isoformat(),
    }
    return out

def calc_menses_lengths(period_ranges):
    """
    period_ranges: list of tuples (start_datetime, end_datetime) sorted by start
    returns list of menses lengths in days (inclusive)
    """
    lengths = []
    for s, e in period_ranges:
        if s is None or e is None:
            lengths.append(None)
            continue
        days = (e - s).days + 1
        lengths.append(days if days > 0 else None)
    return lengths
# ---------- UI ----------
q10, q50, q90, cfg = load_artifacts()

st.caption("Enter past period start dates. The model returns a predicted next start date range.")

n = st.number_input("How many period starts to enter?", min_value=1, max_value=12, value=3, step=1)

starts = []
cols = st.columns(2)
for i in range(int(n)):
    with st.expander(f"Period start #{i+1}", expanded=(i == 0)):
        d = st.date_input("Start date", key=f"s{i}")
        starts.append(datetime.combine(d, datetime.min.time()))

if st.button("Predict next period range"):
    pred = predict_range(starts, q10, q50, q90, cfg)

    st.write("Computed cycle lengths:", pred["lengths"] if pred["lengths"] else "Need ≥2 starts to compute lengths.")
    st.write("Irregular (recent):", pred["is_irregular"])

    st.success(f"Next start (median): {pred['p50_date']}")
    st.markdown(f"**Range (P10–P90):** {pred['p10_date']} → {pred['p90_date']}")
    st.caption(f"P10={pred['p10_days']:.1f} days, P50={pred['p50_days']:.1f}, P90={pred['p90_days']:.1f}")


period_ranges = []

n = st.number_input("How many periods to enter?", min_value=1, max_value=12, value=3, step=1)

for i in range(int(n)):
    with st.expander(f"Period #{i+1}", expanded=(i == 0)):
        s = st.date_input("Start date", key=f"start_{i}")
        e = st.date_input("End date", key=f"end_{i}")
        start_dt = datetime.combine(s, datetime.min.time())
        end_dt = datetime.combine(e, datetime.min.time())
        if end_dt >= start_dt:
            period_ranges.append((start_dt, end_dt))
        else:
            st.warning("End date must be on/after start date.")

period_ranges = sorted(period_ranges, key=lambda x: x[0])
starts = [s for s, _ in period_ranges]

menses_lengths = calc_menses_lengths(period_ranges)

st.subheader("🩸 Menses (bleeding) length")
if any(x is not None for x in menses_lengths):
    clean = [x for x in menses_lengths if x is not None]
    st.write("Menses lengths (days):", clean)
    st.write("Last menses length:", clean[-1])
    st.write("Average menses length:", round(sum(clean)/len(clean), 2))
else:
    st.info("Enter start and end for at least one period to compute menses length.")
