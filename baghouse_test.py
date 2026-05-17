# =============================================================================
# BAGHOUSE FILTER — REAL-TIME TESTING / DEPLOYMENT SCRIPT
# Uses already-trained models for simulated live prediction
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model

# =============================================================================
# 1. LOAD SAVED MODELS
# =============================================================================

MODEL_DIR = r"C:\Users\anany\Downloads\baghouse_outputs"

print("=" * 60)
print("Loading saved models...")
print("=" * 60)

cnn_bilstm = load_model(f"{MODEL_DIR}/cnn_bilstm_model.keras")
embedding_model = load_model(f"{MODEL_DIR}/embedding_model.keras")

lgb_model = joblib.load(f"{MODEL_DIR}/lightgbm_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

print("All models loaded successfully!")

# =============================================================================
# 2. SETTINGS
# =============================================================================

SEQ_LEN = 60
ALPHA = 0.60

CSV_PATH = r"C:\Users\anany\Downloads\baghouse_filter_dataset.csv"

# =============================================================================
# 3. FEATURE ENGINEERING FUNCTION
# =============================================================================

def make_tabular_features(window_df):

    feats = {}

    feats["dp_mean"] = window_df["dp"].mean()
    feats["dp_std"] = window_df["dp"].std()
    feats["dp_max"] = window_df["dp"].max()
    feats["dp_min"] = window_df["dp"].min()
    feats["dp_range"] = feats["dp_max"] - feats["dp_min"]

    feats["dp_last"] = window_df["dp"].iloc[-1]
    feats["dp_trend"] = (
        window_df["dp"].iloc[-1] - window_df["dp"].iloc[0]
    )

    feats["dp_slope_mean"] = window_df["dp_slope"].mean()
    feats["dp_slope_max"] = window_df["dp_slope"].max()

    feats["dp_variance_mean"] = window_df["dp_variance"].mean()
    feats["dp_variance_max"] = window_df["dp_variance"].max()

    feats["dp_var_delta_max"] = window_df["dp_var_delta"].max()

    feats["instability_mean"] = window_df["instability"].mean()
    feats["instability_max"] = window_df["instability"].max()

    feats["pm_mean"] = window_df["pm"].mean()
    feats["pm_std"] = window_df["pm"].std()
    feats["pm_max"] = window_df["pm"].max()

    feats["pm_roll_max_last"] = window_df["pm_roll_max"].iloc[-1]

    feats["pm_zscore_max"] = window_df["pm_zscore"].max()

    feats["pm_spike_sum"] = window_df["pm_spike_flag"].sum()
    feats["pm_spike_rate"] = window_df["pm_spike_flag"].mean()

    feats["dp_pm_corr"] = window_df["dp"].corr(window_df["pm"])

    feats["dp_deviation_mean"] = window_df["dp_deviation"].mean()

    feats["dp_accel_max"] = window_df["dp_accel"].abs().max()

    return feats

# =============================================================================
# 4. REAL-TIME EARLY WARNING SYSTEM
# =============================================================================

class BaghouseEarlyWarningSystem:

    def __init__(self,
                 cnn_model,
                 emb_model,
                 lgbm_model,
                 scaler,
                 le):

        self.cnn = cnn_model
        self.emb = emb_model
        self.lgbm = lgbm_model
        self.scaler = scaler
        self.le = le

        self.buffer = []

        self.cls_names = list(le.classes_)

    # -------------------------------------------------------------------------

    def compute_features(self,
                         dp,
                         pm,
                         dp_rolling,
                         dp_slope,
                         dp_variance,
                         pm_spike):

        hist = self.buffer

        dp_var_delta = (
            dp_variance - hist[-1]["dp_variance"]
        ) if hist else 0

        dp_slope_delta = (
            dp_slope - hist[-1]["dp_slope"]
        ) if hist else 0

        pm_roll_max = max(
            [h["pm_val"] for h in hist[-6:]] + [pm]
        ) if hist else pm

        dp_deviation = dp - dp_rolling

        dp_accel = dp_slope_delta

        instability = dp_variance * abs(dp_slope)

        pm_vals = [h["pm_val"] for h in hist[-30:]] + [pm]

        pm_mean = np.mean(pm_vals)
        pm_std = max(np.std(pm_vals), 1e-3)

        pm_zscore = (pm - pm_mean) / pm_std

        return {
            "dp": dp,
            "pm_val": pm,
            "dp_slope": dp_slope,
            "dp_variance": dp_variance,
            "pm_spike_flag": pm_spike,
            "dp_rolling_mean": dp_rolling,
            "dp_var_delta": dp_var_delta,
            "dp_slope_delta": dp_slope_delta,
            "pm_roll_max": pm_roll_max,
            "dp_deviation": dp_deviation,
            "dp_accel": dp_accel,
            "instability": instability,
            "pm_zscore": pm_zscore
        }

    # -------------------------------------------------------------------------

    def buffer_to_arrays(self):

        win = self.buffer[-SEQ_LEN:]

        feature_order = [
            "dp",
            "pm_val",
            "dp_slope",
            "dp_variance",
            "pm_spike_flag",
            "dp_rolling_mean",
            "dp_var_delta",
            "dp_slope_delta",
            "pm_roll_max",
            "dp_deviation",
            "dp_accel",
            "instability",
            "pm_zscore"
        ]

        raw_seq = np.array([
            [h[f] for f in feature_order]
            for h in win
        ], dtype=np.float32)

        seq_scaled = self.scaler.transform(raw_seq)

        seq_in = seq_scaled[np.newaxis, :, :]

        tmp_df = pd.DataFrame(win).rename(columns={"pm_val": "pm"})

        tab = np.array(
            list(make_tabular_features(tmp_df).values()),
            dtype=np.float32
        ).reshape(1, -1)

        return seq_in, tab

    # -------------------------------------------------------------------------

    def ingest(self,
               dp,
               pm,
               dp_rolling,
               dp_slope,
               dp_variance,
               pm_spike):

        entry = self.compute_features(
            dp,
            pm,
            dp_rolling,
            dp_slope,
            dp_variance,
            pm_spike
        )

        self.buffer.append(entry)

        # -------------------------------------------------------------
        # Need 60 readings first
        # -------------------------------------------------------------

        if len(self.buffer) < SEQ_LEN:

            return {
                "condition": "Initializing",
                "warning_msg":
                f"Warming up ({len(self.buffer)}/{SEQ_LEN})"
            }

        # -------------------------------------------------------------
        # Prepare inputs
        # -------------------------------------------------------------

        seq_in, tab_in = self.buffer_to_arrays()

        # -------------------------------------------------------------
        # CNN prediction
        # -------------------------------------------------------------

        cnn_p = self.cnn.predict(
            [seq_in, tab_in],
            verbose=0
        )[0]

        # -------------------------------------------------------------
        # Embedding + LightGBM
        # -------------------------------------------------------------

        emb = self.emb.predict(
            [seq_in, tab_in],
            verbose=0
        )

        lgb_in = np.hstack([emb, tab_in])

        lgb_p = self.lgbm.predict_proba(lgb_in)[0]

        # -------------------------------------------------------------
        # Hybrid prediction
        # -------------------------------------------------------------

        hybrid_p = ALPHA * cnn_p + (1 - ALPHA) * lgb_p

        pred_idx = np.argmax(hybrid_p)

        pred_cls = self.cls_names[pred_idx]

        confidence = hybrid_p[pred_idx]

        # -------------------------------------------------------------
        # Message generation
        # -------------------------------------------------------------

        msg = {
            "Normal":
                f"✓ NORMAL | Confidence={confidence:.3f}",

            "Clogging":
                f"⚠ CLOGGING DETECTED | Confidence={confidence:.3f}",

            "Pre_Tear":
                f"🚨 PRE-TEAR WARNING | Confidence={confidence:.3f}",

            "Tearing":
                f"🔴 TEAR DETECTED | Confidence={confidence:.3f}"
        }

        return {
            "condition": pred_cls,
            "warning_msg": msg.get(pred_cls, "Unknown"),
            "probabilities": {
                c: round(float(p), 4)
                for c, p in zip(self.cls_names, hybrid_p)
            }
        }

# =============================================================================
# 5. LOAD DATASET
# =============================================================================

print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print(f"Dataset loaded: {len(df)} rows")

# =============================================================================
# 6. CREATE SYSTEM
# =============================================================================

ews = BaghouseEarlyWarningSystem(
    cnn_model=cnn_bilstm,
    emb_model=embedding_model,
    lgbm_model=lgb_model,
    scaler=scaler,
    le=le
)

# =============================================================================
# 7. SIMULATED REAL-TIME TESTING
# =============================================================================

print("\n" + "=" * 60)
print("STARTING REAL-TIME SIMULATION")
print("=" * 60)

for i, row in df.iterrows():

    result = ews.ingest(
        dp=row["dp"],
        pm=row["pm"],
        dp_rolling=row["dp_rolling_mean"],
        dp_slope=row["dp_slope"],
        dp_variance=row["dp_variance"],
        pm_spike=int(row["pm_spike_flag"])
    )

    print(f"\nTime Step: {i}")
    print(result["warning_msg"])

    if result["condition"] != "Initializing":

        print("Probabilities:")

        for cls, prob in result["probabilities"].items():

            print(f"  {cls:10s}: {prob:.4f}")

# =============================================================================
# 8. FINISHED
# =============================================================================

print("\n" + "=" * 60)
print("REAL-TIME TESTING COMPLETE")
print("=" * 60)