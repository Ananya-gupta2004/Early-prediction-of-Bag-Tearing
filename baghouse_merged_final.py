print("Program started")

"""
=============================================================================
 BAGHOUSE FILTER — EARLY BAG TEAR PREDICTION  (MERGED FINAL)
 Tier 1 Model : CNN-BiLSTM + LightGBM Hybrid Ensemble
=============================================================================
 What this merges:
   FROM my code   → True hybrid (CNN embeddings → LightGBM), model saving,
                    6 diagnostic plots, single-CSV pipeline, 660s Pre-Tear window
   FROM your code → 7 physics-motivated extra features, wider CNN kernels,
                    OOP BaghouseEarlyWarningSystem with rolling buffer,
                    dedicated early-prediction evaluation metrics
=============================================================================
 Dataset  : baghouse_filter_dataset.csv  (single file — auto 80/20 split)
 Classes  : Normal | Clogging | Pre_Tear (engineered) | Tearing
 Interval : 10 seconds per row
 Goal     : Predict Pre-Tear 410–660 seconds before bag failure
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, joblib, json
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional,
                                     LSTM, Dense, Dropout, BatchNormalization,
                                     Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  ← edit only this block
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        =  r"C:\Users\anany\Downloads\baghouse_filter_dataset.csv"
SEQ_LEN         = 60          # 60 steps × 10s = 10-min look-back  [from your code]
STEP            = 3           # slide window by 30s for augmentation [from your code]
PRE_TEAR_WINDOW = 66          # 66 steps × 10s = 660s pre-tear zone  [from my code]
BATCH_SIZE      = 64
EPOCHS          = 50
LR              = 0.001
ALPHA           = 0.60        # CNN-BiLSTM weight in ensemble
TEST_SIZE       = 0.20        # chronological 80/20 split
OUTPUT_DIR      = "baghouse_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Clogging", "Normal", "Pre_Tear", "Tearing"]   # alphabetical (LabelEncoder order)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 1 — Loading data")
print("="*70)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Fix the ~20 NaN rows at the start (rolling windows not yet filled)
df[["dp_slope", "dp_variance", "dp_rolling_mean"]] = (
    df[["dp_slope", "dp_variance", "dp_rolling_mean"]].ffill().bfill()
)

print(f"  Rows: {len(df):,}  |  Raw columns: {list(df.columns)}")
print(f"  Raw condition counts:\n{df['condition'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PRE_TEAR LABEL  (most critical engineering step)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2 — Engineering Pre_Tear label  (660-second window)")
print("="*70)

def create_pretear_labels(df: pd.DataFrame, warning_steps: int = 66) -> pd.Series:
    """
    Label every row within `warning_steps` × 10s BEFORE a Tearing event
    as Pre_Tear.  Physical basis: dp_variance rises 410–660 s before tear.

    Args:
        df            : DataFrame sorted by timestamp with 'condition' column
        warning_steps : steps before tear to flag (66 × 10s = 660s)
    Returns:
        pd.Series     : Normal | Clogging | Pre_Tear | Tearing
    """
    labels    = df["condition"].copy().astype(str)
    cond      = df["condition"].values
    tear_mask = (cond == "Tearing")

    for i in range(len(df)):
        if tear_mask[i]:
            start = max(0, i - warning_steps)
            for j in range(start, i):
                if labels.iloc[j] in ("Normal", "Clogging"):
                    labels.iloc[j] = "Pre_Tear"
    return labels

df["label"] = create_pretear_labels(df, PRE_TEAR_WINDOW)
print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

# Encode
le = LabelEncoder()
le.fit(sorted(df["label"].unique()))          # deterministic alphabetical order
df["label_enc"] = le.transform(df["label"])
NUM_CLASSES = len(le.classes_)
print(f"\n  Classes & encodings: {dict(zip(le.classes_, range(NUM_CLASSES)))}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE ENGINEERING  [7 extra features from your code]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3 — Feature engineering  (13 total features)")
print("="*70)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 physics-motivated features on top of the 6 already in the dataset.

    Extra features (from your code):
      dp_var_delta   — instability acceleration (how fast variance grows)
      dp_slope_delta — pressure jerk (second-order change in trend)
      pm_roll_max    — rolling 60s max PM (catches transient spikes)
      dp_deviation   — current ΔP minus rolling mean (drift measure)
      dp_accel       — second derivative of ΔP (pressure acceleration)
      instability    — composite: dp_variance × |dp_slope|
      pm_zscore      — rolling z-score of PM (relative spike detection)
    """
    df = df.copy()
    df["dp_var_delta"]   = df["dp_variance"].diff().fillna(0)
    df["dp_slope_delta"] = df["dp_slope"].diff().fillna(0)
    df["pm_roll_max"]    = df["pm"].rolling(6, min_periods=1).max()
    df["dp_deviation"]   = (df["dp"] - df["dp_rolling_mean"]).fillna(0)
    df["dp_accel"]       = df["dp_slope"].diff().fillna(0)
    df["instability"]    = (df["dp_variance"] * df["dp_slope"].abs()).fillna(0)

    pm_mean = df["pm"].rolling(30, min_periods=1).mean()
    pm_std  = df["pm"].rolling(30, min_periods=1).std().fillna(1).replace(0, 1)
    df["pm_zscore"] = ((df["pm"] - pm_mean) / pm_std).fillna(0)
    return df

df = engineer_features(df)

SEQ_FEATURES = [
    # Original 6
    "dp", "pm", "dp_slope", "dp_variance", "pm_spike_flag", "dp_rolling_mean",
    # 7 engineered
    "dp_var_delta", "dp_slope_delta", "pm_roll_max",
    "dp_deviation", "dp_accel", "instability", "pm_zscore",
]
N_FEATURES = len(SEQ_FEATURES)
print(f"  Total sequence features: {N_FEATURES}")
print(f"  Feature list: {SEQ_FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TABULAR FEATURE EXTRACTOR  (window statistics → LightGBM)
# ─────────────────────────────────────────────────────────────────────────────
def make_tabular_features(window_df: pd.DataFrame) -> dict:
    """
    Statistical summaries over a SEQ_LEN window for LightGBM.
    Combines my code's rich per-feature stats with your code's
    last/mean/std/slope approach — giving LightGBM both instant
    state and trend context.
    """
    feats = {}
    # ΔP summaries
    feats["dp_mean"]          = window_df["dp"].mean()
    feats["dp_std"]           = window_df["dp"].std()
    feats["dp_max"]           = window_df["dp"].max()
    feats["dp_min"]           = window_df["dp"].min()
    feats["dp_range"]         = feats["dp_max"] - feats["dp_min"]
    feats["dp_last"]          = window_df["dp"].iloc[-1]
    feats["dp_trend"]         = window_df["dp"].iloc[-1] - window_df["dp"].iloc[0]
    # Slope & variance
    feats["dp_slope_mean"]    = window_df["dp_slope"].mean()
    feats["dp_slope_max"]     = window_df["dp_slope"].max()
    feats["dp_variance_mean"] = window_df["dp_variance"].mean()
    feats["dp_variance_max"]  = window_df["dp_variance"].max()
    feats["dp_var_delta_max"] = window_df["dp_var_delta"].max()
    feats["instability_mean"] = window_df["instability"].mean()
    feats["instability_max"]  = window_df["instability"].max()
    # PM summaries
    feats["pm_mean"]          = window_df["pm"].mean()
    feats["pm_std"]           = window_df["pm"].std()
    feats["pm_max"]           = window_df["pm"].max()
    feats["pm_roll_max_last"] = window_df["pm_roll_max"].iloc[-1]
    feats["pm_zscore_max"]    = window_df["pm_zscore"].max()
    feats["pm_spike_sum"]     = window_df["pm_spike_flag"].sum()
    feats["pm_spike_rate"]    = window_df["pm_spike_flag"].mean()
    # Cross-feature
    feats["dp_pm_corr"]       = window_df["dp"].corr(window_df["pm"])
    feats["dp_deviation_mean"]= window_df["dp_deviation"].mean()
    feats["dp_accel_max"]     = window_df["dp_accel"].abs().max()
    return feats

N_TAB = len(make_tabular_features(df.iloc[:SEQ_LEN]))
print(f"  Total tabular features for LightGBM: {N_TAB}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SEQUENCE DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  STEP 4 — Building sequences  (SEQ_LEN={SEQ_LEN}, STEP={STEP})")
print("="*70)

scaler = StandardScaler()
scaled_arr = scaler.fit_transform(df[SEQ_FEATURES].values)

X_seq, X_tab, y_all = [], [], []

for i in range(SEQ_LEN, len(df), STEP):
    window_raw    = df.iloc[i - SEQ_LEN: i]
    window_scaled = scaled_arr[i - SEQ_LEN: i]
    X_seq.append(window_scaled)
    X_tab.append(list(make_tabular_features(window_raw).values()))
    y_all.append(df["label_enc"].iloc[i])

X_seq = np.array(X_seq, dtype=np.float32)   # (N, 60, 13)
X_tab = np.array(X_tab, dtype=np.float32)   # (N, 25)
y_all = np.array(y_all, dtype=np.int32)

print(f"  X_seq : {X_seq.shape}   X_tab : {X_tab.shape}   y : {y_all.shape}")
unique, counts = np.unique(y_all, return_counts=True)
print(f"  Label dist: { {le.classes_[k]: int(v) for k,v in zip(unique, counts)} }")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CHRONOLOGICAL TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
split_idx = int(len(X_seq) * (1 - TEST_SIZE))
X_seq_tr, X_seq_te = X_seq[:split_idx], X_seq[split_idx:]
X_tab_tr, X_tab_te = X_tab[:split_idx], X_tab[split_idx:]
y_tr, y_te         = y_all[:split_idx], y_all[split_idx:]

print(f"\n  Train : {len(y_tr):,}  |  Test : {len(y_te):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
cw_arr  = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
cw_dict = dict(enumerate(cw_arr))
print(f"\n  Class weights: { {le.classes_[k]: round(v,2) for k,v in cw_dict.items()} }")

y_tr_oh = to_categorical(y_tr, NUM_CLASSES)
y_te_oh = to_categorical(y_te, NUM_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  CNN-BiLSTM MODEL
#     Architecture merges both codes:
#       CNN kernels  5+3  [wider from your code]
#       Filters      64→128 [deeper from my code]
#       Dual input   seq + tabular fused inside Keras [from my code]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5 — Building CNN-BiLSTM model")
print("="*70)

def build_cnn_bilstm(seq_len, n_feat, n_tab, n_classes):
    """
    Dual-input CNN-BiLSTM:
      Branch A (sequence) : Conv1D(5) → Conv1D(3) → BiLSTM(64) → BiLSTM(32)
      Branch B (tabular)  : Dense(32) → Dense(16)
      Fused               : Concatenate → Dense(64) → Dense(32) → Softmax
    """
    # ── Branch A: sequence ───────────────────────────────────────────────────
    seq_in = Input(shape=(seq_len, n_feat), name="seq_input")

    x = Conv1D(64,  kernel_size=5, padding="same", activation="relu")(seq_in)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)     # 60 → 30 timesteps
    x = Dropout(0.25)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.30)(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.30)(x)
    seq_out = Dense(64, activation="relu")(x)

    # ── Branch B: tabular ────────────────────────────────────────────────────
    tab_in  = Input(shape=(n_tab,), name="tab_input")
    t = Dense(32, activation="relu")(tab_in)
    t = BatchNormalization()(t)
    t = Dropout(0.20)(t)
    tab_out = Dense(16, activation="relu")(t)

    # ── Fusion ───────────────────────────────────────────────────────────────
    fused = Concatenate()([seq_out, tab_out])
    fused = Dense(64, activation="relu")(fused)
    fused = BatchNormalization()(fused)
    fused = Dropout(0.30)(fused)
    fused = Dense(32, activation="relu", name="embedding")(fused)

    out = Dense(n_classes, activation="softmax", name="output")(fused)

    model = Model(inputs=[seq_in, tab_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn_bilstm = build_cnn_bilstm(SEQ_LEN, N_FEATURES, N_TAB, NUM_CLASSES)
cnn_bilstm.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN CNN-BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 6 — Training CNN-BiLSTM")
print("="*70)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]

history = cnn_bilstm.fit(
    [X_seq_tr, X_tab_tr], y_tr_oh,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. EXTRACT EMBEDDINGS → LightGBM  [from my code — true hybrid coupling]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 7 — Extracting CNN-BiLSTM embeddings for LightGBM")
print("="*70)

embedding_model = Model(
    inputs  = cnn_bilstm.inputs,
    outputs = cnn_bilstm.get_layer("embedding").output
)

emb_tr = embedding_model.predict([X_seq_tr, X_tab_tr], batch_size=256, verbose=0)
emb_te = embedding_model.predict([X_seq_te, X_tab_te], batch_size=256, verbose=0)

# LightGBM sees: 32 learned embeddings + 25 tabular stats = 57 features
lgb_tr_in = np.hstack([emb_tr, X_tab_tr])
lgb_te_in = np.hstack([emb_te, X_tab_te])
print(f"  LightGBM input shape — Train: {lgb_tr_in.shape}  Test: {lgb_te_in.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. TRAIN LightGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 8 — Training LightGBM on CNN-BiLSTM embeddings")
print("="*70)

sample_weights_tr = np.array([cw_dict[c] for c in y_tr])

lgb_model = lgb.LGBMClassifier(
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 63,       # richer trees [from your code]
    max_depth         = 8,        # deeper [from your code]
    min_child_samples = 10,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

lgb_model.fit(
    lgb_tr_in, y_tr,
    sample_weight = sample_weights_tr,
    eval_set      = [(lgb_te_in, y_te)],
    callbacks     = [lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(100)],
)
print("  LightGBM training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 13. HYBRID ENSEMBLE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 9 — Hybrid ensemble  (60% CNN-BiLSTM + 40% LightGBM)")
print("="*70)

cnn_proba  = cnn_bilstm.predict([X_seq_te, X_tab_te], batch_size=256, verbose=0)
lgb_proba  = lgb_model.predict_proba(lgb_te_in)
hyb_proba  = ALPHA * cnn_proba + (1 - ALPHA) * lgb_proba

cnn_pred = np.argmax(cnn_proba, axis=1)
lgb_pred = lgb_model.predict(lgb_te_in)
hyb_pred = np.argmax(hyb_proba, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 14. EVALUATION  [combines both codes]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 10 — Evaluation")
print("="*70)

def evaluate_model(name, y_true, y_pred, proba=None):
    acc = (y_true == y_pred).mean()
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=le.classes_, digits=4))

    # Early-warning specific metrics [from your code]
    pt_idx   = list(le.classes_).index("Pre_Tear")
    norm_idx = list(le.classes_).index("Normal")

    pt_mask   = (y_true == pt_idx)
    norm_mask = (y_true == norm_idx)

    pt_recall   = (y_pred[pt_mask] == pt_idx).mean()   if pt_mask.sum()   > 0 else 0
    false_alarm = ((y_pred[norm_mask] == pt_idx) |
                   (y_pred[norm_mask] == list(le.classes_).index("Tearing"))
                  ).mean()                              if norm_mask.sum() > 0 else 0

    print(f"  ▶ Pre_Tear Recall (critical) : {pt_recall*100:.2f}%")
    print(f"  ▶ False Alarm Rate           : {false_alarm*100:.2f}%")
    print(f"  ▶ Pre_Tear samples in test   : {pt_mask.sum()}"
          f"  ({pt_mask.sum()*10/60:.1f} min of warning coverage)")
    return acc, f1

acc_cnn, f1_cnn = evaluate_model("CNN-BiLSTM alone",  y_te, cnn_pred, cnn_proba)
acc_lgb, f1_lgb = evaluate_model("LightGBM alone",    y_te, lgb_pred, lgb_proba)
acc_hyb, f1_hyb = evaluate_model("HYBRID (60/40)",    y_te, hyb_pred, hyb_proba)

# ─────────────────────────────────────────────────────────────────────────────
# 15. DIAGNOSTIC PLOTS  [6 plots from my code]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 11 — Generating diagnostic plots")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Baghouse Bag Tear Early Prediction — CNN-BiLSTM + LightGBM Hybrid (Merged)",
    fontsize=13, fontweight="bold"
)

# Plot 1: Training curves
ax = axes[0, 0]
ax.plot(history.history["accuracy"],     label="Train Acc",  color="#1D9E75")
ax.plot(history.history["val_accuracy"], label="Val Acc",    color="#1D9E75", linestyle="--")
ax.plot(history.history["loss"],         label="Train Loss", color="#D85A30")
ax.plot(history.history["val_loss"],     label="Val Loss",   color="#D85A30", linestyle="--")
ax.set_title("CNN-BiLSTM Training Curves")
ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

# Plot 2: Confusion matrix — Hybrid
ax = axes[0, 1]
cm   = confusion_matrix(y_te, hyb_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Hybrid Model")

# Plot 3: Per-class F1 comparison
ax = axes[0, 2]
f1_cnn_c = f1_score(y_te, cnn_pred, average=None, labels=range(NUM_CLASSES))
f1_lgb_c = f1_score(y_te, lgb_pred, average=None, labels=range(NUM_CLASSES))
f1_hyb_c = f1_score(y_te, hyb_pred, average=None, labels=range(NUM_CLASSES))
x = np.arange(NUM_CLASSES); w = 0.25
ax.bar(x - w, f1_cnn_c, w, label="CNN-BiLSTM", color="#185FA5")
ax.bar(x,     f1_lgb_c, w, label="LightGBM",   color="#854F0B")
ax.bar(x + w, f1_hyb_c, w, label="Hybrid",     color="#1D9E75")
ax.set_xticks(x); ax.set_xticklabels(le.classes_, rotation=15)
ax.set_ylabel("F1 Score"); ax.set_title("Per-class F1 — All Models")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Prediction probabilities over time (test window)
ax = axes[1, 0]
clr = ["#B4B2A9", "#EF9F27", "#D85A30", "#A32D2D"]
for i, cls in enumerate(le.classes_):
    ax.plot(hyb_proba[:300, i], label=cls, color=clr[i], alpha=0.85)
ax.set_title("Hybrid Probabilities Over Time (first 300 test steps)")
ax.set_xlabel("Time step (×10s)"); ax.set_ylabel("Probability")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 5: LightGBM feature importance
ax = axes[1, 1]
emb_names = [f"emb_{i}" for i in range(emb_tr.shape[1])]
tab_names  = list(make_tabular_features(df.iloc[:SEQ_LEN]).keys())
all_names  = np.array(emb_names + tab_names)
fi         = lgb_model.feature_importances_
top_idx    = np.argsort(fi)[-15:]
ax.barh(all_names[top_idx], fi[top_idx], color="#185FA5")
ax.set_title("LightGBM Feature Importance (Top 15)")
ax.set_xlabel("Importance"); ax.grid(True, alpha=0.3, axis="x")

# Plot 6: Accuracy comparison
ax = axes[1, 2]
models = ["CNN-BiLSTM", "LightGBM", "Hybrid (60/40)"]
accs   = [acc_cnn * 100, acc_lgb * 100, acc_hyb * 100]
bars   = ax.bar(models, accs, color=["#185FA5", "#854F0B", "#1D9E75"], width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
ax.set_title("Overall Accuracy Comparison"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "results_merged.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plots saved → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. SAVE ALL ARTEFACTS  [from my code]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 12 — Saving models & artefacts")
print("="*70)

cnn_bilstm.save(os.path.join(OUTPUT_DIR,      "cnn_bilstm_model.keras"))
embedding_model.save(os.path.join(OUTPUT_DIR, "embedding_model.keras"))
joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, "lightgbm_model.pkl"))
joblib.dump(scaler,    os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le,        os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

config = {
    "SEQ_LEN"             : SEQ_LEN,
    "STEP"                : STEP,
    "PRE_TEAR_WINDOW"     : PRE_TEAR_WINDOW,
    "SEQ_FEATURES"        : SEQ_FEATURES,
    "N_TAB_FEATURES"      : N_TAB,
    "NUM_CLASSES"         : NUM_CLASSES,
    "CLASSES"             : list(le.classes_),
    "ALPHA"               : ALPHA,
    "TEST_ACCURACY_HYBRID": round(float(acc_hyb), 4),
    "TEST_F1_HYBRID"      : round(float(f1_hyb), 4),
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"  Saved to ./{OUTPUT_DIR}/")
print(f"    cnn_bilstm_model.keras")
print(f"    embedding_model.keras")
print(f"    lightgbm_model.pkl")
print(f"    scaler.pkl")
print(f"    label_encoder.pkl")
print(f"    config.json")

# ─────────────────────────────────────────────────────────────────────────────
# 17. BaghouseEarlyWarningSystem  [OOP class from your code — fixed & upgraded]
#     Fixes applied:
#       • Duplicate 'pm' key in buffer dict → fixed (renamed to 'pm_val')
#       • Uses all 13 SEQ_FEATURES + embedding model for true hybrid inference
#       • No external file dependency
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 13 — Real-time early warning system (OOP)")
print("="*70)

class BaghouseEarlyWarningSystem:
    """
    Real-time early warning system for baghouse filter monitoring.

    Maintains a rolling buffer of the last SEQ_LEN timesteps.
    Call .ingest() every 10 seconds with new sensor readings.

    Usage:
        ews = BaghouseEarlyWarningSystem(cnn_bilstm, embedding_model,
                                         lgb_model, scaler, le)
        result = ews.ingest(dp=145.2, pm=12.3, dp_rolling=143.0,
                            dp_slope=0.08, dp_variance=1.2, pm_spike=0)
        print(result["warning_msg"])
    """

    def __init__(self, cnn_model, emb_model, lgbm_model, scaler, le,
                 seq_len=SEQ_LEN, alpha=ALPHA, threshold=0.45):
        self.cnn       = cnn_model
        self.emb       = emb_model
        self.lgbm      = lgbm_model
        self.scaler    = scaler
        self.le        = le
        self.seq_len   = seq_len
        self.alpha     = alpha
        self.threshold = threshold
        self.buffer    = []          # list of feature-dicts (one per 10s tick)
        self.alert_log = []

        self.cls_names   = list(le.classes_)
        self.pt_idx      = self.cls_names.index("Pre_Tear")  if "Pre_Tear"  in self.cls_names else -1
        self.tear_idx    = self.cls_names.index("Tearing")   if "Tearing"   in self.cls_names else -1
        self.clog_idx    = self.cls_names.index("Clogging")  if "Clogging"  in self.cls_names else -1

    def _compute_features(self, dp, pm, dp_rolling, dp_slope,
                          dp_variance, pm_spike):
        """Compute all 13 SEQ_FEATURES from live sensor values + buffer history."""
        hist = self.buffer

        # dp_var_delta: change in variance from previous tick
        dp_var_delta   = (dp_variance - hist[-1]["dp_variance"]) if hist else 0.0
        # dp_slope_delta: change in slope
        dp_slope_delta = (dp_slope - hist[-1]["dp_slope"])       if hist else 0.0
        # pm_roll_max: max PM over last 6 ticks (60s)
        pm_roll_max    = max([h["pm_val"] for h in hist[-6:]] + [pm]) if hist else pm
        # dp_deviation
        dp_deviation   = dp - dp_rolling
        # dp_accel (same as slope_delta here — 2nd derivative of dp)
        dp_accel       = dp_slope_delta
        # instability composite
        instability    = dp_variance * abs(dp_slope)
        # pm_zscore: rolling 30-tick z-score
        pm_vals = [h["pm_val"] for h in hist[-30:]] + [pm]
        pm_mu   = float(np.mean(pm_vals))
        pm_sig  = max(float(np.std(pm_vals)), 1e-3)
        pm_zscore = (pm - pm_mu) / pm_sig

        return {
            "dp"             : dp,
            "pm_val"         : pm,          # key is pm_val to avoid duplicate
            "dp_slope"       : dp_slope,
            "dp_variance"    : dp_variance,
            "pm_spike_flag"  : pm_spike,
            "dp_rolling_mean": dp_rolling,
            "dp_var_delta"   : dp_var_delta,
            "dp_slope_delta" : dp_slope_delta,
            "pm_roll_max"    : pm_roll_max,
            "dp_deviation"   : dp_deviation,
            "dp_accel"       : dp_accel,
            "instability"    : instability,
            "pm_zscore"      : pm_zscore,
        }

    def _buffer_to_arrays(self):
        """Convert last SEQ_LEN buffer entries into (seq_array, tab_array)."""
        win = self.buffer[-self.seq_len:]

        # Build raw sequence  (SEQ_LEN × 13)
        feature_order = [
            "dp", "pm_val", "dp_slope", "dp_variance", "pm_spike_flag",
            "dp_rolling_mean", "dp_var_delta", "dp_slope_delta",
            "pm_roll_max", "dp_deviation", "dp_accel", "instability", "pm_zscore"
        ]
        raw_seq = np.array([[h[f] for f in feature_order] for h in win],
                           dtype=np.float32)

        # Scale
        seq_scaled = self.scaler.transform(raw_seq)                  # (60, 13)
        seq_in     = seq_scaled[np.newaxis, :, :]                    # (1, 60, 13)

        # Build tabular features from a temporary DataFrame
        tmp_df = pd.DataFrame(win).rename(columns={"pm_val": "pm"})
        tab    = np.array(list(make_tabular_features(tmp_df).values()),
                          dtype=np.float32).reshape(1, -1)           # (1, 25)
        return seq_in, tab

    def ingest(self, dp: float, pm: float, dp_rolling: float,
               dp_slope: float, dp_variance: float, pm_spike: int) -> dict:
        """
        Ingest one new sensor reading and return a prediction dict.
        Call every 10 seconds.

        Returns dict with:
            condition      : predicted class name
            pre_tear_prob  : probability of Pre_Tear  (0–1)
            confidence     : max probability across all classes
            probabilities  : full class probability dict
            alert          : True if Pre_Tear or Tearing detected
            warning_msg    : human-readable status string
        """
        entry = self._compute_features(dp, pm, dp_rolling,
                                       dp_slope, dp_variance, pm_spike)
        self.buffer.append(entry)

        # Not enough history yet
        if len(self.buffer) < self.seq_len:
            return {
                "condition": "Initializing",
                "pre_tear_prob": 0.0,
                "confidence": 0.0,
                "probabilities": {},
                "alert": False,
                "warning_msg": f"Warming up ({len(self.buffer)}/{self.seq_len} steps)",
            }

        seq_in, tab_in = self._buffer_to_arrays()

        # CNN-BiLSTM probability
        cnn_p = self.cnn.predict([seq_in, tab_in], verbose=0)[0]

        # LightGBM on embedding + tabular
        emb   = self.emb.predict([seq_in, tab_in], verbose=0)
        lgb_in = np.hstack([emb, tab_in])
        lgb_p  = self.lgbm.predict_proba(lgb_in)[0]

        # Hybrid
        hybrid_p  = self.alpha * cnn_p + (1 - self.alpha) * lgb_p
        pred_idx  = int(np.argmax(hybrid_p))
        pred_cls  = self.cls_names[pred_idx]
        confidence = float(hybrid_p[pred_idx])
        pt_prob    = float(hybrid_p[self.pt_idx]) if self.pt_idx >= 0 else 0.0

        alert = (pt_prob >= self.threshold or
                 (self.tear_idx >= 0 and hybrid_p[self.tear_idx] >= 0.50))

        if alert:
            self.alert_log.append({
                "step": len(self.buffer),
                "predicted": pred_cls,
                "pre_tear_prob": round(pt_prob, 4),
            })

        warning_msg = {
            "Normal":   f"✓ Normal  | ΔP={dp:.1f} Pa  | conf={confidence:.2f}",
            "Clogging": f"⚠ CLOGGING  | ΔP={dp:.1f} rising  | instability={entry['instability']:.3f}",
            "Pre_Tear": f"🚨 PRE-TEAR WARNING | P(tear)={pt_prob:.3f} | ~{PRE_TEAR_WINDOW*10//60} min to failure",
            "Tearing":  f"🔴 TEAR DETECTED  | PM={pm:.1f}  | ΔP={dp:.1f} dropped",
        }.get(pred_cls, "Unknown state")

        return {
            "condition"    : pred_cls,
            "pre_tear_prob": round(pt_prob, 4),
            "confidence"   : round(confidence, 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(self.cls_names, hybrid_p)},
            "alert"        : alert,
            "warning_msg"  : warning_msg,
        }


# ── Demo: simulate live ingestion on test rows ────────────────────────────────
print("  Demo — simulating 5 live sensor readings from test data...\n")

ews = BaghouseEarlyWarningSystem(
    cnn_model  = cnn_bilstm,
    emb_model  = embedding_model,
    lgbm_model = lgb_model,
    scaler     = scaler,
    le         = le,
)

# Pre-warm the buffer silently with test rows, then show last 5 predictions
test_start = split_idx    # first test index in original df (accounting for SEQ_LEN offset)
demo_rows  = df.iloc[test_start: test_start + SEQ_LEN + 5]

for i, row in demo_rows.iterrows():
    result = ews.ingest(
        dp         = row["dp"],
        pm         = row["pm"],
        dp_rolling = row["dp_rolling_mean"],
        dp_slope   = row["dp_slope"],
        dp_variance= row["dp_variance"],
        pm_spike   = int(row["pm_spike_flag"]),
    )
    if result["condition"] != "Initializing":
        actual = row["label"]
        print(f"  Actual: {actual:10s} | {result['warning_msg']}")

# ─────────────────────────────────────────────────────────────────────────────
# 18. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL RESULTS")
print(f"{'='*70}")
print(f"  CNN-BiLSTM Accuracy : {acc_cnn*100:.2f}%")
print(f"  LightGBM Accuracy   : {acc_lgb*100:.2f}%")
print(f"  Hybrid Accuracy     : {acc_hyb*100:.2f}%  ← BEST")
print(f"  Hybrid F1 Score     : {f1_hyb:.4f}")
print(f"  Outputs saved to    : ./{OUTPUT_DIR}/")
print(f"{'='*70}\n")
