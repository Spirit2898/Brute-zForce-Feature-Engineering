# streamlit_app.py
# -------------------------------------------------------------
# Brute Force Feature Engineering — step-by-step UI as requested
#
# What this app covers (exactly to your spec up to this step):
# 1) Title + file uploader (CSV/XLS/XLSX)
# 2) Step to drop columns (user selects multiple, press a button to apply)
# 3) Show column dtypes; ask for categorical columns to convert to integer codes 1..k
#    NOTE: You wrote "one hot encode" but also "convert to 1,2,3..." — that's *label encoding*,
#    not one‑hot. Here we implement label encoding to integers (NaNs -> 0).
# 4) Select base features (multi-select) and a single target (exactly one)
#
# You can continue building additional steps after this file.
# -------------------------------------------------------------

import streamlit as st  # UI framework for building the app
import pandas as pd  # Data manipulation for CSV/Excel handling
from typing import List, Dict  # For type hints to keep things clear

# ---------------------------
# Page config & title
# ---------------------------
st.set_page_config(
    page_title="Brute Force Feature Engineering",  # Sets the browser tab title for clarity
    layout="wide"  # Uses the full width for a roomier UI
)

st.title("Brute Force Feature Engineering")  # Main title exactly as requested
st.caption("Upload → Drop Columns → Encode Categoricals → Choose Base Features & Target → Cloose Model → Chose Metrics → Train")  # Quick guide

# ---------------------------
# Session state initializers
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None  # The working DataFrame that we mutate step-by-step
if "original_df" not in st.session_state:
    st.session_state.original_df = None  # A pristine copy of the uploaded dataset (optional safety)
if "dropped_cols" not in st.session_state:
    st.session_state.dropped_cols = set()  # Tracks which columns were dropped so far
if "encoded_cols" not in st.session_state:
    st.session_state.encoded_cols = set()  # Tracks which columns were encoded so far
if "encoders" not in st.session_state:
    st.session_state.encoders = {}  # Stores mappings used for label encoding (for reference)
if "base_features" not in st.session_state:
    st.session_state.base_features = []  # Stores selected base features
if "target" not in st.session_state:
    st.session_state.target = None  # Stores selected target


# ---------------------------
# Helpers
# ---------------------------

def _load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into a pandas DataFrame.

    If the file extension indicates Excel (xls/xlsx), uses pandas.read_excel.
    Otherwise, falls back to pandas.read_csv.
    """
    name = uploaded_file.name.lower()  # Normalize extension checks
    if name.endswith(".xlsx") or name.endswith(".xls"):  # Excel branch
        return pd.read_excel(uploaded_file)  # Read via read_excel
    return pd.read_csv(uploaded_file)  # CSV branch


def label_encode_columns(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict]:
    """
    Convert selected categorical columns to integer codes 0..k-1.
    No missing values expected since data is pre-cleaned.
    Deterministic via pandas.factorize(sort=True).
    """
    metadata: Dict[str, Dict] = {}
    for col in cols:
        # Check for any remaining NaN values (should not exist after cleaning)
        if pd.isna(df[col]).any():
            st.error(f"⚠️ **Unexpected missing values found in column '{col}' during encoding!**")
            st.error("This should not happen as missing values were removed during data loading.")
            st.stop()
        
        # 0..k-1 for present categories, no missing values to handle
        codes, uniques = pd.factorize(df[col], sort=True)
        df[col] = codes.astype("int64")

        # category -> 0..k-1 mapping
        mapping = {str(u): int(i) for i, u in enumerate(uniques)}
        metadata[col] = {
            "uniques": [str(u) for u in uniques.tolist()],
            "mapping": mapping,
            "na_code": None,  # No missing values to handle
        }
    return metadata  # Return per-column metadata


# ---------------------------
# Step 1 — Upload dataset
# ---------------------------
uploaded = st.file_uploader(
    "Upload your dataset (CSV or Excel)",  # Clear instructions
    type=["csv", "xlsx", "xls"],  # Restrict to expected formats
    accept_multiple_files=False  # Exactly one dataset at a time
)

if uploaded is not None and st.session_state.df is None:
    df = _load_dataframe(uploaded)  # Load the dataset into memory
    
    # Remove rows with any NaN values immediately after loading
    original_shape = df.shape
    df_cleaned = df.dropna()
    cleaned_shape = df_cleaned.shape
    
    if original_shape[0] != cleaned_shape[0]:
        st.warning(f"🧹 **Data Cleaning**: Removed {original_shape[0] - cleaned_shape[0]} rows with missing values")
        st.info(f"Dataset shape: {original_shape[0]} → {cleaned_shape[0]} rows ({cleaned_shape[1]} columns)")
    else:
        st.success("✅ **No missing values detected** - dataset is clean!")
    
    st.session_state.df = df_cleaned.copy()  # Keep a working copy to mutate safely
    st.session_state.original_df = df.copy()  # Store the original with missing values for reference

# Show a quick preview if available
if st.session_state.df is not None:
    st.subheader("Preview")  # Helps user confirm the data looks right
    with st.expander("Show first 50 rows", expanded=False):
        st.dataframe(st.session_state.df.head(50), use_container_width=True)  # Bounded preview
    st.caption(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

# Short-circuit until a dataset is uploaded
if st.session_state.df is None:
    st.info("Please upload a CSV/XLS/XLSX file to proceed.")  # Friendly nudge
    st.stop()  # Stop rendering further steps

# ---------------------------
# Step 2 — Drop columns (with button to apply)
# ---------------------------
st.subheader("Step 1: Drop Columns")
current_cols = list(st.session_state.df.columns)  # Current available columns
cols_to_drop = st.multiselect(
    "Select columns to drop",  # Prompt for dropping columns
    options=current_cols,  # Options reflect current DF state
    help="Choose one or more columns. Click 'Drop Selected Columns' to apply."
)

apply_drop = st.button("Drop Selected Columns", type="primary", disabled=(len(cols_to_drop) == 0))
if apply_drop:
    st.session_state.df.drop(columns=cols_to_drop, inplace=True, errors="ignore")  # Drop safely
    st.session_state.dropped_cols.update(cols_to_drop)  # Track what we dropped
    st.success(f"Dropped {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")  # Feedback

# ---------------------------
# Show dtypes & select categoricals for integer encoding 1..k
# ---------------------------
st.subheader("Step 2: Column Types")
with st.expander("View column dtypes", expanded=False):
    dtypes_table = pd.DataFrame(
        {"column": st.session_state.df.columns, "dtype": st.session_state.df.dtypes.astype(str).values})
    st.dataframe(dtypes_table, use_container_width=True)  # Clear overview of dtypes

st.subheader("Step 3: Encode Categorical Columns (to 0, 1, 2, … consecutive integers)")
remaining_cols = [c for c in st.session_state.df.columns]  # After any drops
# Heuristic guess for categoricals: object/category dtypes
likely_cats = [
    c for c in remaining_cols
    if (st.session_state.df[c].dtype == "object") or pd.api.types.is_categorical_dtype(st.session_state.df[c])
]

cols_to_encode = st.multiselect(
    "Select categorical columns to convert to integer codes",
    options=remaining_cols,
    default=likely_cats,  # Pre-select likely categoricals
    help="This performs label encoding (not one-hot). Each unique value → 0..k-1 consecutive integers."
)

apply_encode = st.button("Convert Selected Columns", disabled=(len(cols_to_encode) == 0))
if apply_encode:
    meta = label_encode_columns(st.session_state.df, cols_to_encode)  # Apply label encoding
    st.session_state.encoded_cols.update(cols_to_encode)  # Track encoded columns
    st.session_state.encoders.update(meta)  # Save mappings for transparency
    st.success(f"Converted {len(cols_to_encode)} column(s) to integer codes.")
    with st.expander("Show encoding mappings", expanded=False):
        for col, info in meta.items():
            st.markdown(f"**{col}** — Encoded to consecutive integers 0, 1, 2, ...")
            mapping_df = pd.DataFrame(
                {"category": list(info["mapping"].keys()), "code": list(info["mapping"].values())})
            st.dataframe(mapping_df, use_container_width=True)

# ---------------------------
# Step 4 — Select target first, then base features
# ---------------------------
st.subheader("Step 4: Choose Target & Base Features")
all_current = list(st.session_state.df.columns)  # Current columns after ops

# Step 4a: Select target first
st.write("**4a. Select Target Column**")

st.session_state.target = st.selectbox(
    "Select the target column (exactly one)",  # Single selection for target
    options=all_current,
    index=0 if all_current else None,
    help="Choose the column you want to predict",
    key="target_selector"
)

# Step 4b: Auto-populate base features and allow removal
st.write("**4b. Configure Base Features**")
if st.session_state.target:
    # Auto-populate all columns except target as base features
    available_features = [c for c in all_current if c != st.session_state.target]

    # Track target changes to reset base features when target changes
    if "previous_target" not in st.session_state:
        st.session_state.previous_target = st.session_state.target

    # If target changed, reset base features
    if st.session_state.target != st.session_state.previous_target:
        st.session_state.base_features = available_features.copy()
        st.session_state.previous_target = st.session_state.target

    # Initialize base_features if not exists or clean up invalid features
    if "base_features" not in st.session_state or st.session_state.base_features is None:
        st.session_state.base_features = available_features.copy()  # Auto-select all
    else:
        # Clean up base_features: remove any features that are no longer available
        # (e.g., if user changed target, the old target might be in base_features)
        valid_features = [f for f in st.session_state.base_features if f in available_features]

        # If we cleaned out everything, restore to all available features
        if not valid_features:
            st.session_state.base_features = available_features.copy()
        else:
            st.session_state.base_features = valid_features

    # Allow user to remove features they don't want
    st.session_state.base_features = st.multiselect(
        "Base feature columns (remove unwanted features)",
        options=available_features,
        default=st.session_state.base_features,
        help="All non-target columns are selected by default. Remove any you don't want to use.",
        key="base_features_selector"
    )

    # Summary box
    st.success(
        f"**Target:** {st.session_state.target}\n\n"
        f"**Base features ({len(st.session_state.base_features)}):** "
        f"{', '.join(st.session_state.base_features) if st.session_state.base_features else 'None selected'}"
    )

    if len(st.session_state.base_features) == 0:
        st.error("⚠️ You must select at least one base feature to continue!")

else:
    st.warning("Please select a target column first.")
    st.session_state.base_features = []

    # OLD Summary box (replaced above)
    # st.info(
    f"Base features (\u2192 {len(st.session_state.base_features)}): "
    f"{', '.join(st.session_state.base_features) if st.session_state.base_features else 'None selected'}\n\n"
    f"Target: {st.session_state.target if st.session_state.target else 'None selected'}"
# )

# -------------------------------------------------------------
# Complexity Notes (for current steps only)
# -------------------------------------------------------------
# Upload/Preview:
#   Time: O( N * M ) to load N rows × M cols from disk; preview is O(min(N,50) * M)
#   Space: O( N * M ) for the in-memory DataFrame + small overhead for preview table
# Drop Columns:
#   Time: O( M ) where M is number of columns to drop (pandas drops by label — effectively linear in M)
#   Space: O(1) extra; in-place drop mutates the DataFrame (shallow copies of blocks possible internally)
# Label Encoding (per column):
#   Time: O( N + K log K ) — N to map values; K = number of unique categories (sorted for determinism)
#   Space: O( K ) for mapping + O( N ) for the integer-typed column (replacing original)
# Feature/Target Selection:
#   Time: O( M ) for rendering selection widgets where M = current number of columns
#   Space: O( M ) transient UI state + lists stored in session_state


# ===========================
# Step 5 — Split X (features) and y (target)
# ===========================
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

# App-wide defaults
RANDOM_STATE = 42
N_SPLITS_DEFAULT = 5
SHUFFLE_DEFAULT = True


# ===========================
# Utility functions for metrics computation
# ===========================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=1, keepdims=True)


def _predict_proba_like(est, X):
    """Return (proba, classes). If no predict_proba, try decision_function.
    Binary → (n_samples,); Multiclass → (n_samples, n_classes)."""
    classes = getattr(est, "classes_", None)
    if hasattr(est, "predict_proba"):
        P = est.predict_proba(X)
        if P.ndim == 2 and P.shape[1] == 2:
            return P[:, 1], est.classes_
        return P, est.classes_
    if hasattr(est, "decision_function"):
        S = est.decision_function(X)
        if np.ndim(S) == 1:
            return _sigmoid(S), classes
        return _softmax(S), classes
    return None, classes


def _binary_conf_counts(y_true, y_pred, pos_label=None):
    # Ensure label order [neg, pos]
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if pos_label is None:
        # Heuristic: use the second class as positive (consistent with classes_ order for many estimators)
        pos_label = labels[-1]
    neg_label = [l for l in labels if l != pos_label][0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, fn, tp = cm.ravel()
    return tp, fp, tn, fn


def _ks_stat(y_true, y_score):
    # Empirical CDF difference between positives and negatives
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_scores = y_score[y_true == y_true.max()]
    neg_scores = y_score[y_true != y_true.max()]
    # Sort and compute CDFs on union grid
    grid = np.sort(np.unique(np.concatenate([pos_scores, neg_scores])))

    def ecdf(vals, grid):
        if len(vals) == 0:
            return np.zeros_like(grid, dtype=float)
        ranks = np.searchsorted(np.sort(vals), grid, side="right")
        return ranks / float(len(vals))

    cdf_pos = ecdf(pos_scores, grid)
    cdf_neg = ecdf(neg_scores, grid)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _brier_multiclass(y_true, proba, classes):
    # Mean squared error between one-hot and probabilities
    from sklearn.preprocessing import label_binarize
    Y = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((Y - proba) ** 2, axis=1)))


def _compute_metrics_one_fold(y_true, y_pred, proba, classes, label_kind, selected):
    """Compute a selected set of metrics for one validation fold.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels for the validation split.
    y_pred : array-like
        Discrete predictions from the estimator for the validation split.
    proba : array-like or None
        Predicted probability for the positive class (binary) or
        class-probability matrix of shape (n_samples, n_classes) (multiclass),
        or None if the estimator does not expose calibrated scores.
    classes : array-like or None
        Class labels corresponding to columns of ``proba`` for multiclass.
        When None, will fall back to unique labels in ``y_true``.
    label_kind : {"binary", "multiclass"}
        Detected task type for classification.
    selected : List[str]
        Names of the metrics to compute.
    """
    from math import sqrt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
        roc_auc_score, average_precision_score, log_loss, brier_score_loss, jaccard_score,
        fowlkes_mallows_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
        precision_recall_fscore_support
    )
    from sklearn.preprocessing import label_binarize

    m = {}

    # Metrics that require probabilities/scores
    needs_proba = {
        "ROC AUC", "Partial AUC", "PR AUC (Average Precision)", "PR AUC (macro)", "PR AUC (micro)",
        "Log Loss", "Brier Score", "R-Precision"
    }

    # Placeholders for metrics not yet implemented in this UI
    unsupported = {
        "Partial AUC", "R-Precision", "Brier Skill Score", "Spherical Score",
        "ECE", "MCE", "Adaptive ECE", "Calibration Slope", "Calibration Intercept",
        "Hosmer–Lemeshow", "Spiegelhalter's Z", "Top-k Accuracy", "Precision@k", "Recall@k", "Hit@k",
        "TPR@FPR", "FPR@TPR", "DET minDCF", "Fβ", "Macro Fβ"
    }

    # Check for degenerate cases
    if len(np.unique(y_true)) <= 1:
        print(f"Warning: y_true has only {len(np.unique(y_true))} unique values: {np.unique(y_true)}")
    if len(np.unique(y_pred)) <= 1:
        print(f"Warning: y_pred has only {len(np.unique(y_pred))} unique values: {np.unique(y_pred)}")

    # Binary-derived counts and convenience terms
    if label_kind == "binary":
        tp, fp, tn, fn = _binary_conf_counts(y_true, y_pred)
        print(f"Binary confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")  # Debug
        tpr = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity/recall
        tnr = tn / (tn + fp) if (tn + fp) else 0.0  # specificity
        fpr = 1.0 - tnr
        fnr = 1.0 - tpr
        ppv = tp / (tp + fp) if (tp + fp) else 0.0  # precision
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        fdr = 1.0 - ppv
        _for = fn / (fn + tn) if (fn + tn) else 0.0  # false omission rate
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0  # threat score
        youden = tpr + tnr - 1.0
        markedness = ppv + npv - 1.0
        gmean = sqrt(max(tpr, 0.0) * max(tnr, 0.0))

    # Compute each requested metric
    for name in selected:
        try:
            if name in unsupported:
                m[name] = np.nan
                continue

            # --- General classification metrics ---
            if name == "Accuracy":
                acc = accuracy_score(y_true, y_pred)
                m[name] = acc
                if acc >= 0.99:
                    print(f"WARNING: Perfect/near-perfect accuracy ({acc:.4f}) detected for metric '{name}'")
            elif name == "Error Rate":
                m[name] = 1.0 - accuracy_score(y_true, y_pred)
            elif name in ("F1", "Dice Coefficient"):
                avg = "binary" if label_kind == "binary" else "macro"
                f1 = f1_score(y_true, y_pred, average=avg)
                m[name] = f1
                if f1 >= 0.99:
                    print(f"WARNING: Perfect/near-perfect F1 score ({f1:.4f}) detected for metric '{name}'")
            elif name == "Fowlkes–Mallows":
                m[name] = fowlkes_mallows_score(y_true, y_pred)
            elif name == "Balanced Accuracy":
                m[name] = balanced_accuracy_score(y_true, y_pred)
            elif name == "Balanced Error Rate (BER)":
                m[name] = 1.0 - balanced_accuracy_score(y_true, y_pred)
            elif name == "Jaccard (IoU)":
                avg = "binary" if label_kind == "binary" else "macro"
                m[name] = jaccard_score(y_true, y_pred, average=avg)
            elif name in ("MCC", "Multiclass MCC"):
                m[name] = matthews_corrcoef(y_true, y_pred)
            elif name == "Cohen's κ":
                m[name] = cohen_kappa_score(y_true, y_pred)
            elif name == "Quadratic-weighted κ":
                m[name] = cohen_kappa_score(y_true, y_pred, weights="quadratic")

            # --- Binary-only derived metrics ---
            elif label_kind == "binary" and name == "Precision (PPV)":
                m[name] = precision_score(y_true, y_pred, zero_division=0)
            elif label_kind == "binary" and name == "Recall (TPR/Sensitivity)":
                m[name] = recall_score(y_true, y_pred, zero_division=0)
            elif label_kind == "binary" and name == "Specificity (TNR)":
                m[name] = tnr
            elif label_kind == "binary" and name == "NPV":
                m[name] = npv
            elif label_kind == "binary" and name == "FPR":
                m[name] = fpr
            elif label_kind == "binary" and name == "FNR":
                m[name] = fnr
            elif label_kind == "binary" and name == "FDR":
                m[name] = fdr
            elif label_kind == "binary" and name == "FOR":
                m[name] = _for
            elif label_kind == "binary" and name == "Youden's J (Informedness)":
                m[name] = youden
            elif label_kind == "binary" and name == "Markedness":
                m[name] = markedness
            elif label_kind == "binary" and name == "Threat Score (CSI)":
                m[name] = csi
            elif label_kind == "binary" and name == "LR+":
                m[name] = (tpr / fpr) if fpr > 0 else np.inf
            elif label_kind == "binary" and name == "LR-":
                m[name] = (fnr / tnr) if tnr > 0 else np.inf
            elif label_kind == "binary" and name == "Diagnostic Odds Ratio (DOR)":
                lr_p = (tpr / fpr) if fpr > 0 else np.inf
                lr_n = (fnr / tnr) if tnr > 0 else np.inf
                m[name] = (lr_p / lr_n) if (
                        np.isfinite(lr_p) and np.isfinite(lr_n) and lr_n != 0) else np.nan
            elif label_kind == "binary" and name == "KS Statistic":
                m[name] = np.nan if proba is None else _ks_stat(y_true, proba)

            # --- Multiclass aggregates ---
            elif label_kind == "multiclass" and name in ("Macro Precision", "Macro Recall", "Macro F1"):
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                             zero_division=0)
                m.update({"Macro Precision": p, "Macro Recall": r, "Macro F1": f})
            elif label_kind == "multiclass" and name in ("Micro Precision", "Micro Recall", "Micro F1"):
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro",
                                                             zero_division=0)
                m.update({"Micro Precision": p, "Micro Recall": r, "Micro F1": f})
            elif label_kind == "multiclass" and name in ("Weighted Precision", "Weighted Recall",
                                                         "Weighted F1"):
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                             zero_division=0)
                m.update({"Weighted Precision": p, "Weighted Recall": r, "Weighted F1": f})

            # --- Probability-based metrics ---
            elif name in needs_proba:
                if proba is None:
                    m[name] = np.nan
                else:
                    if name == "ROC AUC":
                        if label_kind == "binary":
                            m[name] = roc_auc_score(y_true, proba)
                        else:
                            m[name] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
                    elif name == "PR AUC (Average Precision)":
                        if label_kind == "binary":
                            m[name] = average_precision_score(y_true, proba)
                        else:
                            # Multiclass macro AP (average of per-class AP)
                            Y = label_binarize(y_true, classes=classes)
                            m[name] = float(np.mean([
                                average_precision_score(Y[:, i], proba[:, i])
                                for i in range(proba.shape[1])
                            ]))
                    elif name == "PR AUC (macro)":
                        Y = label_binarize(y_true, classes=classes)
                        m[name] = float(np.mean([
                            average_precision_score(Y[:, i], proba[:, i])
                            for i in range(proba.shape[1])
                        ]))
                    elif name == "PR AUC (micro)":
                        Y = label_binarize(y_true, classes=classes)
                        m[name] = average_precision_score(Y.ravel(), proba.ravel())
                    elif name == "Log Loss":
                        if label_kind == "binary":
                            P = np.vstack([1 - proba, proba]).T
                            m[name] = log_loss(y_true, P, labels=classes)
                        else:
                            m[name] = log_loss(y_true, proba, labels=classes)
                    elif name == "Brier Score":
                        if label_kind == "binary":
                            m[name] = brier_score_loss(y_true, proba)
                        else:
                            m[name] = _brier_multiclass(y_true, proba, classes)
        except Exception as err:
            print(f"Error computing metric '{name}': {err}")  # Debug print
            m[name] = np.nan

    return m


st.subheader("Step 5: Create X and y")
if st.session_state.target is None or len(st.session_state.base_features) == 0:
    st.warning("Please choose base features and a single target in Step 4 to continue.")
    st.stop()

# Build X and y from the user's choices
st.session_state.y = st.session_state.df[st.session_state.target]
st.session_state.X_df = st.session_state.df[st.session_state.base_features].copy()

# Reconfirmation printout
with st.expander("Reconfirm selections", expanded=True):
    st.write("**Base features (columns):**")
    st.code(
        "\n".join([str(c) for c in st.session_state.base_features]) if st.session_state.base_features else "<none>",
        language="text",
    )
    st.write("**Target:**")
    st.code(str(st.session_state.target), language="text")
    st.caption(
        f"X_df shape: {st.session_state.X_df.shape}  |  y length: {len(st.session_state.y)}"
    )


# ===========================
# Step 6 — Detect task type and let user confirm
# ===========================

def detect_task_type(y: pd.Series) -> str:
    """Heuristic suggestion: classification if labels look discrete; else regression."""
    # Non-numeric or categorical ⇒ classification
    if (y.dtype == "object") or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    # Integer dtype with few unique values ⇒ classification
    if pd.api.types.is_integer_dtype(y):
        uniq = pd.unique(y.dropna())
        if len(uniq) <= 20:
            return "classification"
    return "regression"


suggested_task = detect_task_type(st.session_state.y)
st.info(
    f"Suggested task type: **{suggested_task}** · unique labels: "
    f"{len(pd.unique(st.session_state.y.dropna()))} · dtype: {st.session_state.y.dtype}"
)

TASK_TYPE = st.radio(
    "Select the problem type",
    options=["classification", "regression"],
    index=0 if suggested_task == "classification" else 1,
    horizontal=True,
)

# Store final task type
st.session_state.task_type = TASK_TYPE

# Class label diagnostics (for classification)
if st.session_state.task_type == "classification":
    uniq_vals = pd.unique(st.session_state.y.dropna())
    n_classes = len(uniq_vals)
    label_kind = "binary" if n_classes == 2 else "multiclass"
    st.session_state.class_label_kind = label_kind
    st.caption(
        f"Detected **{label_kind}** classification (classes: {n_classes}). "
        f"Examples: {list(uniq_vals)[:10]}{' …' if n_classes > 10 else ''}"
    )


# ===========================
# Step 7 — Cross-validation splitter
# ===========================

def get_cv(task_type: str, n_splits: int, shuffle: bool, random_state: int):
    """Return an appropriate CV splitter."""
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


with st.expander("Cross-validation settings", expanded=True):
    n_splits = st.slider("Number of folds", min_value=3, max_value=10, value=N_SPLITS_DEFAULT, step=1)
    shuffle_cv = st.checkbox("Shuffle folds", value=SHUFFLE_DEFAULT)
    rs = st.number_input("Random state", value=RANDOM_STATE, step=1)

st.session_state.cv = get_cv(st.session_state.task_type, n_splits, shuffle_cv, int(rs))
st.success(
    f"Using {'StratifiedKFold' if st.session_state.task_type == 'classification' else 'KFold'} with "
    f"n_splits={n_splits}, shuffle={shuffle_cv}, random_state={int(rs)}"
)


# ===========================
# Step 8 — Model selection & hyperparameters (list-based controls)
# ===========================

def build_estimator(task_type: str, model_name: str, use_gpu: bool):
    """
    Return an estimator (fit(X,y) -> predict / predict_proba) with sensible defaults.
    Supports: xgboost, lightgbm, catboost, and many sklearn models.
    """
    model_name = model_name.lower()

    # ---------------- Gradient Boosting Libraries ----------------
    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        common = dict(
            n_estimators=300, max_depth=6, subsample=0.9, colsample_bytree=0.8,
            learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=1
        )
        if task_type == "classification":
            params = dict(**common)
            if use_gpu:
                params.update(tree_method="gpu_hist", predictor="gpu_predictor")
            else:
                params.update(tree_method="hist")
            return XGBClassifier(**params)
        else:
            params = dict(**common)
            if use_gpu:
                params.update(tree_method="gpu_hist", predictor="gpu_predictor")
            else:
                params.update(tree_method="hist")
            return XGBRegressor(**params)

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        common = dict(
            n_estimators=600, max_depth=-1, subsample=0.9, colsample_bytree=0.8,
            learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=1
        )
        device_arg = {"device": "gpu"} if use_gpu else {}
        return LGBMClassifier(**common, **device_arg) if task_type == "classification" else LGBMRegressor(**common,
                                                                                                          **device_arg)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        common = dict(
            iterations=600, depth=6, learning_rate=0.05, random_state=RANDOM_STATE,
            verbose=False, thread_count=1
        )
        task_args = dict(task_type="GPU") if use_gpu else dict(task_type="CPU")
        return CatBoostClassifier(**common, **task_args) if task_type == "classification" else CatBoostRegressor(
            **common, **task_args)

    # ---------------- scikit-learn Classifiers ----------------
    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    elif model_name == "ridge_clf":
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(random_state=RANDOM_STATE)

    elif model_name == "sgd_clf":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="log_loss", max_iter=1000, random_state=RANDOM_STATE)

    elif model_name == "dt_clf":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=RANDOM_STATE)

    elif model_name == "rf_clf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == "et_clf":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == "gb_clf":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

    elif model_name == "hgb_clf":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    elif model_name == "mlp_clf":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)

    elif model_name == "knn_clf":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

    elif model_name == "svc":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=RANDOM_STATE)

    elif model_name == "gnb":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    elif model_name == "ada_clf":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=RANDOM_STATE)

    elif model_name == "bag_clf":
        from sklearn.ensemble import BaggingClassifier
        return BaggingClassifier(random_state=RANDOM_STATE)

    # ---------------- scikit-learn Regressors ----------------
    elif model_name == "lin_reg":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    elif model_name == "ridge_reg":
        from sklearn.linear_model import Ridge
        return Ridge(random_state=RANDOM_STATE)

    # ---------------- Unsupported ----------------
    else:
        raise ValueError(
            f"Unknown MODEL_NAME='{model_name}'. Supported models include GBMs and many sklearn classifiers/regressors.")


# Model options filtered by task type
CLASSIFICATION_MODELS = [
    "xgboost", "lightgbm", "catboost", "logreg", "ridge_clf", "sgd_clf", "dt_clf", "rf_clf",
    "et_clf", "gb_clf", "hgb_clf", "mlp_clf", "knn_clf", "svc", "gnb", "ada_clf", "bag_clf",
]
REGRESSION_MODELS = ["xgboost", "lightgbm", "catboost", "lin_reg", "ridge_reg"]

st.subheader("Step 8: Choose Model & Hyperparameters")
use_gpu = st.checkbox("Use GPU (if available)", value=False)
model_list = CLASSIFICATION_MODELS if st.session_state.task_type == "classification" else REGRESSION_MODELS

model_name = st.selectbox(
    "Select model",
    options=model_list,
    index=(model_list.index("xgboost") if "xgboost" in model_list else 0),
)

# Parameter controls per model (non-text inputs only)
user_params = {}
with st.expander("Model hyperparameters", expanded=True):
    if model_name in ("xgboost", "lightgbm", "catboost"):
        # Shared GBM-like knobs
        n_estimators = st.select_slider("n_estimators", options=[100, 200, 300, 400, 600, 800, 1000], value=300)
        lr = st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
        if model_name == "xgboost":
            max_depth = st.select_slider("max_depth", options=list(range(3, 13)), value=6)
            subsample = st.select_slider("subsample", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.9)
            colsample = st.select_slider("colsample_bytree", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
            user_params.update(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth,
                               subsample=subsample, colsample_bytree=colsample)
        elif model_name == "lightgbm":
            max_depth = st.select_slider("max_depth (-1 means no limit)", options=[-1, 4, 6, 8, 10, 12], value=-1)
            subsample = st.select_slider("subsample", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.9)
            colsample = st.select_slider("colsample_bytree", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8)
            user_params.update(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth,
                               subsample=subsample, colsample_bytree=colsample)
        else:  # catboost
            depth = st.select_slider("depth", options=list(range(4, 11)), value=6)
            user_params.update(iterations=n_estimators, learning_rate=lr, depth=depth)

    elif model_name == "rf_clf":
        n_estimators = st.select_slider("n_estimators", options=[100, 200, 300, 500, 800], value=200)
        max_features = st.selectbox("max_features", options=["sqrt", "log2", None], index=0)
        user_params.update(n_estimators=n_estimators, max_features=max_features)

    elif model_name == "svc":
        kernel = st.selectbox("kernel", options=["rbf", "linear", "poly", "sigmoid"], index=0)
        C = st.select_slider("C", options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        gamma = st.selectbox("gamma", options=["scale", "auto"], index=0)
        user_params.update(kernel=kernel, C=C, gamma=gamma)

    elif model_name == "knn_clf":
        n_neighbors = st.select_slider("n_neighbors", options=list(range(3, 31, 2)), value=15)
        weights = st.selectbox("weights", options=["uniform", "distance"], index=0)
        user_params.update(n_neighbors=n_neighbors, weights=weights)

    elif model_name == "mlp_clf":
        arch = st.selectbox("hidden_layer_sizes", options=["(100,)", "(128, 64)", "(256, 128)", "(256, 128, 64)"],
                            index=0)
        activation = st.selectbox("activation", options=["relu", "tanh", "logistic"], index=0)
        user_params.update(hidden_layer_sizes=eval(arch), activation=activation)

    elif model_name == "logreg":
        penalty = st.selectbox("penalty", options=["l2"], index=0)
        C = st.select_slider("C", options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        solver = st.selectbox("solver", options=["lbfgs", "saga"], index=0)
        user_params.update(penalty=penalty, C=C, solver=solver)

    elif model_name == "dt_clf":
        max_depth = st.select_slider("max_depth", options=list(range(2, 21)), value=8)
        criterion = st.selectbox("criterion", options=["gini", "entropy", "log_loss"], index=0)
        user_params.update(max_depth=max_depth, criterion=criterion)

    elif model_name == "et_clf":
        n_estimators = st.select_slider("n_estimators", options=[100, 200, 300, 500], value=300)
        user_params.update(n_estimators=n_estimators)

    elif model_name == "gb_clf":
        lr = st.select_slider("learning_rate", options=[0.01, 0.03, 0.05, 0.1], value=0.05)
        max_depth = st.select_slider("max_depth", options=list(range(2, 9)), value=3)
        user_params.update(learning_rate=lr, max_depth=max_depth)

    elif model_name == "hgb_clf":
        max_depth = st.select_slider("max_depth", options=[None, 4, 6, 8, 10], value=None)
        l2 = st.select_slider("l2_regularization", options=[0.0, 0.01, 0.1, 1.0], value=0.0)
        user_params.update(max_depth=max_depth, l2_regularization=l2)

    elif model_name == "ridge_clf":
        alpha = st.select_slider("alpha", options=[0.1, 1.0, 10.0, 100.0], value=1.0)
        user_params.update(alpha=alpha)

    elif model_name == "sgd_clf":
        loss = st.selectbox("loss", options=["log_loss", "hinge", "modified_huber"], index=0)
        alpha = st.select_slider("alpha", options=[1e-5, 1e-4, 1e-3], value=1e-4)
        user_params.update(loss=loss, alpha=alpha)

    elif model_name == "ada_clf":
        n_estimators = st.select_slider("n_estimators", options=[50, 100, 200, 300], value=100)
        learning_rate = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=1.0)
        user_params.update(n_estimators=n_estimators, learning_rate=learning_rate)

    elif model_name == "bag_clf":
        n_estimators = st.select_slider("n_estimators", options=[10, 50, 100, 200, 300], value=10)
        user_params.update(n_estimators=n_estimators)

    elif model_name == "gnb":
        var_smoothing = st.select_slider("var_smoothing (log10)", options=[-12, -10, -9, -8], value=-9)
        user_params.update(var_smoothing=10.0 ** var_smoothing)

    elif model_name == "lin_reg":
        pass  # No primary hyperparameters here

    elif model_name == "ridge_reg":
        alpha = st.select_slider("alpha", options=[0.1, 1.0, 10.0, 100.0], value=1.0)
        user_params.update(alpha=alpha)

# Instantiate estimator with defaults, then override with user params
try:
    est = build_estimator(st.session_state.task_type, model_name, use_gpu)
    if user_params:
        est.set_params(**user_params)
    st.session_state.estimator = est
    st.success(f"Configured model: {model_name} with params: {user_params if user_params else '{defaults}'}")
except Exception as e:
    st.error(f"Error building model '{model_name}': {e}")

# ===========================
# Step 9 — Metric selection (primary + up to 3 extras; classification only for now)
# ===========================

BINARY_ONLY = [
    "Precision (PPV)", "Recall (TPR/Sensitivity)", "Specificity (TNR)", "NPV", "FPR", "FNR", "FDR", "FOR",
    "Youden’s J (Informedness)", "Markedness",
    "LR+", "LR-", "Diagnostic Odds Ratio (DOR)",
    "Equal Error Rate (EER)", "TPR@FPR", "FPR@TPR", "DET minDCF",
    "Threat Score (CSI)", "KS Statistic"
]
MULTICLASS_ONLY = [
    "Macro Precision", "Macro Recall", "Macro F1", "Macro Fβ",
    "Micro Precision", "Micro Recall", "Micro F1",
    "Weighted Precision", "Weighted Recall", "Weighted F1",
    "Top-k Accuracy", "Precision@k", "Recall@k", "Hit@k",
    "OvR ROC AUC (macro)", "OvO ROC AUC (macro)", "PR AUC (macro)", "PR AUC (micro)",
    "Multiclass MCC", "Cohen’s κ", "Quadratic-weighted κ", "Ranked Probability Score (RPS)"
]
BOTH_CLF = [
    "Accuracy", "Error Rate", "F1", "Fβ", "Fowlkes–Mallows",
    "Balanced Accuracy", "Balanced Error Rate (BER)", "G-Mean",
    "Jaccard (IoU)", "Dice Coefficient",
    "MCC", "ROC AUC", "Partial AUC", "PR AUC (Average Precision)", "R-Precision",
    "Log Loss", "Brier Score", "Brier Skill Score", "Spherical Score",
    "ECE", "MCE", "Adaptive ECE", "Calibration Slope", "Calibration Intercept",
    "Hosmer–Lemeshow", "Spiegelhalter’s Z"
]

if st.session_state.task_type == "classification":
    detected_kind = st.session_state.class_label_kind  # 'binary' or 'multiclass'
    if detected_kind == "binary":
        metric_pool = sorted(set(BOTH_CLF + BINARY_ONLY))
        default_primary = "ROC AUC"
    else:
        metric_pool = sorted(set(BOTH_CLF + MULTICLASS_ONLY))
        default_primary = "Macro F1"

    st.subheader("Step 9: Choose Metrics (exactly 4)")
    st.caption(f"Detected label type: **{detected_kind}** — showing relevant metrics.")

    primary_metric = st.selectbox("Primary metric", options=metric_pool,
                                  index=(metric_pool.index(default_primary) if default_primary in metric_pool else 0))

    extra_options = [m for m in metric_pool if m != primary_metric]
    extra_metrics = st.multiselect("Pick up to 3 additional metrics", options=extra_options, default=[])

    # Enforce the 4-metric rule strictly
    total_selected = 1 + len(extra_metrics)
    if total_selected > 4:
        st.error(
            f"You selected {total_selected}. Please choose at most 3 additional metrics (total = 4 including primary).")
    elif total_selected < 4:
        st.warning(
            f"You have selected {total_selected}. Please select {4 - total_selected} more to reach 4 total metrics.")
    else:
        st.success("Great — exactly 4 metrics selected.")

    st.session_state.primary_metric = primary_metric
    st.session_state.extra_metrics = extra_metrics
    st.session_state.selected_metrics = [primary_metric] + extra_metrics
else:
    st.subheader("Step 9: Metrics")
    st.info("Regression metrics selection will be added in the next step you requested.")

# ===========================
# Step 10 — Run CV and report selected metrics
# ===========================
from math import sqrt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss, jaccard_score,
    fowlkes_mallows_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
import numpy as np

st.subheader("Step 10: Evaluate Model with Selected Metrics")

# Guardrails
if st.session_state.task_type != "classification":
    st.info("Regression evaluation is coming next — this step runs for classification only.")
else:
    # Must have exactly 4 metrics
    selected_metrics = st.session_state.get("selected_metrics", [])
    can_run = isinstance(selected_metrics, list) and len(selected_metrics) == 4 and ("estimator" in st.session_state)

    if not can_run:
        st.warning("Select exactly 4 metrics in Step 9 and make sure a model is configured in Step 8.")
    else:
        # ===========================
        # Cross-validation execution
        # ===========================
        from sklearn.base import clone
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        # Prepare data
        X = st.session_state.X_df.values
        y = st.session_state.y.values
        label_kind = st.session_state.class_label_kind  # 'binary' or 'multiclass'
        cv = st.session_state.cv
        est = st.session_state.estimator
        metrics_to_compute = st.session_state.selected_metrics

        # Data quality checks
        st.subheader("🔍 Data Quality Analysis")
        with st.expander("Data Quality Report", expanded=True):
            # Check for data issues
            st.write(f"**Dataset shape:** {X.shape[0]} rows × {X.shape[1]} features")
            st.write(f"**Target distribution:** {dict(zip(*np.unique(y, return_counts=True)))}")

            # Check for class imbalance
            unique_y, counts_y = np.unique(y, return_counts=True)
            if len(unique_y) > 1:
                min_class_ratio = min(counts_y) / sum(counts_y)
                if min_class_ratio < 0.05:
                    st.error(
                        f"⚠️ **SEVERE CLASS IMBALANCE**: Smallest class represents only {min_class_ratio:.2%} of data")
                    st.error("This can cause models to predict only the majority class!")
                elif min_class_ratio < 0.1:
                    st.warning(f"⚠️ **CLASS IMBALANCE**: Smallest class represents {min_class_ratio:.2%} of data")

            # Check for missing values (should not exist after cleaning)
            if np.any(np.isnan(X)):
                st.error("⚠️ **UNEXPECTED MISSING VALUES DETECTED** in features")
                st.error("This should not happen as missing values were removed during data loading.")
            if np.any(pd.isna(y)):
                st.error("⚠️ **UNEXPECTED MISSING VALUES DETECTED** in target")
                st.error("This should not happen as missing values were removed during data loading.")
            else:
                st.success("✅ **No missing values** - data is clean as expected!")

            # Check for constant features
            constant_features = []
            for i in range(X.shape[1]):
                if len(np.unique(X[:, i])) == 1:
                    constant_features.append(i)
            if constant_features:
                st.warning(f"⚠️ **CONSTANT FEATURES DETECTED**: Features {constant_features} have no variation")

            # Additional diagnostic suggestions
            st.write("**🔧 Common Issues That Cause Perfect Accuracy:**")
            st.write("1. **Data Leakage**: Target variable or future information leaked into features")
            st.write("2. **Duplicate Rows**: Same samples appear multiple times")
            st.write("3. **Trivial Dataset**: Too easy to predict (e.g., synthetic data)")
            st.write("4. **Model Overfitting**: Model memorizing training data")
            st.write("5. **Single Class Prediction**: Model always predicts majority class")

        st.write("\n")
        run_eval = st.button("Run Cross-Validation", type="primary")

        if run_eval:
            fold_rows = []  # Per-fold metric rows
            per_fold_preds = []  # Optional: keep predictions per fold
            per_fold_probas = []  # Optional: keep scores/probabilities per fold
            unique_labels = np.unique(y[~pd.isnull(y)])  # Consistent label order for CM
            cm_total = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

            # Debug information
            st.info(f"Starting CV with {len(unique_labels)} unique labels: {unique_labels}")
            st.info(f"Data shape: X={X.shape}, y={y.shape}")
            st.info(f"Task type: {label_kind}, Metrics: {metrics_to_compute}")

            # Iterate CV folds
            for fold_idx, (tr, va) in enumerate(cv.split(X, y)):
                Xtr, Xva = X[tr], X[va]
                ytr, yva = y[tr], y[va]

                # Fresh clone each fold
                model = clone(est)
                try:
                    model.fit(Xtr, ytr)
                except Exception as fit_err:
                    st.error(f"Fold {fold_idx + 1}: model.fit failed — {fit_err}")
                    continue

                # Predictions
                try:
                    yhat = model.predict(Xva)

                    # Check for problematic predictions in each fold
                    if len(np.unique(yhat)) == 1:
                        st.warning(f"⚠️ Fold {fold_idx + 1}: Model predicting only one class: {np.unique(yhat)[0]}")

                except Exception as pred_err:
                    st.error(f"Fold {fold_idx + 1}: model.predict failed — {pred_err}")
                    continue

                # Probabilities / decision scores (if available)
                proba, classes = _predict_proba_like(model, Xva)

                # Collect confusion matrix for this fold (robust to missing labels in fold)
                try:
                    cm = confusion_matrix(yva, yhat, labels=unique_labels)
                    cm_total += cm
                except Exception:
                    pass

                # Debug fold information
                print(f"Fold {fold_idx + 1}: yva unique={np.unique(yva)}, yhat unique={np.unique(yhat)}")

                # Compute selected metrics
                metrics_dict = _compute_metrics_one_fold(yva, yhat, proba,
                                                         classes if classes is not None else unique_labels,
                                                         label_kind, metrics_to_compute)
                metrics_dict = {k: float(v) if v is not None and not isinstance(v, (str, bytes)) else v for k, v in
                                metrics_dict.items()}
                metrics_dict.update({"fold": fold_idx + 1, "n_val": int(len(va))})

                # Debug metrics
                print(f"Fold {fold_idx + 1} metrics: {metrics_dict}")
                fold_rows.append(metrics_dict)
                per_fold_preds.append(pd.DataFrame({"fold": fold_idx + 1, "y_true": yva, "y_pred": yhat}))

                # Store probability outputs when available (binary: 1D; multiclass: 2D)
                if proba is not None:
                    if proba.ndim == 1:
                        per_fold_probas.append(pd.DataFrame({"fold": fold_idx + 1, "score": proba}))
                    else:
                        # Multiclass — one column per class
                        prob_df = pd.DataFrame(proba, columns=[f"p[{c}]" for c in
                                                               (classes if classes is not None else unique_labels)])
                        prob_df.insert(0, "fold", fold_idx + 1)
                        per_fold_probas.append(prob_df)

            # Collate results
            if len(fold_rows) == 0:
                st.error("No successful folds were completed. Please adjust settings and try again.")
            else:
                results_df = pd.DataFrame(fold_rows).set_index("fold").sort_index()
                metric_cols = [m for m in metrics_to_compute if m in results_df.columns]

                # Summary (mean ± std)
                summary = pd.DataFrame({
                    "mean": results_df[metric_cols].mean(axis=0, skipna=True),
                    "std": results_df[metric_cols].std(axis=0, ddof=1, skipna=True),
                })

                st.subheader("Results — Per Fold")
                st.dataframe(results_df[metric_cols + ["n_val"]], use_container_width=True)

                st.subheader("Results — Summary")
                st.dataframe(summary, use_container_width=True)

                # Downloads
                csv_results = results_df.to_csv(index=True)
                st.download_button("Download per-fold metrics (CSV)", data=csv_results,
                                   file_name="cv_metrics_folds.csv", mime="text/csv")

                csv_summary = summary.to_csv(index=True)
                st.download_button("Download summary metrics (CSV)", data=csv_summary,
                                   file_name="cv_metrics_summary.csv", mime="text/csv")

                # Confusion matrix (aggregated)
                with st.expander("Aggregated Confusion Matrix (across folds)", expanded=False):
                    cm_df = pd.DataFrame(cm_total, index=[f"true={c}" for c in unique_labels],
                                         columns=[f"pred={c}" for c in unique_labels])
                    st.dataframe(cm_df, use_container_width=True)

                    # Optional heatmap for smaller label spaces
                    if len(unique_labels) <= 15:
                        fig = plt.figure()
                        plt.imshow(cm_total, interpolation='nearest')
                        plt.title('Aggregated Confusion Matrix')
                        plt.xticks(ticks=np.arange(len(unique_labels)), labels=[str(c) for c in unique_labels],
                                   rotation=45, ha='right')
                        plt.yticks(ticks=np.arange(len(unique_labels)), labels=[str(c) for c in unique_labels])
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.colorbar()
                        st.pyplot(fig)

                # Persist to session_state for downstream steps
                st.session_state.cv_results_df = results_df
                st.session_state.cv_summary_df = summary
                st.session_state.cv_confusion_matrix = cm_total
                st.success("Cross-validation complete.")

    # ===========================
    # Step 11 — (Placeholder) Regression metrics UI
    # ===========================
    if st.session_state.task_type == "regression":
        st.info(
            "Regression metrics selection & evaluation will be implemented next. Suggested metrics: RMSE, MAE, R², MAPE.")

    # ===========================
    # Step 12 — Train on all data & report metrics
    # ===========================
    st.subheader("Step 12: Train on all data & report metrics")

    run_full = st.button("Run Model on Full Dataset", type="primary")

    if run_full:
        # Basic guardrails
        if "estimator" not in st.session_state:
            st.error("Please configure a model in Step 8 first.")
        elif "X_df" not in st.session_state or st.session_state.X_df is None:
            st.error("Please build X in Step 5 first.")
        elif "y" not in st.session_state or st.session_state.y is None:
            st.error("Please build y in Step 5 first.")
        else:
            from sklearn.base import clone
            import numpy as np
            import pandas as pd
            from math import sqrt

            # Fallback helper if not defined (e.g., task_type != 'classification' earlier)
            if "_predict_proba_like" not in globals():
                def _sigmoid(x):
                    return 1.0 / (1.0 + np.exp(-x))


                def _softmax(Z):
                    Z = Z - np.max(Z, axis=1, keepdims=True)
                    e = np.exp(Z)
                    return e / np.sum(e, axis=1, keepdims=True)


                def _predict_proba_like(est, X):
                    classes = getattr(est, "classes_", None)
                    if hasattr(est, "predict_proba"):
                        P = est.predict_proba(X)
                        if P.ndim == 2 and P.shape[1] == 2:
                            return P[:, 1], est.classes_
                        return P, est.classes_
                    if hasattr(est, "decision_function"):
                        S = est.decision_function(X)
                        if np.ndim(S) == 1:
                            return _sigmoid(S), classes
                        return _softmax(S), classes
                    return None, classes

            # Gather data and model
            X = st.session_state.X_df.values
            y = st.session_state.y.values
            model = clone(st.session_state.estimator)

            try:
                st.info(f"Training model on full dataset: {X.shape[0]} samples, {X.shape[1]} features")
                model.fit(X, y)
                st.success("Model training completed successfully!")
            except Exception as fit_err:
                st.error(f"Model fit failed: {fit_err}")
            else:
                try:
                    yhat = model.predict(X)

                    # Critical debugging information
                    st.info(f"Prediction completed. Shape: {yhat.shape}")
                    st.info(f"Unique values in y (true): {np.unique(y)}")
                    st.info(f"Unique values in yhat (predicted): {np.unique(yhat)}")

                    # Check if model is just predicting one class
                    if len(np.unique(yhat)) == 1:
                        st.error(f"⚠️ **PROBLEM DETECTED**: Model is predicting only one class: {np.unique(yhat)[0]}")
                        st.error(
                            "This explains why you're getting perfect accuracy (1.0) - the model isn't learning properly!")

                    # Check for perfect predictions (overfitting indicator)
                    accuracy = np.mean(y == yhat)
                    if accuracy >= 0.99:
                        st.warning(f"⚠️ **OVERFITTING DETECTED**: Training accuracy = {accuracy:.4f}")
                        st.warning(
                            "Perfect or near-perfect training accuracy often indicates overfitting or data leakage.")

                    # Show prediction details
                    with st.expander("🔍 Detailed Prediction Analysis", expanded=False):
                        st.write("**First 20 predictions vs actual:**")
                        comparison_df = pd.DataFrame({
                            'Actual': y[:20],
                            'Predicted': yhat[:20],
                            'Correct': y[:20] == yhat[:20]
                        })
                        st.dataframe(comparison_df)

                        st.write("**Prediction Summary:**")
                        st.write(f"- Total samples: {len(y)}")
                        st.write(f"- Correct predictions: {np.sum(y == yhat)}")
                        st.write(f"- Incorrect predictions: {np.sum(y != yhat)}")
                        st.write(f"- Accuracy: {accuracy:.4f}")

                        # Check for systematic prediction patterns
                        if len(np.unique(yhat)) == 1:
                            st.error(
                                f"🚨 **ROOT CAUSE FOUND**: Model predicts ONLY class '{np.unique(yhat)[0]}' for all samples!")
                            st.error(
                                "**Solution**: Check for class imbalance, adjust model parameters, or try a different algorithm.")

                except Exception as pred_err:
                    st.error(f"Prediction failed: {pred_err}")
                    yhat = None

                if st.session_state.task_type == "classification" and yhat is not None:
                    # Collect probabilities if available
                    proba, classes = _predict_proba_like(model, X)
                    # Pull selected metrics from Step 9
                    selected = st.session_state.get("selected_metrics", [])
                    if not isinstance(selected, list) or len(selected) == 0:
                        st.warning("No metrics selected in Step 9. Please select them above.")
                    else:
                        # Use previously defined fold metric computer
                        label_kind = st.session_state.get("class_label_kind", "binary")
                        # If helper was not defined (shouldn't happen for classification), define a minimal inline fallback
                        if "_compute_metrics_one_fold" not in globals():
                            from sklearn.metrics import (
                                accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                                roc_auc_score, average_precision_score, log_loss, brier_score_loss, jaccard_score,
                                fowlkes_mallows_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                                precision_recall_fscore_support
                            )
                            from sklearn.preprocessing import label_binarize


                            def _binary_conf_counts(y_true, y_pred, pos_label=None):
                                labels = np.unique(np.concatenate([y_true, y_pred]))
                                if pos_label is None:
                                    pos_label = labels[-1]
                                neg_label = [l for l in labels if l != pos_label][0]
                                cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
                                tn, fp, fn, tp = cm.ravel()
                                return tp, fp, tn, fn


                            def _ks_stat(y_true, y_score):
                                y_true = np.asarray(y_true)
                                y_score = np.asarray(y_score)
                                pos_scores = y_score[y_true == y_true.max()]
                                neg_scores = y_score[y_true != y_true.max()]
                                grid = np.sort(np.unique(np.concatenate([pos_scores, neg_scores])))

                                def ecdf(vals, grid):
                                    if len(vals) == 0:
                                        return np.zeros_like(grid, dtype=float)
                                    ranks = np.searchsorted(np.sort(vals), grid, side="right")
                                    return ranks / float(len(vals))

                                cdf_pos = ecdf(pos_scores, grid)
                                cdf_neg = ecdf(neg_scores, grid)
                                return float(np.max(np.abs(cdf_pos - cdf_neg)))


                            def _brier_multiclass(y_true, proba, classes):
                                Y = label_binarize(y_true, classes=classes)
                                return float(np.mean(np.sum((Y - proba) ** 2, axis=1)))


                            def _compute_metrics_one_fold(y_true, y_pred, proba, classes, label_kind, selected):
                                m = {}
                                needs_proba = {"ROC AUC", "PR AUC (Average Precision)", "PR AUC (macro)",
                                               "PR AUC (micro)", "Log Loss", "Brier Score"}
                                # shortcuts
                                if label_kind == "binary":
                                    tp, fp, tn, fn = _binary_conf_counts(y_true, y_pred)
                                    tpr = tp / (tp + fn) if (tp + fn) else 0.0
                                    tnr = tn / (tn + fp) if (tn + fp) else 0.0
                                    fpr = 1.0 - tnr
                                    fnr = 1.0 - tpr
                                    ppv = tp / (tp + fp) if (tp + fp) else 0.0
                                    npv = tn / (tn + fn) if (tn + fn) else 0.0
                                    fdr = 1.0 - ppv
                                    _for = fn / (fn + tn) if (fn + tn) else 0.0
                                    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
                                    youden = tpr + tnr - 1.0
                                    markedness = ppv + npv - 1.0
                                for name in selected:
                                    try:
                                        if name == "Accuracy":
                                            m[name] = accuracy_score(y_true, y_pred)
                                        elif name in ("F1", "Dice Coefficient"):
                                            avg = "binary" if label_kind == "binary" else "macro"
                                            m[name] = f1_score(y_true, y_pred, average=avg)
                                        elif name == "Balanced Accuracy":
                                            m[name] = balanced_accuracy_score(y_true, y_pred)
                                        elif name == "Jaccard (IoU)":
                                            avg = "binary" if label_kind == "binary" else "macro"
                                            m[name] = jaccard_score(y_true, y_pred, average=avg)
                                        elif name in ("MCC", "Multiclass MCC"):
                                            m[name] = matthews_corrcoef(y_true, y_pred)
                                        elif name == "Cohen’s κ":
                                            m[name] = cohen_kappa_score(y_true, y_pred)
                                        elif label_kind == "binary" and name == "Precision (PPV)":
                                            m[name] = precision_score(y_true, y_pred, zero_division=0)
                                        elif label_kind == "binary" and name == "Recall (TPR/Sensitivity)":
                                            m[name] = recall_score(y_true, y_pred, zero_division=0)
                                        elif label_kind == "binary" and name == "Specificity (TNR)":
                                            m[name] = tnr
                                        elif label_kind == "binary" and name == "FPR":
                                            m[name] = fpr
                                        elif label_kind == "binary" and name == "FNR":
                                            m[name] = fnr
                                        elif label_kind == "binary" and name == "Youden’s J (Informedness)":
                                            m[name] = youden
                                        elif label_kind == "binary" and name == "Markedness":
                                            m[name] = markedness
                                        elif label_kind == "binary" and name == "Threat Score (CSI)":
                                            m[name] = csi
                                        elif label_kind == "binary" and name == "KS Statistic":
                                            m[name] = np.nan if proba is None else _ks_stat(y_true, proba)
                                        elif name in needs_proba:
                                            if proba is None:
                                                m[name] = np.nan
                                            else:
                                                if name == "ROC AUC":
                                                    if label_kind == "binary":
                                                        m[name] = roc_auc_score(y_true, proba)
                                                    else:
                                                        m[name] = roc_auc_score(y_true, proba, multi_class="ovr",
                                                                                average="macro")
                                                elif name == "PR AUC (Average Precision)":
                                                    if label_kind == "binary":
                                                        m[name] = average_precision_score(y_true, proba)
                                                    else:
                                                        Y = label_binarize(y_true, classes=classes)
                                                        m[name] = float(np.mean(
                                                            [average_precision_score(Y[:, i], proba[:, i]) for i in
                                                             range(proba.shape[1])]))
                                                elif name == "PR AUC (macro)":
                                                    Y = label_binarize(y_true, classes=classes)
                                                    m[name] = float(np.mean(
                                                        [average_precision_score(Y[:, i], proba[:, i]) for i in
                                                         range(proba.shape[1])]))
                                                elif name == "PR AUC (micro)":
                                                    Y = label_binarize(y_true, classes=classes)
                                                    m[name] = average_precision_score(Y.ravel(), proba.ravel())
                                                elif name == "Log Loss":
                                                    if label_kind == "binary":
                                                        P = np.vstack([1 - proba, proba]).T
                                                        m[name] = log_loss(y_true, P, labels=classes)
                                                    else:
                                                        m[name] = log_loss(y_true, proba, labels=classes)
                                                elif name == "Brier Score":
                                                    if label_kind == "binary":
                                                        m[name] = brier_score_loss(y_true, proba)
                                                    else:
                                                        m[name] = _brier_multiclass(y_true, proba, classes)
                                    except Exception:
                                        m[name] = np.nan
                                return m
                        # Compute and display
                        unique_labels = np.unique(y[~pd.isnull(y)])
                        metrics_full = _compute_metrics_one_fold(
                            y_true=y,
                            y_pred=yhat,
                            proba=proba,
                            classes=(classes if classes is not None else unique_labels),
                            label_kind=label_kind,
                            selected=selected,
                        )
                        # Order columns by user's selection
                        ordered = {k: metrics_full.get(k, np.nan) for k in selected}
                        res_df = pd.DataFrame([ordered])
                        st.subheader("Metrics on Training Data (fit on all rows)")
                        st.dataframe(res_df, use_container_width=True)
                        st.download_button(
                            "Download metrics (CSV)",
                            data=res_df.to_csv(index=False),
                            file_name="train_full_metrics.csv",
                            mime="text/csv",
                        )
                        st.session_state.train_full_metrics = res_df

                elif st.session_state.task_type == "regression" and yhat is not None:
                    # Basic regression metrics since selection UI is not yet implemented
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                    # RMSE
                    rmse = sqrt(mean_squared_error(y, yhat))
                    mae = mean_absolute_error(y, yhat)
                    r2 = r2_score(y, yhat)
                    # Safe MAPE (ignore zeros in denominator)
                    denom = np.where(np.abs(y) < 1e-8, np.nan, np.abs(y))
                    mape = float(np.nanmean(np.abs((y - yhat) / denom)) * 100.0)
                    reg_df = pd.DataFrame([
                        {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mape}
                    ])
                    st.subheader("Regression Metrics on Training Data (fit on all rows)")
                    st.dataframe(reg_df, use_container_width=True)
                    st.download_button(
                        "Download regression metrics (CSV)",
                        data=reg_df.to_csv(index=False),
                        file_name="train_full_regression_metrics.csv",
                        mime="text/csv",
                    )
                    st.session_state.train_full_regression_metrics = reg_df


