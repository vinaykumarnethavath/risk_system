"""
FASS — Fraud-Adjusted Scalability Scoring (Legacy Module)
"""

import os
import re
import sys
import json
import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve


# ─── Environment Helpers ─────────────────────────────────────────────


def in_colab() -> bool:
    """Check if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def prompt_upload_if_needed(path: str) -> None:
    """Prompt user to upload input.csv when running in Colab."""
    if os.path.exists(path):
        return
    if in_colab():
        print("input.csv not found. Please upload input.csv")
        from google.colab import files  # noqa: F401
        files.upload()
    else:
        print("input.csv not found in current folder.")
        print("Place input.csv next to this script and run again.")
        sys.exit(1)


# ─── Feature Engineering Helpers ─────────────────────────────────────


def to_year(s):
    """Extract a 4-digit year from an arbitrary string."""
    try:
        s = str(s)
        m = re.search(r"(19|20)\d{2}", s)
        return int(m.group(0)) if m else np.nan
    except Exception:
        return np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]+", "", regex=True),
        errors="coerce",
    )


def pick_target(df: pd.DataFrame) -> str:
    """Auto-detect the target column from common naming conventions."""
    cols = {c.lower(): c for c in df.columns}
    for k in ["status", "acquired", "label", "is_success", "success"]:
        if k in cols:
            return cols[k]
    raise ValueError(
        "Target column not found. Expected one of: "
        "status, acquired, label, is_success, success"
    )


def map_target(y_raw: pd.Series) -> pd.Series:
    """Map raw target values to binary 0/1."""
    y = y_raw.astype(str).str.lower().str.strip()
    y = y.map(
        lambda v: 1
        if v in ["acquired", "ipo", "1", "true", "success", "successful"]
        else 0
    )
    return y.astype(int)


def drop_leakage_like(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Remove columns that could leak target information."""
    bad_keys = [
        "status", "acquired", "acquisition", "closed", "shutdown",
        "ipo", "went_public", "exit", "closed_at", "acquired_at", "ipo_date",
    ]
    keep = []
    for c in df.columns:
        if c == target_col:
            continue
        low = c.lower()
        if any(k in low for k in bad_keys):
            continue
        keep.append(c)
    return df[keep].copy()


def add_light_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineer lightweight features from raw columns."""
    X = X.copy()

    # Extract year-like values from date/text columns
    for c in list(X.columns):
        if X[c].dtype == object:
            if any(
                k in c.lower()
                for k in ["founded", "first", "last", "closed", "ipo", "year", "date"]
            ):
                X[c + "_year"] = X[c].apply(to_year)
            X[c] = X[c].astype(str).replace({"nan": np.nan})

    # Coerce numeric-like text columns
    num_like = []
    for c in list(X.columns):
        if X[c].dtype == object:
            sample = "".join(X[c].astype(str).head(50).tolist())
            if re.search(r"\d", sample):
                num_like.append(c)
    for c in num_like:
        X[c + "_num"] = coerce_numeric(X[c])

    # Sector-based fraud prior
    def sector_prior(row, text_cols):
        text = " ".join([str(row.get(c, "")) for c in text_cols]).lower()
        s = 1
        if any(
            w in text
            for w in [
                "fintech", "finance", "payment", "payments", "wallet",
                "lending", "insurance", "trading", "crypto", "bank",
            ]
        ):
            s = 3
        elif any(w in text for w in ["ecommerce", "marketplace", "retail", "auction"]):
            s = 2
        elif any(w in text for w in ["health", "pharma", "biotech", "medical"]):
            s = 2
        elif any(w in text for w in ["gaming", "bet", "gamble"]):
            s = 2
        return s

    text_cols = [
        c for c in X.columns
        if any(
            k in c.lower()
            for k in ["category", "market", "industry", "sector", "type", "tags"]
        )
    ]
    X["SectorFraudPrior"] = X.apply(lambda r: sector_prior(r, text_cols), axis=1)
    return X


# ─── Model Pipeline ─────────────────────────────────────────────────


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """Build a scikit-learn preprocessing + GBM classification pipeline."""
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    pre = ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        sparse_threshold=0.3,
    )
    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


# ─── Evaluation Helpers ─────────────────────────────────────────────


def choose_threshold(
    y_true: np.ndarray, p: np.ndarray, target_acc: float = 0.96
) -> Tuple[float, float]:
    """Find the probability threshold closest to the target accuracy."""
    ths = np.arange(0.01, 0.99, 0.001)
    accs = np.array([accuracy_score(y_true, (p >= t).astype(int)) for t in ths])
    band = (accs >= target_acc - 0.01) & (accs <= target_acc + 0.01)
    if band.any():
        idx = np.argmin(np.abs(accs[band] - target_acc))
        thr = ths[band][idx]
        acc = accs[band][idx]
    else:
        idx = np.argmax(accs)
        thr = ths[idx]
        acc = accs[idx]
    return float(thr), float(acc)


def top_decile_lift(y_true: np.ndarray, p: np.ndarray) -> float:
    """Compute lift in the top 10% of predicted probabilities."""
    n = len(y_true)
    k = max(1, int(0.10 * n))
    order = np.argsort(-p)
    top = y_true[order][:k].mean()
    avg = y_true.mean()
    return float(top / avg) if avg > 0 else float("nan")


def export_threshold_sweep(
    y_true: np.ndarray, p: np.ndarray, path: str
) -> None:
    """Export accuracy at various thresholds to CSV."""
    ths = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in ths:
        yhat = (p >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "accuracy": round(float(accuracy_score(y_true, yhat)), 4),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_reliability_plot(
    y_true: np.ndarray, p: np.ndarray, path: str
) -> None:
    """Save a reliability (calibration) diagram as PNG."""
    import matplotlib.pyplot as plt

    fr, ob = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Ideal")
    plt.plot(fr, ob, marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def model_summary(pipe: Pipeline, X_fit: pd.DataFrame, topk: int = 20) -> pd.DataFrame:
    """Extract top-k most important features from the fitted pipeline."""
    try:
        pre: ColumnTransformer = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        feat_names = []
        num_names = pre.transformers_[0][2]
        feat_names.extend(list(num_names))
        oh: OneHotEncoder = pre.named_transformers_["cat"].named_steps["oh"]
        cat_raw = pre.transformers_[1][2]
        oh_names = oh.get_feature_names_out(cat_raw)
        feat_names.extend(list(oh_names))
        importances = clf.feature_importances_
        idx = np.argsort(-np.abs(importances))[:topk]
        out = pd.DataFrame({
            "feature": np.array(feat_names)[idx],
            "importance": importances[idx],
        })
        return out
    except Exception:
        return pd.DataFrame()


# ─── Main Entry Point ────────────────────────────────────────────────


def main():
    """Run the full FASS pipeline: load → engineer → train → evaluate → export."""
    warnings.filterwarnings("ignore")
    csv_path = "input.csv"
    prompt_upload_if_needed(csv_path)

    # Load data
    df = pd.read_csv(csv_path)
    tcol = pick_target(df)
    y = map_target(df[tcol])
    X = df.drop(columns=[tcol]).copy()
    X = drop_leakage_like(pd.concat([X], axis=1), target_col=tcol)

    # Feature engineering
    X = add_light_features(X)

    # Train / validation split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build and fit pipeline
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_tr, y_tr)

    # Calibrate
    pipe = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    pipe.fit(X_tr, y_tr)

    # Evaluate
    p_va = pipe.predict_proba(X_va)[:, 1]
    thr, acc_va = choose_threshold(y_va.values, p_va, target_acc=0.96)

    auc = roc_auc_score(y_va, p_va)
    aupr = average_precision_score(y_va, p_va)
    brier = brier_score_loss(y_va, p_va)
    lift10 = top_decile_lift(y_va.values, p_va)

    yhat_va = (p_va >= thr).astype(int)
    cm = confusion_matrix(y_va, yhat_va)

    print("Validation Metrics")
    print(json.dumps({
        "accuracy": round(float(acc_va), 3),
        "threshold": round(float(thr), 3),
        "auc": round(float(auc), 3),
        "aupr": round(float(aupr), 3),
        "brier": round(float(brier), 3),
        "lift_top_10pct": round(float(lift10), 3) if not math.isnan(lift10) else None,
    }, indent=2))

    print("Confusion matrix [TN, FP, FN, TP]:")
    print(cm.ravel().tolist())

    # Exports
    export_threshold_sweep(y_va.values, p_va, "threshold_sweep.csv")
    try:
        export_reliability_plot(y_va.values, p_va, "reliability.png")
    except Exception:
        print("Could not save reliability plot.")

    # Score full dataset
    p_all = pipe.predict_proba(X)[:, 1]
    pred_all = (p_all >= thr).astype(int)
    risk = np.rint(100 * (1 - p_all)).astype(int)
    bucket = pd.cut(
        risk, bins=[-1, 33, 66, 101], labels=["Low", "Medium", "High"]
    ).astype(str)

    name_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["name", "company", "org", "startup"])
    ]
    id_cols = name_cols[:1] if name_cols else []

    out = pd.DataFrame({
        **({id_cols[0]: df[id_cols[0]].astype(str)} if id_cols else {}),
        "pred": pred_all,
        "prob_success": p_all,
        "risk_score": risk,
        "bucket": bucket,
    })
    out.to_csv("output.csv", index=False)
    print("Wrote output.csv")
    print("Wrote threshold_sweep.csv")
    print("Wrote reliability.png")

    fs = model_summary(pipe, X_tr, topk=20)
    if not fs.empty:
        fs.to_csv("top_features.csv", index=False)
        print("Wrote top_features.csv")


if __name__ == "__main__":
    main()
