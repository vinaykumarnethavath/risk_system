"""
Hybrid Risk Scoring — Forensic Detection Module 
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


# ─── Forensic Statistics ─────────────────────────────────────────────


def get_benford_score(prices: pd.Series) -> float:
    """
    Forensic signal: detect non-natural price distributions
    indicative of manual manipulation using Benford's Law.

    Returns the L2-norm deviation from the expected first-digit
    distribution.
    """

    def first_digit(n):
        if pd.isna(n) or n <= 0:
            return np.nan
        return int(str(n)[0])

    digits = prices.apply(first_digit).dropna()
    if len(digits) < 12:
        return 0

    obs = digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    exp = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.linalg.norm(obs - exp))


# ─── Feature Engineering ─────────────────────────────────────────────


def engineer_signals(
    data: pd.DataFrame, reference: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Encapsulated feature engineering to ensure train/test consistency.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to transform.
    reference : pd.DataFrame, optional
        The training set used to compute lookup statistics.
        If None, ``data`` is used as its own reference (training mode).

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix.
    """
    X = pd.DataFrame(index=data.index)
    ref = reference if reference is not None else data

    # 1. Benford forensic score per supply type
    benford_map = (
        ref.groupby("tender_supplytype")["bid_price"]
        .apply(get_benford_score)
        .to_dict()
    )
    X["forensic_benford_score"] = (
        data["tender_supplytype"].map(benford_map).fillna(0)
    )

    # 2. Market capture HHI per buyer
    hhi_map = (
        ref.groupby("buyer_id")["bidder_id"]
        .apply(lambda x: np.sum(x.value_counts(normalize=True) ** 2))
        .to_dict()
    )
    X["market_capture_hhi"] = data["buyer_id"].map(hhi_map).fillna(0)

    # 3. Price z-score within CPV category
    data_log = np.log1p(data["bid_price"])
    ref_log = np.log1p(ref["bid_price"])
    cpv_stats = (
        ref.assign(lp=ref_log)
        .groupby("tender_maincpv")["lp"]
        .agg(["mean", "std"])
    )
    joined = data.join(cpv_stats, on="tender_maincpv")
    X["price_z_score"] = (data_log - joined["mean"]) / joined["std"].replace(0, 1)
    X["price_z_score"] = X["price_z_score"].fillna(0)

    # 4. Decision latency
    dead = pd.to_datetime(data["tender_biddeadline"], errors="coerce")
    award = pd.to_datetime(data["tender_awarddecisiondate"], errors="coerce")
    X["decision_latency_days"] = (award - dead).dt.days.fillna(-1)

    # 5. Competition and complexity signals
    X["bidder_count"] = data["lot_bidscount"].fillna(0)
    X["desc_complexity"] = data["tender_description_length"].fillna(0)
    X["sme_participation"] = (
        data["lot_smebidscount"] / data["lot_bidscount"].replace(0, 1)
    ).fillna(0)

    # 6. Buyer market presence
    X["buyer_market_presence"] = (
        data["buyer_id"]
        .map(ref["buyer_id"].value_counts(normalize=True))
        .fillna(0)
    )

    return X.replace([np.inf, -np.inf], 0)


# ─── Tier-1 Research Pipeline ────────────────────────────────────────


def run_tier_1_research_pipeline(file_path: str) -> dict:
    """
    Full forensic risk scoring pipeline with temporal validation.

    Parameters
    ----------
    file_path : str
        Path to the unprocessed procurement dataset CSV.

    Returns
    -------
    dict
        Dictionary of evaluation metrics (ROC-AUC, PR-AUC, capture, lift).
    """
    df = pd.read_csv(file_path)

    # Temporal split
    train_df = df[df["tender_year"] <= 2018].copy()
    test_df = df[df["tender_year"] >= 2019].copy()

    y_train = (train_df["cri"] > 0.5).astype(int)
    y_test = (test_df["cri"] > 0.5).astype(int)

    # Engineer features
    X_train = engineer_signals(train_df)
    X_test = engineer_signals(test_df, reference=train_df)

    # Hybrid anomaly scoring
    iso = IsolationForest(contamination=0.1, random_state=42)
    X_train["hybrid_anomaly_score"] = iso.fit_predict(
        X_train[["bidder_count", "desc_complexity"]]
    )
    X_test["hybrid_anomaly_score"] = iso.predict(
        X_test[["bidder_count", "desc_complexity"]]
    )

    # Supervised model
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, l2_regularization=2.0
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # Decile analysis
    eval_df = (
        pd.DataFrame({"actual": y_test, "prob": probs})
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )
    eval_df["decile"] = pd.qcut(eval_df.index, 10, labels=range(1, 11))

    decile_report = (
        eval_df.groupby("decile")
        .agg(
            avg_risk_prob=("prob", "mean"),
            risk_found=("actual", "sum"),
            total_audited=("actual", "count"),
        )
        .reset_index()
    )
    decile_report["cumulative_capture"] = (
        decile_report["risk_found"].cumsum() / y_test.sum()
    ) * 100
    decile_report["lift"] = (
        decile_report["risk_found"] / decile_report["total_audited"]
    ) / y_test.mean()
    decile_report.to_csv("tier_1_decile_report.csv", index=False)

    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    p, r, _ = precision_recall_curve(y_test, probs)
    ax1.plot(
        r, p, color="darkblue", lw=2,
        label=f"PR-AUC: {average_precision_score(y_test, probs):.3f}",
    )
    ax1.set_title("Out-of-Time Performance: Precision-Recall (2019-2021)")
    ax1.set_xlabel("Recall (Detection Rate)")
    ax1.set_ylabel("Precision (Audit Accuracy)")
    ax1.legend()

    p_true, p_pred = calibration_curve(y_test, probs, n_bins=10)
    ax2.plot(p_pred, p_true, marker="s", color="darkred", label="Temporal Model")
    ax2.plot([0, 1], [0, 1], "--k", label="Ideal")
    ax2.set_title("Temporal Risk Calibration (Reliability Diagram)")
    ax2.set_xlabel("Predicted Risk")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    plt.savefig("tier_1_research_dashboard.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Export scores
    final_output = pd.DataFrame({
        "tender_id": test_df["tender_id"],
        "risk_prob": probs,
        "risk_score": (probs * 100).astype(int),
        "actual_label": y_test,
    })
    final_output.to_csv("tier_1_risk_scores.csv", index=False)

    return {
        "ROC-AUC (2019-2021)": round(roc_auc_score(y_test, probs), 4),
        "PR-AUC (2019-2021)": round(average_precision_score(y_test, probs), 4),
        "Top_20pct_Capture": f"{round(decile_report.iloc[1]['cumulative_capture'], 1)}%",
        "Model_Alpha_Lift": round(decile_report.iloc[0]["lift"], 2),
    }


if __name__ == "__main__":
    metrics = run_tier_1_research_pipeline(
        "IS_DIB_2023(unprocessed dataset).csv"
    )
    print(json.dumps(metrics, indent=2))
