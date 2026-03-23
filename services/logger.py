"""
logger.py — Structured Logging and Model Evaluation Metrics
============================================================
Provides a unified logging framework for the Corporate Intelligence
platform with structured output, timing decorators, and model
evaluation metric computation.
"""

import os
import sys
import time
import json
import logging
import functools
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd


# ─── Logger Setup ───────────────────────────────────────────────────


def setup_logger(
    name: str = "corporate_intel",
    level: str = "INFO",
    log_file: str = None,
) -> logging.Logger:
    """
    Create a configured logger with structured formatting.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str, optional
        Path to log file. If None, logs to stdout only.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger
logger = setup_logger()


# ─── Timing Decorator ──────────────────────────────────────────────


def timed(func: Callable = None, *, label: str = None) -> Callable:
    """
    Decorator to log execution time of a function.

    Usage:
        @timed
        def my_function(): ...

        @timed(label="Custom Label")
        def my_function(): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_label = label or fn.__name__
            logger.info(f"[START] {fn_label}")
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"[DONE]  {fn_label} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"[FAIL]  {fn_label} ({elapsed:.2f}s) — {type(e).__name__}: {e}")
                raise
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ─── Model Evaluation Metrics ──────────────────────────────────────


class EvaluationMetrics:
    """
    Compute and store model evaluation metrics for scoring engines.
    Tracks performance across multiple runs for comparison.
    """

    def __init__(self):
        self.history = []

    def compute_scoring_metrics(
        self,
        predicted_scores: np.ndarray,
        actual_outcomes: np.ndarray = None,
        label: str = "run",
    ) -> dict:
        """
        Compute distribution and quality metrics for scoring output.

        Parameters
        ----------
        predicted_scores : np.ndarray
            Model output scores (0–100 for TSS, 0–1 for probabilities).
        actual_outcomes : np.ndarray, optional
            Ground truth labels (1 = failure, 0 = success) for
            supervised evaluation.
        label : str
            Label for this evaluation run.

        Returns
        -------
        dict
            Comprehensive metrics dictionary.
        """
        metrics = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(predicted_scores),
        }

        # Distribution metrics
        metrics["distribution"] = {
            "mean": round(float(np.mean(predicted_scores)), 4),
            "std": round(float(np.std(predicted_scores)), 4),
            "median": round(float(np.median(predicted_scores)), 4),
            "min": round(float(np.min(predicted_scores)), 4),
            "max": round(float(np.max(predicted_scores)), 4),
            "q25": round(float(np.percentile(predicted_scores, 25)), 4),
            "q75": round(float(np.percentile(predicted_scores, 75)), 4),
            "iqr": round(float(np.percentile(predicted_scores, 75) - np.percentile(predicted_scores, 25)), 4),
            "skewness": round(float(pd.Series(predicted_scores).skew()), 4),
            "kurtosis": round(float(pd.Series(predicted_scores).kurtosis()), 4),
        }

        # Bucket distribution
        if np.max(predicted_scores) <= 1.0:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        else:
            bins = [0, 20, 40, 60, 80, 100]
            labels = ["F/D", "C", "B", "B+/A", "A/A+"]

        bucket_counts = pd.cut(predicted_scores, bins=bins, labels=labels).value_counts()
        metrics["bucket_distribution"] = bucket_counts.to_dict()

        # Supervised metrics (if ground truth available)
        if actual_outcomes is not None and len(actual_outcomes) == len(predicted_scores):
            # Rank-based metrics
            sorted_idx = np.argsort(-predicted_scores)
            sorted_actual = actual_outcomes[sorted_idx]

            # Top-k capture rates
            for k_pct in [10, 20, 30]:
                k = max(1, int(len(predicted_scores) * k_pct / 100))
                top_k_actual = sorted_actual[:k]
                capture = float(top_k_actual.sum() / max(1, actual_outcomes.sum()))
                metrics[f"top_{k_pct}pct_capture"] = round(capture, 4)

            # Lift
            overall_rate = float(actual_outcomes.mean())
            if overall_rate > 0:
                top_10_rate = float(sorted_actual[:max(1, len(sorted_actual)//10)].mean())
                metrics["top_decile_lift"] = round(top_10_rate / overall_rate, 4)

            # Gini coefficient
            metrics["gini_coefficient"] = round(
                self._compute_gini(actual_outcomes, predicted_scores), 4
            )

        self.history.append(metrics)
        return metrics

    @staticmethod
    def _compute_gini(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute normalised Gini coefficient."""
        try:
            sorted_idx = np.argsort(predicted)
            sorted_actual = actual[sorted_idx]
            n = len(sorted_actual)
            cumsum = np.cumsum(sorted_actual)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_actual))) / (n * np.sum(sorted_actual)) - (n + 1) / n
            return float(gini)
        except Exception:
            return 0.0

    def compare_runs(self) -> pd.DataFrame:
        """
        Compare metrics across all recorded evaluation runs.

        Returns
        -------
        pd.DataFrame
            Comparison table with one row per run.
        """
        if not self.history:
            return pd.DataFrame()

        rows = []
        for h in self.history:
            row = {
                "label": h["label"],
                "timestamp": h["timestamp"],
                "n_samples": h["n_samples"],
                "mean_score": h["distribution"]["mean"],
                "std_score": h["distribution"]["std"],
                "median_score": h["distribution"]["median"],
            }
            for key in ["top_10pct_capture", "top_20pct_capture", "gini_coefficient", "top_decile_lift"]:
                if key in h:
                    row[key] = h[key]
            rows.append(row)

        return pd.DataFrame(rows)

    def export_history(self, path: str) -> None:
        """Export evaluation history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Exported evaluation history to {path}")


# ─── Pipeline Logger ───────────────────────────────────────────────


class PipelineLogger:
    """
    Track pipeline execution with timing and status for each step.
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.steps = []
        self.start_time = time.perf_counter()
        logger.info(f"Pipeline '{pipeline_name}' started")

    def log_step(self, step_name: str, status: str = "OK", details: dict = None):
        """Log a completed pipeline step."""
        elapsed = time.perf_counter() - self.start_time
        entry = {
            "step": step_name,
            "status": status,
            "elapsed_total": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            entry["details"] = details
        self.steps.append(entry)

        icon = "✓" if status == "OK" else ("⚠" if status == "WARN" else "✗")
        logger.info(f"  {icon} {step_name} [{status}] ({elapsed:.2f}s total)")

    def summary(self) -> dict:
        """Return pipeline execution summary."""
        total = time.perf_counter() - self.start_time
        ok = sum(1 for s in self.steps if s["status"] == "OK")
        warn = sum(1 for s in self.steps if s["status"] == "WARN")
        fail = sum(1 for s in self.steps if s["status"] == "FAIL")

        summary = {
            "pipeline": self.pipeline_name,
            "total_time": round(total, 2),
            "steps_total": len(self.steps),
            "steps_ok": ok,
            "steps_warn": warn,
            "steps_fail": fail,
            "steps": self.steps,
        }

        status = "SUCCESS" if fail == 0 else "FAILED"
        logger.info(
            f"Pipeline '{self.pipeline_name}' {status} "
            f"({ok} ok, {warn} warn, {fail} fail) in {total:.2f}s"
        )
        return summary

    def export(self, path: str) -> None:
        """Export pipeline log to JSON."""
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, default=str)


if __name__ == "__main__":
    # Demo logging
    logger.info("Testing logger setup")
    logger.warning("This is a warning")

    # Demo timed decorator
    @timed(label="Sample Computation")
    def sample_compute():
        time.sleep(0.1)
        return 42

    result = sample_compute()

    # Demo evaluation metrics
    evaluator = EvaluationMetrics()
    scores = np.random.uniform(0, 100, 50)
    outcomes = (scores < 40).astype(int)
    metrics = evaluator.compute_scoring_metrics(scores, outcomes, label="demo_run")
    print(json.dumps(metrics, indent=2, default=str))

    # Demo pipeline logger
    pl = PipelineLogger("test_pipeline")
    pl.log_step("Step 1: Load data", "OK")
    pl.log_step("Step 2: Process", "OK", {"rows": 1000})
    pl.log_step("Step 3: Score", "WARN", {"note": "low confidence"})
    print(json.dumps(pl.summary(), indent=2))
