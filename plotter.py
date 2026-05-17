import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; swap to "TkAgg" or remove for live display
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

from anomaly_detector import AnomalyEvent

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _save(fig, filename: str, show: bool) -> str:
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return path


class DataPlotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_time_series(self, cols: Optional[List[str]] = None,
                         show: bool = False) -> str:
        cols = cols or ["vibration_rms", "temperature_c", "rpm"]
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            healthy = self.df[self.df["label"] == 0]
            faulty  = self.df[self.df["label"] == 1]
            ax.plot(healthy["timestamp"], healthy[col], color="steelblue",
                    alpha=0.6, linewidth=0.8, label="Healthy")
            ax.plot(faulty["timestamp"],  faulty[col],  color="tomato",
                    alpha=0.6, linewidth=0.8, label="Faulty")
            ax.set_ylabel(col)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time")
        fig.suptitle("Bearing Sensor Time Series", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, "time_series.png", show)

    def plot_feature_distribution(self, show: bool = False) -> str:
        features = ["vibration_rms", "temperature_c", "rpm", "load"]
        fig, axes = plt.subplots(1, len(features), figsize=(14, 4))
        for ax, feat in zip(axes, features):
            self.df[self.df["label"] == 0][feat].plot.hist(
                ax=ax, bins=30, alpha=0.6, color="steelblue", label="Healthy")
            self.df[self.df["label"] == 1][feat].plot.hist(
                ax=ax, bins=30, alpha=0.6, color="tomato", label="Faulty")
            ax.set_title(feat)
            ax.legend(fontsize=8)
        fig.suptitle("Feature Distributions: Healthy vs Faulty", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, "feature_distributions.png", show)

    def plot_health_score(self, scores: pd.Series, show: bool = False) -> str:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.df["timestamp"], scores, color="mediumseagreen",
                linewidth=1.0, label="Health Score")
        ax.axhline(y=0.5, color="orange",  linestyle="--", linewidth=1, label="Warning (0.5)")
        ax.axhline(y=0.3, color="crimson", linestyle="--", linewidth=1, label="Critical (0.3)")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time")
        ax.set_ylabel("Health Score")
        ax.set_title("Bearing Health Score Over Time", fontsize=13, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return _save(fig, "health_score.png", show)

    def plot_anomalies(self, scores: pd.Series, events: List[AnomalyEvent],
                       show: bool = False) -> str:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.df["timestamp"], scores, color="mediumseagreen",
                linewidth=1.0, label="Health Score")
        warnings  = [e for e in events if e.severity == "warning"]
        criticals = [e for e in events if e.severity == "critical"]
        if warnings:
            ax.scatter([e.timestamp for e in warnings],
                       [e.health_score for e in warnings],
                       color="orange", s=20, zorder=5, label=f"Warning ({len(warnings)})")
        if criticals:
            ax.scatter([e.timestamp for e in criticals],
                       [e.health_score for e in criticals],
                       color="crimson", s=20, zorder=5, label=f"Critical ({len(criticals)})")
        ax.axhline(y=0.5, color="orange",  linestyle="--", linewidth=0.8)
        ax.axhline(y=0.3, color="crimson", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time")
        ax.set_ylabel("Health Score")
        ax.set_title("Anomaly Detection Results", fontsize=13, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return _save(fig, "anomalies.png", show)
