from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np


@dataclass
class AnomalyEvent:
    timestamp: datetime
    bearing_id: str
    health_score: float
    severity: str  # "warning" | "critical"

    def __repr__(self):
        return (f"AnomalyEvent({self.bearing_id} @ {self.timestamp} | "
                f"score={self.health_score:.3f} | {self.severity})")


class AnomalyDetector:
    def __init__(self, threshold: float = 0.5, window_size: int = 5):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        self.threshold = threshold
        self.window_size = window_size

    def compute_health_score(self, df: pd.DataFrame, model) -> pd.Series:
        raw_scores = model.predict_proba(df)
        return (pd.Series(raw_scores, index=df.index)
                .rolling(window=self.window_size, min_periods=1, center=True)
                .mean())

    def detect(self, scores: pd.Series, timestamps: pd.Series,
               bearing_id: str) -> List[AnomalyEvent]:
        events = []
        for ts, score in zip(timestamps, scores):
            if score < self.threshold:
                severity = "critical" if score < self.threshold * 0.6 else "warning"
                events.append(AnomalyEvent(
                    timestamp=ts,
                    bearing_id=bearing_id,
                    health_score=float(score),
                    severity=severity,
                ))
        return events

    def summarize(self, events: List[AnomalyEvent]) -> dict:
        if not events:
            return {"total": 0, "warnings": 0, "critical": 0,
                    "min_health_score": None, "first_anomaly": None}
        return {
            "total":           len(events),
            "warnings":        sum(1 for e in events if e.severity == "warning"),
            "critical":        sum(1 for e in events if e.severity == "critical"),
            "min_health_score": round(min(e.health_score for e in events), 4),
            "first_anomaly":   min(e.timestamp for e in events),
        }
