import numpy as np
import pandas as pd
from datetime import datetime, timedelta

FAULT_MULTIPLIERS = {
    "inner_race":      (2.5, 1.30),
    "outer_race":      (2.0, 1.20),
    "rolling_element": (1.8, 1.15),
}

REQUIRED_COLUMNS = ["timestamp", "vibration_rms", "temperature_c", "rpm", "load", "label"]


class BearingSimulator:
    def __init__(self, bearing_id: str, rpm: float = 1500.0, load: float = 0.5):
        self.bearing_id = bearing_id
        self.rpm = rpm
        self.load = load

    def _timestamps(self, n: int, offset: int = 0) -> list:
        base = datetime(2024, 1, 1)
        return [base + timedelta(seconds=i + offset) for i in range(n)]

    def generate_normal(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed=None)
        vibration  = rng.normal(0.20, 0.02, n) * (1 + self.load * 0.10)
        temperature = rng.normal(50.0, 2.0, n) * (1 + self.load * 0.10)
        rpm = rng.normal(self.rpm, 10.0, n)
        return pd.DataFrame({
            "timestamp":     self._timestamps(n),
            "vibration_rms": np.clip(vibration, 0.05, None),
            "temperature_c": np.clip(temperature, 30.0, None),
            "rpm":           np.clip(rpm, 0.0, None),
            "load":          self.load,
            "label":         0,
        })

    def generate_faulty(self, n: int, fault_type: str = "inner_race") -> pd.DataFrame:
        vib_mult, temp_mult = FAULT_MULTIPLIERS.get(fault_type, (2.0, 1.20))
        rng = np.random.default_rng(seed=None)
        vibration   = rng.normal(0.20 * vib_mult, 0.08, n) * (1 + self.load * 0.20)
        temperature = rng.normal(50.0 * temp_mult, 4.0, n) * (1 + self.load * 0.15)
        rpm = rng.normal(self.rpm * 0.97, 20.0, n)
        return pd.DataFrame({
            "timestamp":     self._timestamps(n),
            "vibration_rms": np.clip(vibration, 0.05, None),
            "temperature_c": np.clip(temperature, 30.0, None),
            "rpm":           np.clip(rpm, 0.0, None),
            "load":          self.load,
            "label":         1,
        })

    def generate_dataset(self, n: int = 1000, fault_ratio: float = 0.3,
                         fault_type: str = "inner_race") -> pd.DataFrame:
        if not 0.0 < fault_ratio < 1.0:
            raise ValueError("fault_ratio must be between 0 and 1 (exclusive)")
        n_faulty = int(n * fault_ratio)
        n_normal = n - n_faulty
        df = pd.concat(
            [self.generate_normal(n_normal), self.generate_faulty(n_faulty, fault_type)],
            ignore_index=True,
        )
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        base = datetime(2024, 1, 1)
        df["timestamp"] = [base + timedelta(seconds=i) for i in range(len(df))]
        return df
