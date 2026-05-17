"""
Predictive Maintenance — Unit Test Suite
Run: python tests.py  (from the bearing_pdm/ directory)
"""
import os
import sys
import unittest
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np

from data_simulator   import BearingSimulator, REQUIRED_COLUMNS
from ml_model         import BearingHealthModel
from anomaly_detector import AnomalyDetector, AnomalyEvent
from work_order       import WorkOrder, WorkOrderManager
from plotter          import DataPlotter

# ── helpers ──────────────────────────────────────────────────────────────────

def make_sim(n=400, fault_ratio=0.4):
    sim = BearingSimulator("TEST-001", rpm=1500, load=0.5)
    return sim, sim.generate_dataset(n=n, fault_ratio=fault_ratio)


def trained_model(df):
    m = BearingHealthModel("svm")
    m.train(df)
    return m


# ── BearingSimulator ──────────────────────────────────────────────────────────

class TestBearingSimulator(unittest.TestCase):

    def setUp(self):
        self.sim = BearingSimulator("B001", rpm=1500, load=0.5)

    def test_generate_normal_row_count(self):
        df = self.sim.generate_normal(200)
        self.assertEqual(len(df), 200)

    def test_generate_normal_columns(self):
        df = self.sim.generate_normal(50)
        self.assertEqual(set(df.columns), set(REQUIRED_COLUMNS))

    def test_generate_normal_all_healthy(self):
        df = self.sim.generate_normal(200)
        self.assertTrue((df["label"] == 0).all())

    def test_generate_faulty_all_faulty(self):
        df = self.sim.generate_faulty(200)
        self.assertTrue((df["label"] == 1).all())

    def test_faulty_higher_vibration_than_normal(self):
        normal = self.sim.generate_normal(500)
        faulty = self.sim.generate_faulty(500)
        self.assertGreater(faulty["vibration_rms"].mean(), normal["vibration_rms"].mean())

    def test_faulty_higher_temperature_than_normal(self):
        normal = self.sim.generate_normal(500)
        faulty = self.sim.generate_faulty(500)
        self.assertGreater(faulty["temperature_c"].mean(), normal["temperature_c"].mean())

    def test_dataset_fault_ratio(self):
        df = self.sim.generate_dataset(1000, fault_ratio=0.3)
        actual = (df["label"] == 1).mean()
        self.assertAlmostEqual(actual, 0.3, delta=0.02)

    def test_dataset_total_length(self):
        df = self.sim.generate_dataset(500, fault_ratio=0.4)
        self.assertEqual(len(df), 500)

    def test_dataset_no_missing_values(self):
        df = self.sim.generate_dataset(200)
        self.assertFalse(df.isnull().any().any())

    def test_dataset_timestamps_monotonic(self):
        df = self.sim.generate_dataset(200)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)

    def test_invalid_fault_ratio_raises(self):
        with self.assertRaises(ValueError):
            self.sim.generate_dataset(100, fault_ratio=1.5)

    def test_all_fault_types_generate_data(self):
        for ft in ["inner_race", "outer_race", "rolling_element"]:
            df = self.sim.generate_faulty(50, fault_type=ft)
            self.assertEqual(len(df), 50, f"failed for fault_type={ft}")


# ── BearingHealthModel ────────────────────────────────────────────────────────

class TestBearingHealthModel(unittest.TestCase):

    def setUp(self):
        _, self.df = make_sim(600, 0.4)
        split = int(len(self.df) * 0.8)
        self.train_df = self.df.iloc[:split]
        self.test_df  = self.df.iloc[split:]
        self.model = trained_model(self.train_df)

    def test_predict_returns_correct_length(self):
        preds = self.model.predict(self.test_df)
        self.assertEqual(len(preds), len(self.test_df))

    def test_predict_only_binary_labels(self):
        preds = self.model.predict(self.test_df)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_predict_proba_range(self):
        scores = self.model.predict_proba(self.test_df)
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0).all())

    def test_evaluate_keys(self):
        result = self.model.evaluate(self.test_df)
        for key in ["accuracy", "f1_score", "confusion_matrix"]:
            self.assertIn(key, result)

    def test_accuracy_above_baseline(self):
        result = self.model.evaluate(self.test_df)
        # Majority class is ~60 % healthy, so a decent model must beat that
        self.assertGreater(result["accuracy"], 0.70)

    def test_untrained_model_raises(self):
        m = BearingHealthModel("svm")
        with self.assertRaises(RuntimeError):
            m.predict(self.test_df)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self.model.save(path)
            loaded = BearingHealthModel("svm").load(path)
            orig_preds   = self.model.predict(self.test_df)
            loaded_preds = loaded.predict(self.test_df)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            BearingHealthModel("random_forest")

    def test_xgboost_model_trains(self):
        try:
            m = BearingHealthModel("xgboost")
            m.train(self.train_df)
            preds = m.predict(self.test_df)
            self.assertEqual(len(preds), len(self.test_df))
        except ImportError:
            self.skipTest("xgboost not installed")


# ── AnomalyDetector ───────────────────────────────────────────────────────────

class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.sim = BearingSimulator("B001", rpm=1500, load=0.5)
        self.detector = AnomalyDetector(threshold=0.5, window_size=3)

    def test_healthy_data_few_anomalies(self):
        df_h = self.sim.generate_normal(300)
        df_h["label"] = 0
        _, df_full = make_sim(600, 0.4)
        model = trained_model(df_full)
        scores = self.detector.compute_health_score(df_h, model)
        events = self.detector.detect(scores, df_h["timestamp"], "B001")
        # healthy data should produce very few anomalies
        self.assertLess(len(events), len(df_h) * 0.15)

    def test_faulty_data_many_anomalies(self):
        df_f = self.sim.generate_faulty(300)
        _, df_full = make_sim(600, 0.4)
        model = trained_model(df_full)
        scores = self.detector.compute_health_score(df_f, model)
        events = self.detector.detect(scores, df_f["timestamp"], "B001")
        self.assertGreater(len(events), len(df_f) * 0.3)

    def test_anomaly_event_severity_classification(self):
        # Manually build scores below threshold
        scores = pd.Series([0.45, 0.20, 0.50, 0.10])
        timestamps = pd.Series([datetime(2024, 1, 1, 0, 0, i) for i in range(4)])
        events = self.detector.detect(scores, timestamps, "B001")
        severities = {e.severity for e in events}
        self.assertTrue(severities.issubset({"warning", "critical"}))

    def test_no_anomaly_above_threshold(self):
        scores = pd.Series([0.8, 0.9, 0.75, 0.85])
        timestamps = pd.Series([datetime(2024, 1, 1, 0, 0, i) for i in range(4)])
        events = self.detector.detect(scores, timestamps, "B001")
        self.assertEqual(len(events), 0)

    def test_summarize_empty(self):
        s = self.detector.summarize([])
        self.assertEqual(s["total"], 0)
        self.assertIsNone(s["min_health_score"])

    def test_summarize_counts(self):
        events = [
            AnomalyEvent(datetime.now(), "B1", 0.45, "warning"),
            AnomalyEvent(datetime.now(), "B1", 0.15, "critical"),
            AnomalyEvent(datetime.now(), "B1", 0.10, "critical"),
        ]
        s = self.detector.summarize(events)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["warnings"], 1)
        self.assertEqual(s["critical"], 2)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            AnomalyDetector(threshold=1.5)


# ── WorkOrderManager ──────────────────────────────────────────────────────────

class TestWorkOrderManager(unittest.TestCase):

    def _make_event(self, severity="warning", score=0.4):
        return AnomalyEvent(datetime(2024, 3, 15, 10, 0, 0), "BRG-42", score, severity)

    def test_create_order_fields(self):
        mgr = WorkOrderManager()
        order = mgr.create_order(self._make_event())
        self.assertEqual(order.bearing_id, "BRG-42")
        self.assertEqual(order.severity, "warning")
        self.assertEqual(order.status, "open")
        self.assertIsInstance(order.order_id, str)
        self.assertTrue(len(order.order_id) > 0)

    def test_create_orders_batch_count(self):
        mgr = WorkOrderManager()
        events = [self._make_event("warning"), self._make_event("critical", 0.1)]
        orders = mgr.create_orders_batch(events)
        self.assertEqual(len(orders), 2)
        self.assertEqual(len(mgr.orders), 2)

    def test_update_status(self):
        mgr = WorkOrderManager()
        order = mgr.create_order(self._make_event())
        mgr.update_status(order.order_id, "in_progress")
        self.assertEqual(mgr.orders[0].status, "in_progress")

    def test_update_status_invalid_raises(self):
        mgr = WorkOrderManager()
        order = mgr.create_order(self._make_event())
        with self.assertRaises(ValueError):
            mgr.update_status(order.order_id, "done")

    def test_update_status_unknown_id_raises(self):
        mgr = WorkOrderManager()
        with self.assertRaises(KeyError):
            mgr.update_status("nonexistent-id", "closed")

    def test_list_orders_filter(self):
        mgr = WorkOrderManager()
        o1 = mgr.create_order(self._make_event("warning"))
        o2 = mgr.create_order(self._make_event("critical", 0.1))
        mgr.update_status(o1.order_id, "closed")
        self.assertEqual(len(mgr.list_orders("open")), 1)
        self.assertEqual(len(mgr.list_orders("closed")), 1)
        self.assertEqual(len(mgr.list_orders()), 2)

    def test_export_csv_creates_file(self):
        mgr = WorkOrderManager()
        mgr.create_orders_batch([self._make_event(), self._make_event("critical", 0.1)])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            mgr.export_csv(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                rows = f.readlines()
            self.assertEqual(len(rows), 3)  # header + 2 data rows
        finally:
            os.unlink(path)


# ── DataPlotter ───────────────────────────────────────────────────────────────

class TestDataPlotter(unittest.TestCase):

    def setUp(self):
        _, df = make_sim(300, 0.4)
        self.df = df
        self.plotter = DataPlotter(df)
        model = trained_model(df)
        detector = AnomalyDetector(threshold=0.5, window_size=3)
        self.scores = detector.compute_health_score(df, model)
        self.events = detector.detect(self.scores, df["timestamp"], "B001")

    def test_plot_time_series_creates_file(self):
        path = self.plotter.plot_time_series(show=False)
        self.assertTrue(os.path.isfile(path))

    def test_plot_feature_distribution_creates_file(self):
        path = self.plotter.plot_feature_distribution(show=False)
        self.assertTrue(os.path.isfile(path))

    def test_plot_health_score_creates_file(self):
        path = self.plotter.plot_health_score(self.scores, show=False)
        self.assertTrue(os.path.isfile(path))

    def test_plot_anomalies_creates_file(self):
        path = self.plotter.plot_anomalies(self.scores, self.events, show=False)
        self.assertTrue(os.path.isfile(path))


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegrationPipeline(unittest.TestCase):

    def test_full_pipeline_runs(self):
        from main import run_pipeline
        result = run_pipeline(
            bearing_id="INT-001", n_samples=400, fault_ratio=0.35,
            model_type="svm", show_plots=False,
        )
        self.assertIn("metrics", result)
        self.assertIn("anomaly_summary", result)
        self.assertIn("work_orders", result)
        self.assertGreater(result["metrics"]["accuracy"], 0.65)


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestBearingSimulator,
        TestBearingHealthModel,
        TestAnomalyDetector,
        TestWorkOrderManager,
        TestDataPlotter,
        TestIntegrationPipeline,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
