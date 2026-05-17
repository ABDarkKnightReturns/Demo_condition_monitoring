import sys
import os

from data_simulator   import BearingSimulator
from ml_model         import BearingHealthModel
from anomaly_detector import AnomalyDetector
from plotter          import DataPlotter
from work_order       import WorkOrderManager


def run_pipeline(
    bearing_id:        str   = "BRG-001",
    rpm:               float = 1500.0,
    load:              float = 0.5,
    n_samples:         int   = 1000,
    fault_ratio:       float = 0.35,
    fault_type:        str   = "inner_race",
    model_type:        str   = "svm",
    anomaly_threshold: float = 0.5,
    window_size:       int   = 5,
    export_csv:        str   = "work_orders.csv",
    show_plots:        bool  = False,
) -> dict:
    print(f"\n{'='*55}")
    print(f"  Predictive Maintenance Pipeline — {bearing_id}")
    print(f"{'='*55}")

    # 1. Generate data
    print("\n[1/5] Generating simulated bearing data...")
    sim = BearingSimulator(bearing_id, rpm=rpm, load=load)
    df  = sim.generate_dataset(n=n_samples, fault_ratio=fault_ratio, fault_type=fault_type)
    print(f"      {len(df)} samples | {(df['label']==1).sum()} faulty "
          f"({fault_ratio*100:.0f}%)")

    # 2. Train / evaluate model
    print(f"\n[2/5] Training {model_type.upper()} classifier...")
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    model = BearingHealthModel(model_type=model_type)
    model.train(train_df)
    metrics = model.evaluate(test_df)
    print(f"      Accuracy: {metrics['accuracy']:.4f}  |  F1: {metrics['f1_score']:.4f}")
    print(f"      Confusion matrix: {metrics['confusion_matrix']}")

    # 3. Compute health scores on full dataset
    print("\n[3/5] Computing health scores & detecting anomalies...")
    detector = AnomalyDetector(threshold=anomaly_threshold, window_size=window_size)
    scores   = detector.compute_health_score(df, model)
    events   = detector.detect(scores, df["timestamp"], bearing_id)
    summary  = detector.summarize(events)
    print(f"      Anomalies: {summary['total']} total  |  "
          f"{summary['warnings']} warnings  |  {summary['critical']} critical")
    if summary["first_anomaly"]:
        print(f"      First anomaly: {summary['first_anomaly']}")

    # 4. Plot
    print("\n[4/5] Generating plots  ->  bearing_pdm/plots/")
    plotter = DataPlotter(df)
    plotter.plot_time_series(show=show_plots)
    plotter.plot_feature_distribution(show=show_plots)
    plotter.plot_health_score(scores, show=show_plots)
    plotter.plot_anomalies(scores, events, show=show_plots)
    print("      time_series.png  |  feature_distributions.png  |  "
          "health_score.png  |  anomalies.png")

    # 5. Work orders
    print("\n[5/5] Creating work orders...")
    manager = WorkOrderManager()
    manager.create_orders_batch(events)
    csv_path = os.path.join(os.path.dirname(__file__), export_csv)
    manager.export_csv(csv_path)
    print(f"      {len(manager.orders)} work orders exported -> {export_csv}")
    open_orders    = manager.list_orders("open")
    critical_orders = [o for o in open_orders if o.severity == "critical"]
    if critical_orders:
        print(f"\n  *** {len(critical_orders)} CRITICAL orders require immediate action ***")

    print(f"\n{'='*55}\n")
    return {
        "metrics":     metrics,
        "anomaly_summary": summary,
        "work_orders": manager.orders,
        "scores":      scores,
    }


if __name__ == "__main__":
    run_pipeline()
