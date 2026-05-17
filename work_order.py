import csv
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from anomaly_detector import AnomalyEvent

ACTION_MAP = {
    "warning":  "Schedule inspection within 7 days. Monitor vibration and temperature trends.",
    "critical": "Immediate bearing replacement required. Take equipment offline.",
}


@dataclass
class WorkOrder:
    order_id: str
    bearing_id: str
    created_at: datetime
    severity: str
    description: str
    status: str
    recommended_action: str

    def __repr__(self):
        return (f"WorkOrder({self.order_id[:8]}… | {self.bearing_id} | "
                f"{self.severity} | {self.status})")


class WorkOrderManager:
    def __init__(self):
        self.orders: List[WorkOrder] = []

    def create_order(self, event: AnomalyEvent) -> WorkOrder:
        order = WorkOrder(
            order_id=str(uuid.uuid4()),
            bearing_id=event.bearing_id,
            created_at=event.timestamp,
            severity=event.severity,
            description=(f"Anomaly detected on bearing '{event.bearing_id}'. "
                         f"Health score dropped to {event.health_score:.3f}."),
            status="open",
            recommended_action=ACTION_MAP.get(event.severity, "Review bearing condition."),
        )
        self.orders.append(order)
        return order

    def create_orders_batch(self, events: List[AnomalyEvent]) -> List[WorkOrder]:
        return [self.create_order(e) for e in events]

    def update_status(self, order_id: str, status: str) -> None:
        valid = {"open", "in_progress", "closed"}
        if status not in valid:
            raise ValueError(f"status must be one of {valid}")
        for order in self.orders:
            if order.order_id == order_id:
                order.status = status
                return
        raise KeyError(f"No order found with id '{order_id}'")

    def list_orders(self, status: Optional[str] = None) -> List[WorkOrder]:
        if status is None:
            return list(self.orders)
        return [o for o in self.orders if o.status == status]

    def export_csv(self, path: str) -> None:
        if not self.orders:
            return
        fields = ["order_id", "bearing_id", "created_at", "severity",
                  "description", "status", "recommended_action"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for order in self.orders:
                writer.writerow({k: getattr(order, k) for k in fields})
