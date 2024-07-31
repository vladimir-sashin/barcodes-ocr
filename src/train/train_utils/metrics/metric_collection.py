from torchmetrics import MetricCollection

from src.train.train_utils.metrics.edit_distance import EditDistanceMetric
from src.train.train_utils.metrics.string_match import StringMatchMetric


def get_metrics() -> MetricCollection:
    return MetricCollection(
        {
            'string_match': StringMatchMetric(),
            'edit_distance': EditDistanceMetric(),
        },
    )
