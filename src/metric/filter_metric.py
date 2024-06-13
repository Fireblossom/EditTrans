import json
import torch
from torch import Tensor
from torchmetrics import Metric
from src.metric.my_seqeval import Seqeval
import logging
from datasets import ClassLabel
from src.utils.ner_utils import get_entity_bio
from torchmetrics.classification import MulticlassF1Score

logger = logging.getLogger('lightning')

class SpanFilterMertic(Metric):
    def __init__(self, filter_labels: ClassLabel,
                 compute_on_step=False,
                 dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.add_state("p", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("r", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("c", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0), dist_reduce_fx="sum")
        self.filter_labels = filter_labels

class TokenFilterMetric(Metric):
    def __init__(self, **kwargs: json.Any) -> None:
        super().__init__(**kwargs)