from collections import defaultdict

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class PairAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_pairs = None
        super(PairAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = 0
        self._num_pairs = 0

    def update(self, output):
        self._num_correct += output.sum().item()
        self._num_pairs += output.numel()

    def compute(self):
        if self._num_pairs == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return self._num_correct / self._num_pairs


class PairAccuracyScalers(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_pairs = None
        super(PairAccuracyScalers, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = defaultdict(float)
        self._num_pairs = defaultdict(int)

    def update(self, output):
        for k, value in output.items():
            self._num_correct[k] += value.sum().item()
            self._num_pairs[k] += value.numel()

    def compute(self):
        return {k: v / self._num_pairs[k] for k, v in self._num_correct.items()}
