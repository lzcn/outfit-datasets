from collections import defaultdict
from typing import List
import json
import numpy as np
import torch
from ignite.engine.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class EpochMeter(Metric):
    """Computer the average over an epoch."""

    def __init__(self, output_transform=lambda x: x):
        self._cum_value = None
        self._cum_weight = None
        super(EpochMeter, self).__init__(output_transform=output_transform)

    def reset(self):
        self._cum_value = 0.0
        self._cum_weight = 0.0

    def update(self, output):
        value, weight = output
        self._cum_value += value * weight
        self._cum_weight += weight

    def compute(self):
        if self._cum_weight == 0:
            raise NotComputableError("EpochMeter must have at least one example before it can be computed.")
        return self._cum_value / self._cum_weight


class EpochScalersMeter(Metric):
    def __init__(self, output_transform=lambda x: x):

        self._cum_value = None
        self._cum_weight = None
        super(EpochScalersMeter, self).__init__(output_transform=output_transform)

    def reset(self):
        self._cum_value = defaultdict(float)
        self._cum_weight = 0.0

    def update(self, output):
        scalers, weight = output
        for k, v in scalers.items():
            if torch.is_tensor(v) and len(v.shape) == 0:
                v = v.item()
            self._cum_value[k] += v * weight
        self._cum_weight += weight

    def compute(self):
        if self._cum_weight == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        return {k: v / self._cum_weight for k, v in self._cum_value.items()}


class EpochBundleMetric(Metric):
    """Epoch metric for test engine.

    Args:
        bundlers (dict): a bundler of functions for computing. The keywords are the
            names of metrics and the functions are callable. Each funtion has the same
            signature, (posi: list, nega: list): -> float.
        output_transform (callable): covnert a batch output to (scores, labels, uidxs)
    Methods:
        attach(engine, name=""): attach metric to an engine. If name is given, then
            results are saved to engine.state.metrics[name]. Otherwise, update all
            results to engine.state.metrics.

    Examples:

        .. code-block:: python

            tester // test engine
            bundles = dict(
                accuracy=functional.pair_accuracy,
                loss=functional.pair_rank_loss,
                ndcg=functional.ndcg_score,
                auc=functional.auc_score,
            )
            metrics.EpochBundleMetric(bundles).attach(tester)
            // run tester
            assert tester.state.metric.keys() == bundlers.keys()
    """

    def __init__(self, bundles, output_transform=lambda x: x):
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)
        self._bundles = bundles
        super(EpochBundleMetric, self).__init__(output_transform=output_transform)

    def save(self, path):
        results = self.compute()
        data = {"scores": self._scores, "labels": self._labels}
        data.update(**results)
        with open(path, "w") as f:
            json.dump(data, f)

    def reset(self):
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)

    def update(self, output: List[list]):
        """Update meter.

        Args:
            scores (list): list of scores for outfits
            lables (list): list of labels for outfits
            uidx (list): list of user ids for outfits

        """
        scores, lables, uidx = output
        for x, y, u in zip(scores, lables, uidx):
            self._scores[u].append(x)
            self._labels[u].append(y)

    def compute(self):
        if len(self._scores) == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        metrics = defaultdict(list)
        for u in self._scores.keys():
            label = np.array(self._labels[u]).flatten()
            score = np.array(self._scores[u]).flatten()
            true_ind = np.where(label == 1)
            false_ind = np.where(label == 0)
            for key, func in self._bundles.items():
                metrics[key].append(func(score[true_ind], score[false_ind]))
        return {k: np.sum(v) / len(v) for k, v in metrics.items()}

    def completed(self, engine: Engine, name: str):
        result = self.compute()
        if name:
            engine.state.metrics[name] = result
        else:
            engine.state.metrics.update(result)

    def attach(self, engine, name=""):
        return super().attach(engine, name)
