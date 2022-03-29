import json
from collections import defaultdict
from typing import List

import numpy as np


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


class UserBundleMetric(object):
    """Epoch metric for test engine.

    Args:
        bundlers (dict): a bundler of functions for computing. The keywords are the
            names of metrics and the functions are callable. Each function has the same
            signature, (posi: list, nega: list): -> float.
        output_transform (callable): convert a batch output to (scores, labels, uidxs)
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
        self._output_transform = output_transform

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
            labels (list): list of labels for outfits
            uidx (list): list of user ids for outfits

        """
        scores, labels, uidxs = self._output_transform(output)
        for x, y, u in zip(scores, labels, uidxs):
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
