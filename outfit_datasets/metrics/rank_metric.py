import numpy as np
import sklearn.metrics
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from collections import defaultdict


def to_canonical(posi, nega):
    """Return the canonical representation.

    Parameters
    ----------
    posi: positive scores
    nege: negative scores

    Return
    ------
    y_true: true label 0 for negative sample and 1 for positive
    y_score: predicted score for corresponding samples

    """
    posi, nega = np.array(posi), np.array(nega)
    y_true = np.array([1] * len(posi) + [0] * len(nega))
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    return (y_true, y_score)


def calc_auc(posi, nega):
    """Compute mean auc.

    Parameters
    ----------
    posi: Scores for positive outfits for each user.
    nega: Socres for negative outfits for each user.

    Returns
    -------
    auc: mean AUC score.
    """
    assert len(posi) == len(nega)
    user_auc = []
    for u in posi.keys():
        y_true, y_score = to_canonical(posi[u], nega[u])
        auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        user_auc.append(auc)
    return np.array(user_auc)


def calc_ndcg(posi, nega, wtype="max"):
    """Mean Normalize Discounted cumulative gain (NDCG).

    Parameters
    ----------
    posi: positive scores for each user.
    nega: negative scores for each user.
    wtype: type for discounts

    Returns
    -------
    mean_ndcg : array, shape = [num_users]
        mean ndcg for each user (averaged among all rank)
    avg_ndcg : array, shape = [max(n_samples)], averaged ndcg at each
        position (averaged among all users for given rank)

    """
    assert len(posi) == len(nega)
    u_labels, u_scores = [], []
    for u in posi.keys():
        label, score = to_canonical(posi[u], nega[u])
        u_labels.append(label)
        u_scores.append(score)
    mean_ndcg, ndcg = mean_ndcg_score(u_scores, u_labels, wtype)
    return mean_ndcg, ndcg


def mean_ndcg_score(u_scores, u_labels, wtype="max"):
    """Mean Normalize Discounted cumulative gain (NDCG) for all users.

    Parameters
    ----------
    u_score : array of arrays, shape = [num_users]
        Each array is the predicted scores, shape = [n_samples[u]]
    u_label : array of arrays, shape = [num_users]
        Each array is the ground truth label, shape = [n_samples[u]]
    wtype : 'log' or 'max'
        type for discounts
    Returns
    -------
    mean_ndcg : array, shape = [num_users]
        mean ndcg for each user (averaged among all rank)
    avg_ndcg : array, shape = [max(n_samples)], averaged ndcg at each
        position (averaged among all users for given rank)

    """
    num_users = len(u_scores)
    n_samples = [len(scores) for scores in u_scores]
    max_sample = max(n_samples)
    count = np.zeros(max_sample)
    mean_ndcg = np.zeros(num_users)
    avg_ndcg = np.zeros(max_sample)
    for u in range(num_users):
        ndcg = ndcg_score(u_scores[u], u_labels[u], wtype)
        avg_ndcg[: n_samples[u]] += ndcg
        count[: n_samples[u]] += 1
        mean_ndcg[u] = ndcg.mean()
    return mean_ndcg, avg_ndcg / count


def ndcg_score(y_score, y_label, wtype="max"):
    """Normalize Discounted cumulative gain (NDCG).

    Parameters
    ----------
    y_score : array, shape = [n_samples]
        Predicted scores.
    y_label : array, shape = [n_samples]
        Ground truth label (binary).
    wtype : 'log' or 'max'
        type for discounts
    Returns
    -------
    score : ndcg@m
    References
    ----------
      - [1] Hu Y, Yi X, Davis L S. Collaborative fashion recommendation:
           A functional tensor factorization approach[C]
           Proceedings of the 23rd ACM international conference on Multimedia.
           ACM, 2015: 129-138.
      - [2] Lee C P, Lin C J. Large-scale Linear RankSVM[J].
           Neural computation, 2014, 26(4): 781-817.

    """
    y_score = y_score.reshape(-1)
    y_label = y_label.reshape(-1)
    order = np.argsort(-y_score)
    p_label = np.take(y_label, order)
    i_label = np.sort(y_label)[::-1]
    p_gain = 2 ** p_label - 1
    i_gain = 2 ** i_label - 1
    if wtype.lower() == "max":
        discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))
    else:
        discounts = np.log2(np.arange(len(y_label)) + 2)
    dcg_score = (p_gain / discounts).cumsum()
    idcg_score = (i_gain / discounts).cumsum()
    return dcg_score / idcg_score


class NDCG(Metric):
    """Computer the average over an epoch."""

    def __init__(self, output_transform=lambda x: x):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)
        super(NDCG, self).__init__(output_transform=output_transform)

    def reset(self):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)

    def update(self, output):
        posi, nega, uidx = output
        for p, n, u in zip(posi, nega, uidx):
            self._posi[u].append(p)
            self._nega[u].append(n)

    def compute(self):
        if len(self._posi) == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        return calc_ndcg(self._posi, self._nega)


class AUC(Metric):
    """Computer the average over an epoch."""

    def __init__(self, output_transform=lambda x: x):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)
        super(AUC, self).__init__(output_transform=output_transform)

    def reset(self):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)

    def update(self, output):
        posi, nega, uidx = output
        for p, n, u in zip(posi, nega, uidx):
            self._posi[u].append(p)
            self._nega[u].append(n)

    def compute(self):
        if len(self._posi) == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        return calc_auc(self._posi, self._nega)


class PairRankMetric(Metric):
    """Computer the average over an epoch.

    Inputs
    ------
    - (posi, nega, uidx): pairwise score for users

    Returns:
    - ndcg@n: averaged ndcg@n over all positions
    - mean ndcg: averaged ndcg@n over all users
    - auc: auc for all user
    """

    def __init__(self, output_transform=lambda x: x):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)
        super(PairRankMetric, self).__init__(output_transform=output_transform)

    def reset(self):
        self._posi = defaultdict(list)
        self._nega = defaultdict(list)

    def update(self, output):
        posi, nega, uidx = output
        for p, n, u in zip(posi, nega, uidx):
            self._posi[u].append(p)
            self._nega[u].append(n)

    def compute(self):
        if len(self._posi) == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        mean_ndcg, ndcg = calc_ndcg(self._posi, self._nega)
        auc = calc_auc(self._posi, self._nega)
        return {"mean_ndcg": mean_ndcg, "ndcg": ndcg, "auc": auc}


class RankMetric(Metric):
    """Computer the average over an epoch.

    Inputs
    ------
    - score, label, uidx: score and label of outfits with user id
    Returns:
    - ndcg@n: averaged ndcg@n over all positions
    - mean ndcg: averaged ndcg@n over all users
    - auc: auc for all user
    """

    def __init__(self, output_transform=lambda x: x):
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)
        super(RankMetric, self).__init__(output_transform=output_transform)

    def reset(self):
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)

    def update(self, output):
        scores, lables, uidx = output
        for x, y, u in zip(scores, lables, uidx):
            self._scores[u].append(x)
            self._labels[u].append(y)

    def compute(self):
        if len(self._scores) == 0:
            raise NotComputableError("Metric must have at least one example before it can be computed.")
        scores, labels = [], []
        aucs = []
        for u in self._scores.keys():
            y_true = np.array(self._labels[u]).flatten()
            y_pred = np.array(self._scores[u]).flatten()
            aucs.append(sklearn.metrics.roc_auc_score(y_true, y_pred))
            scores.append(y_pred)
            labels.append(y_true)
        auc = np.array(aucs)
        mean_ndcg, ndcg = mean_ndcg_score(scores, labels)
        return {"mean_ndcg": mean_ndcg, "ndcg": ndcg, "auc": auc}
