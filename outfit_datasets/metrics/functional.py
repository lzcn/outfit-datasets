import numpy as np
from sklearn.metrics import roc_auc_score


def to_canonical(posi, nega):
    r"""Return the canonical representation.

    Args:
        posi (list): positive scores
        nege (list): negative scores

    Return:
        list, list:

            y_true: true label 0 for negative sample and 1 for positive

            y_score: predicted score for corresponding samples

    """
    posi, nega = np.array(posi), np.array(nega)
    y_true = np.array([1] * len(posi) + [0] * len(nega))
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    return (y_true, y_score)


def pair_accuracy(posi: list, nega: list) -> float:
    r"""Compute pairwise accuracy.

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`

    .. math::
        accuracy=\frac{\sum_{i,j} x_i - y_j}{d^2}

    Args:
        posi (list): score for positive outfit
        nega (list): score for negative outfit
    Return:
        float: pair-wise accuracy
    """
    posi = np.array(posi)
    nega = np.array(nega)
    diff = posi.reshape(-1, 1) - nega.reshape(1, -1)
    return (diff > 0).sum() / diff.size


def margin_loss(posi, nega):
    r"""Compute margin loss

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`

    .. math::
        l=\frac{1}{d}\sum_i\left(\max(0.9-x_i,0) + \max(y_i-0.1,0)\right)
    Args:
        posi (list): score for positive outfit
        nega (list): score for negative outfit

    Returns:
        float: margin loss
    """
    posi = np.maximum(0.9 - np.array(posi), 0) ** 2
    nega = np.maximum(np.array(nega) - 0.1, 0) ** 2
    return (posi.sum() + nega.sum()) / (posi.size + nega.size)


def pair_rank_loss(posi, nega):
    r"""Compute pairwise rank loss.

    Give positive and negative scores :math:`x,y\in\mathbb{R}^d`

    .. math::
        l=\frac{1}{d^2}\sum_{i,j} \log\left(1.0+\exp(y_j-x_i))\right)

    Args:
        posi (list) : score for positive outfit
        nega (list) : score for negative outfit
    Return:
        float: pair-wise loss
    """
    posi = np.array(posi)
    nega = np.array(nega)
    diff = posi.reshape(-1, 1) - nega.reshape(1, -1)
    return np.log(1.0 + np.exp(-diff)).sum() / diff.size


def auc_score(posi, nega):
    r"""Compute auc.

    Args:
        posi (list) : score for positive outfit
        nega (list) : score for negative outfit
    Return:
        auc score
    """
    y_true, y_score = to_canonical(posi, nega)
    return roc_auc_score(y_true, y_score)


def ndcg_score(posi, nega):
    r"""Compute ndcg.

    Args:
        posi (list) : score for positive outfit
        nega (list) : score for negative outfit
    Return
        ndcg (float): ndcg score
    """
    y_label, y_score = to_canonical(posi, nega)
    return _ndcg_score(y_score, y_label).mean()


def _ndcg_score(y_score, y_label, wtype="max"):
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
