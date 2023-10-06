import numpy as np
from scipy.stats import mode
from itertools import combinations

### 2023-07-15 : currently no ties are considered and assume full assignment.
### Generalized Borda's method is explained in Qiu, 2018.


def borda(rank_collection):
    """Implements the Borda ranking aggregation. - Actually soft voting, used in

    Parameters
    ----------
    rank_collection : np.ndarray of shape (n_samples, n_labels, n_models)
        Predicted rankings of labels from n_models number of different models.
        i.e. the smaller the value, the more preferred.

    Returns
    -------
    rank_array : np.ndarray of shape (n_samples, n_labels)
    """
    score_array = np.zeros(
        (rank_collection.shape[0], rank_collection.shape[1], rank_collection.shape[1])
    )
    for (i, j) in combinations(range(rank_collection.shape[1]), 2):
        slice_ref = rank_collection[:, i, :]
        slice_compare = rank_collection[:, j, :]
        score_array[:, i, j] = np.sum(slice_ref < slice_compare, axis=1)
        score_array[:, j, i] = np.sum(slice_ref > slice_compare, axis=1)
    final_score = np.sum(score_array, axis=2)
    order = np.argsort(final_score, axis=1)
    rank = np.argsort(order, axis=1)
    return (
        np.ones_like(rank) * rank.shape[1] - rank
    )  # returns lowest value to highest score


def borda_count(rank_collection, weights=None):
    """Implements the actual Borda ranking aggregation.

    Parameters
    ----------
    rank_collection : np.ndarray of shape (n_samples, n_labels, n_models)
        Predicted rankings of labels from n_models number of different models.
        i.e. the smaller the value, the more preferred.
        When there are nan values, we assume that the rank vector's values are adjusted to [1, n_non_nan labels]
    weights : np.ndarray of shape (n_models,)
        Weights, as computed by 'weighted distance * portion of ranks' of each neighbor.

    Returns
    -------
    rank_array : np.ndarray of shape (n_samples, n_labels)
    """
    n_samples, n_labels, n_models = rank_collection.shape
    if np.any(
        np.isnan(rank_collection)
    ):  # For the generalized version with incomplete labels
        n_ranked_labels = np.tile(
            np.expand_dims(np.sum(np.isnan(rank_collection), axis=1), axis=1),
            (1, n_labels, 1),
        )
        score_array = np.divide(
            (n_labels + 1) * (n_ranked_labels - rank_collection + 1),
            n_ranked_labels + 1,
        )  # Scores for ranked labels
        score_array[np.where(np.isnan(score_array))] = 0.5 * (
            n_labels + 1
        )  # Scores for Incomplete labels
        if weights is None:
            score_array = np.sum(score_array, axis=2)
        else:
            score_by_sample = []
            for i in range(score_array.shape[0]):
                score_by_sample.append(
                    np.dot(score_array[i], weights[i]) / np.sum(weights[i])
                )
            score_array = np.vstack(tuple(score_by_sample))
    else:  # complete version
        score_array = np.sum(rank_collection.shape[1] - rank_collection, axis=2)

    rank = score_array.shape[1] - score_array.argsort(axis=1).argsort(axis=1)
    return rank


def copeland(rank_collection):
    """Implements the Copeland ranking aggregation."""
    score_array = np.zeros(
        (rank_collection.shape[0], rank_collection.shape[1], rank_collection.shape[1])
    )

    def f(num_wins, num_models=rank_collection.shape[2]):
        if num_wins > num_models * 0.5:
            return 1
        elif num_wins == num_models * 0.5:
            return 0.5
        else:
            return 0

    for (i, j) in combinations(range(rank_collection.shape[1]), 2):
        slice_ref = rank_collection[:, i, :]
        slice_compare = rank_collection[:, j, :]
        score_array[:, i, j] = np.array(
            list(map(f, np.sum(slice_ref < slice_compare, axis=1)))
        ).reshape(-1, 1)
        score_array[:, j, i] = np.array(
            list(map(f, np.sum(slice_ref > slice_compare, axis=1)))
        ).reshape(-1, 1)
    final_score = np.sum(score_array, axis=2)
    rank = np.argsort(-final_score, axis=1) + np.ones_like(final_score)
    return rank  # returns lowest value to highest score


def modal(rank_collection):
    """Implements the modal ranking aggregation.

    Parameters
    ----------
    rank_collection : np.ndarray of shape (n_samples, n_labels, n_models)
        Predicted rankings of labels from n_models number of different models.
        i.e. the smaller the value, the more preferred.

    Returns
    -------
    rank_array : np.ndarray of shape (n_samples, n_labels)
        Aggregated ranks.
    """
    modes, _ = mode(
        rank_collection, axis=2, keepdims=False
    )  # Note that only one value is returned even with ties
    return modes.reshape(rank_collection.shape[0], rank_collection.shape[1])
