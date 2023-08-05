import numpy as np


def yield_to_ranking(yield_array):
    """Transforms an array of yield values to their rankings.
    Currently, treat 0% yields as ties in the last place. (total # of labels)

    Parameters
    ----------
    yield_array : np.ndarray of shape (n_samples, n_conditions)
        Array of raw yield values.

    Returns
    -------
    ranking_array : np.ndarray of shape (n_samples, n_conditions)
        Array of ranking values. Lower values correspond to higher yields.
    """
    if len(yield_array.shape) == 2:
        raw_rank = yield_array.shape[1] - np.argsort(
            np.argsort(yield_array, axis=1), axis=1
        )
        for i, row in enumerate(yield_array):
            raw_rank[i, np.where(row == 0)[0]] = len(row > 0)
        # print("Raw rank", raw_rank)
    elif len(yield_array.shape) == 1:
        raw_rank = len(yield_array) - np.argsort(np.argsort(yield_array))
        raw_rank[np.where(raw_rank == 0)[0]] = len(raw_rank)
    return raw_rank
