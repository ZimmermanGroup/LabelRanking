import numpy as np
from scipy.stats import mode
from itertools import combinations

### 2023-07-15 : currently no ties are considered and assume full assignment.
### Generalized Borda's method is explained in Qiu, 2018.

def borda(rank_collection):
    """ Implements the Borda ranking aggregation. - Actually soft voting, used in 
    
    Parameters
    ----------
    rank_collection : np.ndarray of shape (n_samples, n_labels, n_models)
        Predicted rankings of labels from n_models number of different models.
        i.e. the smaller the value, the more preferred.

    Returns
    -------
    rank_array : np.ndarray of shape (n_samples, n_labels)
    """
    score_array = np.zeros((rank_collection.shape[0], rank_collection.shape[1], rank_collection.shape[1]))
    for (i, j) in combinations(range(rank_collection.shape[1]), 2) :
        slice_ref = rank_collection[:,i,:]
        slice_compare = rank_collection[:,j,:]
        score_array[:, i, j] = np.sum(slice_ref < slice_compare, axis=1)
        score_array[:, j, i] = np.sum(slice_ref > slice_compare, axis=1)
    final_score = np.sum(score_array, axis=2)
    order = np.argsort(final_score, axis=1)
    rank = np.argsort(order, axis=1)
    return np.ones_like(rank)*rank.shape[1] - rank # returns lowest value to highest score


def borda_count(rank_collection) :
    """ Implements the actual Borda ranking aggregation.
    
    Parameters
    ----------
    rank_collection : np.ndarray of shape (n_samples, n_labels, n_models)
        Predicted rankings of labels from n_models number of different models.
        i.e. the smaller the value, the more preferred.

    Returns
    -------
    rank_array : np.ndarray of shape (n_samples, n_labels)
    """
    n_samples, n_labels, n_models = rank_collection.shape
    score_array = np.zeros((n_samples, n_labels))
    if np.any(np.isnan(rank_collection)) : # For the generalized version
        for i in range(n_samples) :
            ranks_for_sample = rank_collection[i,:,:] # shape (n_labels, n_models)
            score_vector = np.zeros_like(ranks_for_sample)
            for j in range(n_models) : # for each model
                ranks_for_sample_by_model = ranks_for_sample[:,j]
                m_prime = max(ranks_for_sample_by_model)
                for k, r in enumerate(ranks_for_sample_by_model) :
                    if r!=np.nan :
                        score_vector[k,j] = (m_prime + 1 - r)*(n_labels+1)*(m_prime + 1)
                    else :
                        score_vector[k,j] = 0.5*(1+n_labels)
            score_array[i] = score_vector
                
    else : # complete version
        for i in range(rank_collection.shape[0]) :
            ranks_for_sample = rank_collection[i,:,:] # shape (n_labels, n_models)
            score_array[i] = np.sum(
                ranks_for_sample.shape[0] * np.ones_like(ranks_for_sample) - ranks_for_sample,
                axis=1
            )
    order = np.argsort(score_array, axis=1)
    rank = np.argsort(order, axis=1)
    return rank


def copeland(rank_collection):
    """ Implements the Copeland ranking aggregation."""
    score_array = np.zeros((rank_collection.shape[0], rank_collection.shape[1], rank_collection.shape[1]))
    def f(num_wins, num_models=rank_collection.shape[2]) :
        if num_wins > num_models * 0.5 : 
            return 1
        elif num_wins == num_models * 0.5:
            return 0.5
        else :
            return 0
        
    for (i, j) in combinations(range(rank_collection.shape[1]), 2) :
        slice_ref = rank_collection[:,i,:]
        slice_compare = rank_collection[:,j,:]
        score_array[:,i,j] = np.array(list(map(f, np.sum(slice_ref < slice_compare, axis=1)))).reshape(-1,1)
        score_array[:,j,i] = np.array(list(map(f, np.sum(slice_ref > slice_compare, axis=1)))).reshape(-1,1)
    final_score = np.sum(score_array, axis=2)
    order = np.argsort(final_score, axis=1)
    rank = np.argsort(order, axis=1) + np.ones_like(order)
    return len(rank) - rank # returns lowest value to highest score


def modal(rank_collection):
    """ Implements the modal ranking aggregation.
    
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
    modes, _ = mode(rank_collection, axis=2) # Note that only one value is returned even with ties
    return modes.reshape(rank_collection.shape[0], rank_collection.shape[1])


