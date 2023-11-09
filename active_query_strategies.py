import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from copy import deepcopy
from scipy.stats.mstats import rankdata

def get_y_acquired(y_ranking, rem_inds, next_subs_inds, next_cond_inds):
    """Prepares the partial ranking array.

    Parameters
    ----------
    y_ranking : np.ndarray of shape (n_substrates, n_conds)
        Full ranking array of all data.
    rem_inds : list of ints
        Indices that are still available to sample for training data.
    next_subs_inds : list of ints
        Indices of substrates that are sampled.
    next_cond_inds : np.ndarray of shape (len(next_subs_inds), n_conds_to_sample)
        Indices of conditions selected for each substrate.

    Returns
    -------
    y_ranking_acquired : np.ndarray of shape (len(next_subs_inds), n_conds)
        Acquired partial ranking array.
    """
    y_ranking_acquired = deepcopy(
        y_ranking[[rem_inds[x] for x in next_subs_inds]]
    ).astype(float)
    for i, row in enumerate(next_cond_inds):
        inds_to_cover = [x for x in range(y_ranking.shape[1]) if x not in row]
        y_ranking_acquired[i, inds_to_cover] = np.nan
        y_ranking_acquired[i] = rankdata(y_ranking_acquired[i])
        y_ranking_acquired[i, inds_to_cover] = np.nan
    return y_ranking_acquired


def iteration_of_lowest_diff(
    X, y_ranking, rem_inds, predicted_proba_array, n_subs_to_sample, n_conds_to_sample
):
    """Samples reactions that have lowest difference in predicted probability between the top-k entries.

    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Ranking array of all substrates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_proba_array : np.ndarray of shape (n_substrates, n_conditions, n_conditions)
        Element at row i, column j corresponds to the RPC's predicted score of condition i being
        favorable over condition j.
    n_subs_to_sample : int
        Number of substrates to sample.
    n_conds_to_sample : int
        Number of reaction conditions to sample for one substrate.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_subs_to_sample, n_features)
        Input array of the substrates that were subject to data collection.
    y_ranking_acquired : np.ndarray of shape (n_subs_to_sample, n_conds)
        Ranking array of what the experimentalist would see.
        Unsampled reaction conditions have np.nan
    next_subs_inds : np.ndarray of shape (n_subs_to_sample,)
        Indices from REM_INDS that have been sampled in this iteration.
    """
    top_k_proba = np.partition(np.sum(predicted_proba_array, axis=1), 2, axis=1)[
        :, n_conds_to_sample:
    ]
    top_k_conds = np.argpartition(np.sum(predicted_proba_array, axis=1), 2, axis=1)[
        :, n_conds_to_sample:
    ]
    next_subs_inds = np.argpartition(
        np.abs(np.array([max(row) - min(row) for row in top_k_proba])), n_subs_to_sample
    )[:n_subs_to_sample]
    next_cond_inds = top_k_conds[next_subs_inds, :]

    X_acquired = X[[rem_inds[x] for x in next_subs_inds]]
    y_ranking_acquired = get_y_acquired(
        y_ranking, rem_inds, next_subs_inds, next_cond_inds
    )
    return X_acquired, y_ranking_acquired, next_subs_inds


def iteration_of_cond_first(
    X,
    y_ranking,
    train_inds,
    rem_inds,
    predicted_proba_array,
    n_subs_to_sample,
    n_conds_to_sample,
    smiles_list,
    substrate_selection,
):
    """First gets a pair of reactions where the average of predicted probability's deviation from 0.5
     is smallest, across all candidates.
    Then select two substrates with the lowest Tanimoto similarity to those sampled.

    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Ranking array of all substrates.
    train_inds : list of ints
        Indices selected for the training set candidates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_proba_array : np.ndarray of shape (n_substrates, n_conditions, n_conditions)
        Element at row i, column j corresponds to the RPC's predicted score of condition i being
        favorable over condition j.
    n_subs_to_sample : int
        Number of substrates to sample.
    n_conds_to_sample : int
        Number of reaction conditions to sample for one substrate.
    smiles_list : list of str
        List of smiles strings in the TRAINING dataset.
    substrate_selection : str {'farthest', 'quantile', 'skip_one'}
        How to select substrates.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_subs_to_sample, n_features)
        Input array of the substrates that were subject to data collection.
    y_ranking_acquired : np.ndarray of shape (n_subs_to_sample, n_conds)
        Ranking array of what the experimentalist would see.
        Unsampled reaction conditions have np.nan
    next_subs_inds : np.ndarray of shape (n_subs_to_sample,)
        Indices from REM_INDS that have been sampled in this iteration.
    """
    sampled_smiles = []
    rem_smiles = []
    for i, smiles in enumerate(smiles_list):
        if train_inds[i] in rem_inds:
            rem_smiles.append(smiles)
        else:
            sampled_smiles.append(smiles)
    avg_deviation = np.mean(np.abs(predicted_proba_array - 0.5), axis=0)
    inds = np.unravel_index(
        np.argsort(avg_deviation, axis=None),
        (predicted_proba_array.shape[1], predicted_proba_array.shape[2]),
    )
    conds = [inds[0][0], inds[1][0]]
    while len(conds) < n_conds_to_sample:
        rem_conds = [x for x in range(avg_deviation.shape[1]) if x not in conds]
        conds.append(np.argmin(np.mean(avg_deviation[rem_conds, :], axis=1)))
    next_cond_inds = np.repeat(np.array(conds).reshape(1, -1), n_subs_to_sample, axis=0)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=3)
    if substrate_selection == "farthest":
        next_subs_inds = []
        while len(next_subs_inds) < n_subs_to_sample:
            sampled_fp = [
                mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in sampled_smiles
            ]
            rem_fp = [mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in rem_smiles]
            dists = np.zeros((len(rem_fp), len(sampled_fp)))
            for i, fp in enumerate(rem_fp):
                dists[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, sampled_fp))
            sorted_inds = np.argsort(np.mean(dists, axis=1))[::-1]
            if len(next_subs_inds) > 0:
                for ind in sorted_inds:
                    if ind not in next_subs_inds:
                        farthest_ind = ind
                        break
            else:
                farthest_ind = sorted_inds[0]
            next_subs_inds.append(farthest_ind)
            sampled_fp.append(rem_fp[farthest_ind])
    else :
        # Rather than getting the farthest, get ones that divide the range of distances.
        sampled_fp = [
            mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in sampled_smiles
        ]
        rem_fp = [mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in rem_smiles]
        dists = np.zeros((len(rem_fp), len(sampled_fp)))
        for i, fp in enumerate(rem_fp):
            dists[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, sampled_fp))
        sorted_inds = np.argsort(np.mean(dists, axis=1))[::-1]
        if substrate_selection == "quantile" : n_to_add = 1
        elif substrate_selection == "skip_one" : n_to_add = 2
        next_subs_inds = [sorted_inds[int(len(sorted_inds)*(x/(n_subs_to_sample+n_to_add)))] for x in range(n_to_add, n_subs_to_sample+n_to_add)]
    X_acquired = X[[rem_inds[x] for x in next_subs_inds]]
    y_ranking_acquired = get_y_acquired(
        y_ranking, rem_inds, next_subs_inds, next_cond_inds
    )
    return X_acquired, y_ranking_acquired, next_subs_inds

def iteration_of_two_cond_pairs(X,
    y_ranking,
    train_inds,
    rem_inds,
    predicted_proba_array,
    n_subs_to_sample,
    n_conds_to_sample,
    smiles_list,
    substrate_selection,
):
    """First gets a pair of reactions where the average of predicted probability's deviation from 0.5
     is smallest, across all candidates. Select a substrate with the lowest average Tanimoto similarity to those sampled.
    For the other pair of conditions, select substrate that has highest uncertainty.

    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Ranking array of all substrates.
    train_inds : list of ints
        Indices selected for the training set candidates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_proba_array : np.ndarray of shape (n_substrates, n_conditions, n_conditions)
        Element at row i, column j corresponds to the RPC's predicted score of condition i being
        favorable over condition j.
    n_subs_to_sample : int
        Number of substrates to sample.
    n_conds_to_sample : int
        Number of reaction conditions to sample for one substrate.
    smiles_list : list of str
        List of smiles strings in the TRAINING dataset.
    substrate_selection : str {'farthest', 'quantile', 'skip_one'}
        How to select substrates.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_subs_to_sample, n_features)
        Input array of the substrates that were subject to data collection.
    y_ranking_acquired : np.ndarray of shape (n_subs_to_sample, n_conds)
        Ranking array of what the experimentalist would see.
        Unsampled reaction conditions have np.nan
    next_subs_inds : np.ndarray of shape (n_subs_to_sample,)
        Indices from REM_INDS that have been sampled in this iteration.
    """
    sampled_smiles = []
    rem_smiles = []
    for i, smiles in enumerate(smiles_list):
        if train_inds[i] in rem_inds:
            rem_smiles.append(smiles)
        else:
            sampled_smiles.append(smiles)
    avg_deviation = np.mean(np.abs(predicted_proba_array - 0.5), axis=0)
    inds = np.unravel_index(
        np.argsort(avg_deviation, axis=None),
        (predicted_proba_array.shape[1], predicted_proba_array.shape[2]),
    )
    first_cond_pair = [inds[0][0], inds[1][0]]
    # Getting substrate with highest average distance to sampled compounds
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=3)
    sampled_fp = [
        mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in sampled_smiles
    ]
    rem_fp = [mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in rem_smiles]
    dists = np.zeros((len(rem_fp), len(sampled_fp)))
    for i, fp in enumerate(rem_fp):
        dists[i] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, sampled_fp))
    if substrate_selection == "farthest":
        first_subs_ind = np.argmax(np.mean(dists, axis=1))
    else :
        if substrate_selection == "quantile" : portion = 1/2
        elif substrate_selection == "skip_one" : portion = 2/3
        first_subs_ind = np.argsort(np.mean(dists, axis=1))[int(len(dists)*portion)]
    
    other_cond_pair = [x for x in range(y_ranking.shape[1]) if x not in first_cond_pair]
    # Getting substrate with highest uncertainty for the pair above
    other_subs = np.argsort(
        np.abs(predicted_proba_array[:, other_cond_pair[0], other_cond_pair[1]] - 0.5)
    ) # [::-1]
    if other_subs[0] != first_subs_ind :
        other_subs_ind = other_subs[0]
    else :
        other_subs_ind = other_subs[1]
    next_subs_inds = [first_subs_ind] + [other_subs_ind]
    next_cond_inds = np.array([[x for x in first_cond_pair], [x for x in other_cond_pair]])

    X_acquired = X[[rem_inds[x] for x in next_subs_inds]]
    y_ranking_acquired = get_y_acquired(
        y_ranking, rem_inds, next_subs_inds, next_cond_inds
    )
    return X_acquired, y_ranking_acquired, next_subs_inds

def iteration_of_random(X, y_ranking, rem_inds, n_subs_to_sample, n_conds_to_sample):
    """Selects substrates and reaction conditions randomly, for baseline purposes.

    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Ranking array of all substrates.
    train_inds : list of ints
        Indices selected for the training set candidates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    n_subs_to_sample : int
        Number of substrates to sample.
    n_conds_to_sample : int
        Number of reaction conditions to sample for one substrate.
    smiles_list : list of str
        List of smiles strings in the TRAINING dataset.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_subs_to_sample, n_features)
        Input array of the substrates that were subject to data collection.
    y_ranking_acquired : np.ndarray of shape (n_subs_to_sample, n_conds)
        Ranking array of what the experimentalist would see.
        Unsampled reaction conditions have np.nan
    next_subs_inds : np.ndarray of shape (n_subs_to_sample,)
        Indices from REM_INDS that have been sampled in this iteration.
    """
    next_subs_inds = np.random.choice(
        np.arange(len(rem_inds)), n_subs_to_sample, replace=False
    )
    next_cond_inds = np.random.randint(
        0, y_ranking.shape[1], size=(n_subs_to_sample, n_conds_to_sample)
    )
    X_acquired = X[next_subs_inds]
    y_ranking_acquired = get_y_acquired(
        y_ranking, rem_inds, next_subs_inds, next_cond_inds
    )
    return X_acquired, y_ranking_acquired, next_subs_inds

def iteration_of_all_cond(X, y_ranking, train_inds, rem_inds, predicted_proba_array, n_subs_to_sample, n_conds_to_sample, train_smiles, substrate_selection) :
    """Selects a substrate that has either
    the lowest range across the scores
    the lowest average deviation of predicted probability values closest to 0.5

    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Ranking array of all substrates.
    train_inds : list of ints
        Indices selected for the training set candidates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_proba_array : np.ndarray of shape (n_substrates, n_conditions, n_conditions)
        Element at row i, column j corresponds to the RPC's predicted score of condition i being
        favorable over condition j.
    n_subs_to_sample : int
        Number of substrates to sample.
    n_conds_to_sample : int
        Number of reaction conditions to sample for one substrate.
    smiles_list : list of str
        List of smiles strings in the TRAINING dataset.
    substrate_selection : str {'farthest', 'quantile', 'skip_one'}
        How to select substrates.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_subs_to_sample, n_features)
        Input array of the substrates that were subject to data collection.
    y_ranking_acquired : np.ndarray of shape (n_subs_to_sample, n_conds)
        Ranking array of what the experimentalist would see.
        Unsampled reaction conditions have np.nan
    next_subs_inds : np.ndarray of shape (n_subs_to_sample,)
        Indices from REM_INDS that have been sampled in this iteration.
    """
    score_array = np.sum(predicted_proba_array, axis=2)
    next_subs_inds = np.argmin(np.ptp(score_array, axis=1))
    if n_subs_to_sample == 1 :
        next_subs_inds = [int(next_subs_inds)]
        next_cond_inds = np.array([[1,2,3,4]])
    X_acquired = X[next_subs_inds]
    y_ranking_acquired = get_y_acquired(y_ranking, rem_inds, next_subs_inds, next_cond_inds)
    return X_acquired, y_ranking_acquired, next_subs_inds

def iteration_of_explore_rfr(X, y_yield, rem_inds, predicted_yield_array, n_rxns_to_sample):
    """ Selects the next set of reactions based on highest uncertainty. 
    
    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_yield : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Yield array of all substrates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_yield_array : np.ndarray of shape (n_remaining_reactions, n_trees)
        Yield predictions with current RFR.
    n_rxns_to_sample : int
        Number of reactions to sample.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_rxns_to_sample, n_features)
        Input array of substrates that were subject to data collection.
    y_yield_acquired : np.ndarray of shape (n_reactions, )
        Yield values from the reactions collected.
    next_rxn_inds : list of ints of length (n_rxns_to_sample)
        Reaction indices that has been sampled from this iteration.
    """
    next_inds = np.argsort(np.std(predicted_yield_array, axis=1))[-1 * n_rxns_to_sample :]
    next_rxn_inds = [rem_inds[x] for x in next_inds]
    X_acquired = X[next_rxn_inds]
    y_yield_acquired = y_yield.flatten()[next_rxn_inds]
    return X_acquired, y_yield_acquired, next_rxn_inds

def iteration_of_exploit_rfr(X, y_yield, rem_inds, predicted_yield_array, n_rxns_to_sample):
    """ Selects the next set of reactions based on highest uncertainty. 
    
    Parameters
    ----------
    X : np.ndarray of shape (n_substrates, n_features)
        Full input array.
    y_yield : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Yield array of all substrates.
    rem_inds : list of ints
        Indices that are still available for sampling.
    predicted_yield_array : np.ndarray of shape (n_remaining_reactions, n_trees)
        Yield predictions with current RFR.
    n_rxns_to_sample : int
        Number of reactions to sample.

    Returns
    -------
    X_acquired : np.ndarray of shape (n_rxns_to_sample, n_features)
        Input array of substrates that were subject to data collection.
    y_yield_acquired : np.ndarray of shape (n_reactions, )
        Yield values from the reactions collected.
    next_rxn_inds : list of ints of length (n_rxns_to_sample)
        Reaction indices that has been sampled from this iteration.
    """
    next_inds = np.argsort(predicted_yield_array)[-1 * n_rxns_to_sample :]
    next_rxn_inds = [rem_inds[x] for x in next_inds]
    X_acquired = X[next_rxn_inds]
    y_yield_acquired = y_yield.flatten()[next_rxn_inds]
    return X_acquired, y_yield_acquired, next_rxn_inds