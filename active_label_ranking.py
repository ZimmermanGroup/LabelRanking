from label_ranking import *
from dataloader import *
from evaluator import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Cluster import Butina
from scipy.stats.mstats import rankdata
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm import tqdm

import argparse

np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Specify the evaluation to run.")
    parser.add_argument(
        "--dataset",
        choices=["amine", "fragment", "whole_amine", "whole_bromide"],
        help="Which of the natureHTE dataset to use.",
    )
    parser.add_argument(
        "--strategy",
        choices=["condition_first", "lowest_diff", "two_condition_pairs", "random", "full_rfr", "full_rpc"],
        help="Which AL acquisition strategy to use.",
    )
    parser.add_argument(
        "--initialization",
        choices=["random", "cluster"],
        help="How to select the 6 initial substrate pairs.",
    )
    parser.add_argument(
        "--substrate_selection",
        choices=["farthest","quantile","skip_one"],
        help="How to select the substrates for the condition-first and two-condition-pair strategies."
    )
    parser.add_argument(
        "--n_initial_subs",
        default=6,
        type=int,
        help="Number of initial substrates to sample.",
    )
    parser.add_argument(
        "--n_conds_to_sample",
        default=2,
        type=int,
        help="Number of reaction conditions to sample for each substrate.",
    )
    parser.add_argument(
        "--n_subs_to_sample",
        default=2,
        type=int,
        help="Number of substrates to sample in each iteration.",
    )
    parser.add_argument(
        "--n_test_subs",
        default=4,
        type=int,
        help="Number of test substrates with different top-performing condition to leave out as test cases.\\\
            A total of n_conditions * n_test_subs number of substrates will be left out.",
    )
    parser.add_argument(
        "--n_evals",
        default=25,
        type=int,
        help="""Number of evaluations to conduct.
            If the initialization process is random, does this number of initialization selections and train_test splits,
            which gives a total number of squared iterations.
            If the initialization process is deterministic, does this number of random train_test splits.
        """,
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save resulting scores in an excel file.",
    )
    args = parser.parse_args()
    return args


def rr(test_yield, test_rank, pred_rank):
    predicted_highest_yield_inds = np.argpartition(pred_rank, 1, axis=1)[:, :1]
    best_retrieved_yield = [
        np.max(test_yield[i, row]) for i, row in enumerate(predicted_highest_yield_inds)
    ]
    actual_inds_with_that_yield = [
        np.where(test_yield[i, :] == best_y)[0]
        for i, best_y in enumerate(best_retrieved_yield)
    ]
    rr = np.array(
        [1 / np.min(test_rank[a, x]) for a, x in enumerate(actual_inds_with_that_yield)]
    )
    return np.mean(rr)


def kt(test_rank, pred_rank):
    kt = np.array(
        [
            kendalltau(test_rank[i, :], pred_rank[i, :]).statistic
            for i in range(pred_rank.shape[0])
        ]
    )
    return np.mean(kt)


def update_perf_dict(dict_to_update, rr, kt, n_total_rxn, eval_iter, init_iter):
    dict_to_update["Reciprocal Rank"].append(rr)
    dict_to_update["Kendall Tau"].append(kt)
    dict_to_update["Substrates Sampled"].append(n_total_rxn)
    dict_to_update["Evaluation Iteration"].append(eval_iter)
    dict_to_update["Initialization Iteration"].append(init_iter)


def get_train_test_inds(y_ranking, n_test_cases):
    """Randomly selects the test substrates to evaluate on.
    We assume that the dataset of interest has enough number of substrates that has
    each reaction condition as the best-performing one.

    Parameters
    ----------
    y_ranking : np.ndarray of shape (n_substrates, n_conditions)
        Ground truth ranking array of the dataset of interest.
    n_test_cases : int
        Number of test substrates in each reaction condition to select.

    Returns
    -------
    train_inds, test_inds : list of ints
        Which indices will be used for which set.
    """
    test_inds = []
    for i in range(n_test_cases):
        if len(np.where(y_ranking[:, i] == 1)[0]) > n_test_cases :
            test_inds.extend(
                list(
                    np.random.choice(
                        np.where(y_ranking[:, i] == 1)[0], n_test_cases, replace=False
                    )
                )
            )
    train_inds = [x for x in range(y_ranking.shape[0]) if x not in test_inds]
    return train_inds, test_inds


def initial_substrate_selection(
    initialization, train_inds, n_initial_subs, smiles_list=None
):
    """Selects the initial set of substrates to sample with all reaction conditions.

    Parameters
    ----------
    initialization : str {'random', 'cluster'}
        • random : randomly selecting
        • cluster : Centroids of clusters

    train_inds : list of ints
        Indices out of full dataset that can be used as training set.

    n_initial_subs : int
        Number of substrates to sample.

    smiles_list : list of str
        List of smiles in the TRAINING dataset.

    Returns
    -------
    initial_inds : np.ndarray of shape (n_initial_subs, )
        Indices within the training indices chosen as initial substrates.

    rem_inds : list of ints
        Indices within the training indices that are still available for sampling.
    """
    if initialization == "random":
        initial_inds = np.random.choice(train_inds, n_initial_subs, replace=False)

    elif initialization == "cluster":
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=3)
        fp_list = [
            mfpgen.GetCountFingerprint(Chem.MolFromSmiles(x)) for x in smiles_list
        ]
        dists = []
        for i in range(1, len(fp_list)):
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            dists.extend([1 - x for x in similarities])
        ### Gets the distance threshold that has highest number of membership in the
        ### n_initial_subs-th cluster
        cluster_membership = np.zeros((4, n_initial_subs))
        all_clusters = []
        for i in range(5, 9):
            clusters = Butina.ClusterData(dists, len(fp_list), 0.1 * i, isDistData=True)
            non_single_clusters = sorted([len(x) for x in clusters if len(x) > 1])[::-1]
            if len(non_single_clusters) >= n_initial_subs:
                cluster_membership[i - 5] = non_single_clusters[:n_initial_subs]
            else:
                cluster_membership[
                    i - 5, : len(non_single_clusters)
                ] = non_single_clusters
            all_clusters.append(clusters)
        selected = False
        while not selected:
            for i in range(1, n_initial_subs + 1):
                if np.any(cluster_membership[:, -1 * i]) > 0:
                    clusters_to_use = all_clusters[
                        np.argmax(cluster_membership[:, -1 * i])
                    ]
                    selected = True
        initial_inds = [
            train_inds[cluster[0]] for cluster in clusters_to_use[:n_initial_subs]
        ][:n_initial_subs]
    rem_inds = [x for x in train_inds if x not in initial_inds]
    return initial_inds, rem_inds


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
    # print("LEN SMILES", len(sampled_smiles), len(rem_smiles))
    avg_deviation = np.mean(np.abs(predicted_proba_array - 0.5), axis=0)
    inds = np.unravel_index(
        np.argsort(avg_deviation, axis=None),
        (predicted_proba_array.shape[1], predicted_proba_array.shape[2]),
    )
    conds = [inds[0][0], inds[1][0]]
    while len(conds) < n_conds_to_sample:
        rem_conds = [x for x in range(avg_deviation.shape[1]) if x not in conds]
        conds.append(np.argmin(np.mean(avg_deviation[rem_conds, conds], axis=1)))
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
    )[::-1]
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


def AL_loops(parser, X, y_ranking, y_yield, smiles_list):
    """Conducts loops of active learning until all substrates have been sampled.

    Parameters
    ----------
    parser : argparse object
        Defines how to conduct the experiments.
    X : np.ndarray of shape (n_substrates, n_features)
        Input array of substrates for models.
    y_ranking : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Output array of rankings between reaction conditions for each substrate.
    y_yield : np.ndarray of shape (n_substrates, n_rxn_conditions)
        Array of yield values for each substrate under each reaction condition.

    Returns
    -------
    perf_dict : dict

    """
    perf_dict = {
        "Reciprocal Rank": [],
        "Kendall Tau": [],
        "Substrates Sampled": [],
        "Evaluation Iteration": [],
        "Initialization Iteration": [],
    }
    # Splitting the train-test split so that the random selections don't get affected
    train_inds_list = []
    test_inds_list = []
    for n_iter in range(parser.n_evals):  # Train test splits
        if parser.dataset == "amine":
            train_inds, test_inds = get_train_test_inds(y_ranking, parser.n_test_subs)
            train_inds_list.append(train_inds)
            test_inds_list.append(test_inds)
            n_evals = parser.n_evals
        else : #  conduct 
            top_class = np.where(y_ranking == 1)[1]
            inds = np.arange(len(top_class))
            train_inds, test_inds = train_test_split(inds, test_size=0.5, random_state=42+n_iter, stratify=top_class)
            train_inds_list.append(train_inds)
            test_inds_list.append(test_inds)
            train_inds_list.append(test_inds)
            test_inds_list.append(train_inds)
            n_evals = 2 * parser.n_evals
    
    for n_iter in tqdm(range(n_evals)):  # Train test splits
        train_inds = train_inds_list[n_iter]
        test_inds = test_inds_list[n_iter]
        train_smiles = [x for i, x in enumerate(smiles_list) if i in train_inds]

        if parser.strategy != "full_rfr":
            X_test = X[test_inds]
            y_test = y_ranking[test_inds]
            y_test_yield = y_yield[test_inds]
        else:
            train_inds_for_rfr = []
            test_inds_for_rfr = []
            for i in range(y_ranking.shape[0]):
                if i in train_inds:
                    train_inds_for_rfr.extend(
                        [y_ranking.shape[1] * i + x for x in range(y_ranking.shape[1])]
                    )
                elif i in test_inds:
                    test_inds_for_rfr.extend(
                        [y_ranking.shape[1] * i + x for x in range(y_ranking.shape[1])]
                    )
            X_test = X[test_inds_for_rfr]
            y_test = y_ranking[test_inds]
            y_test_yield = y_yield[test_inds]

        if parser.initialization == "random":
            n_init = parser.n_evals
        else:
            if parser.strategy != "random":
                n_init = 1
            else:
                if parser.dataset == "amine" :
                    n_init = parser.n_evals
                else :
                    n_init = 2

        if parser.strategy not in ["full_rpc", "full_rfr"]:
            initial_inds_list = []
            rem_inds_list = []
            # To separate the random process from random selection in the loop below
            for n in range(n_init) :
                initial_inds, rem_inds = initial_substrate_selection(
                    parser.initialization,
                    train_inds,
                    parser.n_initial_subs,
                    smiles_list=train_smiles,
                )
                initial_inds_list.append(initial_inds)
                rem_inds_list.append(rem_inds)
            for n in range(n_init):  # Initializations
                initial_inds = initial_inds_list[n]
                rem_inds = rem_inds_list[n]
                copied_rem_inds = deepcopy(rem_inds)
                X_sampled = X[initial_inds]
                y_sampled = y_ranking[initial_inds]

                rpc = RPC()
                rpc.fit(X_sampled, y_sampled)
                y_pred = rpc.predict(X_test)
                rem_proba_array = rpc.predict_proba(X[rem_inds])

                while len(rem_inds) > 2:
                    rr_score = rr(y_test_yield, y_test, y_pred)
                    kt_score = kt(y_test, y_pred)
                    update_perf_dict(
                        perf_dict, rr_score, kt_score, y_sampled.shape[0], n_iter, n
                    )
                    if parser.strategy == "lowest_diff":
                        (
                            X_acquired,
                            y_ranking_acquired,
                            next_subs_inds,
                        ) = iteration_of_lowest_diff(
                            X,
                            y_ranking,
                            rem_inds,
                            rem_proba_array,
                            parser.n_subs_to_sample,
                            parser.n_conds_to_sample,
                        )
                    elif parser.strategy == "condition_first":
                        (
                            X_acquired,
                            y_ranking_acquired,
                            next_subs_inds,
                        ) = iteration_of_cond_first(
                            X,
                            y_ranking,
                            train_inds,
                            rem_inds,
                            rem_proba_array,
                            parser.n_subs_to_sample,
                            parser.n_conds_to_sample,
                            train_smiles,
                            parser.substrate_selection
                        )
                    elif parser.strategy == "two_condition_pairs":
                        (
                            X_acquired,
                            y_ranking_acquired,
                            next_subs_inds,
                        ) = iteration_of_two_cond_pairs(
                            X,
                            y_ranking,
                            train_inds,
                            rem_inds,
                            rem_proba_array,
                            parser.n_subs_to_sample,
                            parser.n_conds_to_sample,
                            train_smiles,
                            parser.substrate_selection
                        )
                    elif parser.strategy == "random":
                        (
                            X_acquired,
                            y_ranking_acquired,
                            next_subs_inds,
                        ) = iteration_of_random(
                            X,
                            y_ranking,
                            rem_inds,
                            parser.n_subs_to_sample,
                            parser.n_conds_to_sample,
                        )
                    X_sampled = np.vstack((X_sampled, X_acquired))
                    y_sampled = np.vstack((y_sampled, y_ranking_acquired))
                    rem_inds = [
                        x for i, x in enumerate(rem_inds) if i not in next_subs_inds
                    ]
                    rpc.fit(X_sampled, y_sampled)
                    y_pred = rpc.predict(X_test)
                    rem_proba_array = rpc.predict_proba(X[rem_inds])
                if n_init > 1 and parser.n_evals == 1 :
                    rem_inds = deepcopy(copied_rem_inds)

        elif parser.strategy == "full_rpc":
            X_train = X[train_inds]
            y_train = y_ranking[train_inds]
            rpc = RPC()
            rpc.fit(X_train, y_train)
            y_pred = rpc.predict(X_test)
            update_perf_dict(
                perf_dict,
                rr(y_test_yield, y_test, y_pred),
                kt(y_test, y_pred),
                y_train.shape[0],
                n_iter,
                0,
            )
        elif parser.strategy == "full_rfr":
            X_train = X[train_inds_for_rfr]
            y_train = y_yield[train_inds].flatten()
            inner_ps = PredefinedSplit(
                np.repeat(
                    np.arange(len(train_inds)), y_ranking.shape[1]
                )  # assume that all reaction conditions are aligned in a specific order.
            )
            gcv = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid={
                    "n_estimators": [30, 100, 200],
                    "max_depth": [5, 10, None],
                },
                scoring="r2",
                n_jobs=-1,
                cv=inner_ps,
            )
            gcv.fit(X_train, y_train)
            y_pred_yield = gcv.predict(X_test)
            y_pred = (
                y_ranking.shape[1]
                + 1
                - rankdata(
                    y_pred_yield.reshape((len(test_inds), y_ranking.shape[1])), axis=0
                )
            )
            update_perf_dict(
                perf_dict,
                rr(y_test_yield, y_test, y_pred),
                kt(y_test, y_pred),
                y_train.shape[0],
                n_iter,
                0,
            )
    return perf_dict


def main(parser):
    # array preparation
    if parser.dataset == "amine"  :
        dataset = NatureDataset(False, parser.dataset, 1)
        smiles_list = [
            x for i, x in enumerate(dataset.smiles_list) if i not in dataset.validation_rows
        ]
    else:
        dataset = ScienceDataset(False, parser.dataset, 1)
        smiles_list = dataset.smiles_list

    if parser.strategy == "full_rfr":
        # Please note that we do not consider training a regressor for any of the Science datasets.
        regressor_dataset = NatureDataset(True, parser.dataset, 1)
        X = regressor_dataset.X_fp
    else:
        X = dataset.X_fp
    # both y arrays are of shape (n_substrates, n_conditions)
    y_ranking = dataset.y_ranking
    y_yield = dataset.y_yield

    perf_dict = AL_loops(parser, X, y_ranking, y_yield, smiles_list)
    perf_df = pd.DataFrame(perf_dict)

    if parser.save:
        filename = f"performance_excels/AL/{parser.dataset}_{parser.n_initial_subs}_{parser.n_conds_to_sample}_{parser.n_subs_to_sample}_{parser.n_test_subs}.xlsx"
        if parser.substrate_selection == None or parser.strategy == "random":
            sheetname = f"{parser.strategy}_{parser.initialization}"
        else :
            if parser.strategy == "two_condition_pairs":
                strategyname = "two_condition"
            else :
                strategyname = parser.strategy
            sheetname = f"{strategyname}_{parser.substrate_selection[:5]}_{parser.initialization}"
        if os.path.exists(filename):
            with pd.ExcelWriter(filename, mode="a") as writer:
                perf_df.to_excel(
                    writer, sheet_name=sheetname
                )
        else:
            perf_df.to_excel(
                filename, sheet_name=sheetname
            )

if __name__ == "__main__":
    parser = parse_args()
    main(parser)
