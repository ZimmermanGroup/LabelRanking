import numpy as np
import pandas as pd
import argparse
import joblib
from copy import deepcopy
from dataset_utils import *
from label_ranking import *
from sklr.tree import DecisionTreeLabelRanker
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

# This script runs leave-one-substrate out. 

performance_dict = {"kendall_tau":[], "reciprocal_rank":[], "test_compound":[], "model":[]}


def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument(
        "--test_component",
        action="append",
        help="Which reaction components to consider as 'labels'. Should be either catalyst_ratio or amine_ratio.",
    )
    parser.add_argument(
        "--adversarial",
        action="append",
        help="Whether to perform adversarial controls. Should either be desc or onehot.",
    )
    parser.add_argument(
        "--rfr", action="store_true", help="Include Random Forest Regressor."
    )
    parser.add_argument(
        "--lrrf", action="store_true", help="Include Label Ranking RF as in Qiu, 2018"
    )
    parser.add_argument(
        "--lrt",
        action="store_true",
        help="Include Label Ranking Tree as in Hüllermeier, 2008",
    )
    parser.add_argument(
        "--rpc",
        action="store_true",
        help="Include Pairwise label ranking as in Hüllermeier, 2008",
    )
    parser.add_argument(
        "--ibm",
        action="store_true",
        help="Include Instance-based label ranking with Mallows model as in Hüllermeier, 2009",
    )
    parser.add_argument(
        "--ibpl",
        action="store_true",
        help="Include Instance-based label ranking with Plackett=Luce model as in Hüllermeier, 2010",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline models - avg_yield.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save resulting scores in an excel file.",
    )
    args = parser.parse_args()
    if (
        args.lrrf is False
        and args.rpc is False
        and args.ibm is False
        and args.ibpl is False
        and args.baseline is False
    ):
        parser.error("At least one model is required.")
    return args


def prep_data(feature_type, output_type, test_component):
    """ Prepares dataset for algorithm inputs.
    
    Parameters
    ----------
    feature_type : str {descriptor, onehot}
        Type of input feature. 
    output_type : str {yield, ranking}
        Type of value to predict.
    test_component : str {catalyst_ratio, amine_ratio}
        Which component to put as labels.
    
    Returns
    -------
    X, y : np.ndarray of shape (n_samples, n_features) and (n_samples,)
        Arrays ready to be used for algorithms.
    """
    pass


def update_perf_dict(perf_dict, kt, rr, comp, model):
    perf_dict["kendall_tau"].append(kt)
    perf_dict["reciprocal_rank"].append(rr)
    perf_dict["test_compound"].append(comp)
    perf_dict["model"].append(model)
    # return perf_dict


def evaluate_lr_alg(test_rank, pred_rank, n_rxns, perf_dict, comp, model) :
    kt = kendalltau(test_rank, pred_rank).statistic
    predicted_highest_yield_inds = np.argpartition(pred_rank.flatten(), n_rxns)[:n_rxns]
    rr = 1/np.min(test_rank[predicted_highest_yield_inds])
    update_perf_dict(
        perf_dict, kt, rr,  comp, model
    )
    with open("performance_excels/informer_predictions.txt", "a") as f :
        f.write(f"============={model} predicting {comp}=============\n")
        f.write(f"Actual: {test_rank}\n")
        f.write(f"Predicted: {pred_rank}\n")
        f.write(str(rr)+"\n")
        f.write("\n")
        f.write("\n")


if __name__ == "__main__" :
    parser = parse_args()

    # Load dataset
    informer_df = pd.read_excel("datasets/Informer.xlsx").iloc[:40,:]
    desc_df = pd.read_excel("datasets/Informer.xlsx", sheet_name="descriptors", usecols=[0,1,2,3,4]).iloc[:40, :]
    smiles = pd.read_excel("datasets/Informer.xlsx", sheet_name="smiles", header=None)

    # Dropping compounds where all yields are below 20%
    cols_to_erase = []
    for col in informer_df.columns :
        if np.all(informer_df.loc[:,col].to_numpy() < 20) :
            cols_to_erase.append(col)
    # print(cols_to_erase)
    informer_df = informer_df.loc[:, [x for x in range(1,19) if x not in cols_to_erase]] # leaves 11 compounds
    smiles_list = [x[0] for i, x in enumerate(smiles.values.tolist()) if i+1 not in cols_to_erase]

    # Preparing count morgan fingerprints
    if "desc" in parser.adversarial:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024)
        cfp = [mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(x)) for x in smiles_list]
        cfp_array = np.vstack(tuple(cfp))
        # removing bits that are all 0 or 1 in each column
        cols_to_remove = []
        for i in range(cfp_array.shape[1]) :
            if len(np.unique(cfp_array[:,i])) == 1 :
                cols_to_remove.append(i)
        cfp_array = cfp_array[:, [x for x in range(cfp_array.shape[1]) if x not in cols_to_remove]]
    
    # Initializing performance logging
    if "amine_ratio" in parser.test_component :
        performance_dict_list = [deepcopy(performance_dict), deepcopy(performance_dict)]
        # When 20% of all possible reaction conditions can be used
        n_rxns = 4
    if "catalyst_ratio" in parser.test_component :
        performance_dict_list = [deepcopy(performance_dict), deepcopy(performance_dict), deepcopy(performance_dict), deepcopy(performance_dict)]
        n_rxns = 2

    # Training label ranking algorithms
    if (
        parser.lrrf
        or parser.rpc
        or parser.ibm
        or parser.ibpl
        or parser.lrt
        or parser.baseline
    ):
        yield_array = informer_df.to_numpy()
        if "amine_ratio" in parser.test_component :
            yield_array_list = [
                yield_array[[x for x in range(yield_array.shape[0]) if x%8 < 4], :].T, # 11 x 20
                yield_array[[x for x in range(yield_array.shape[0]) if x%8 >= 4], :].T
            ]
        elif "catalyst_ratio" in parser.test_component:
            yield_array_list = [
                yield_array[[y for y in range(yield_array.shape[0]) if y%4 == x],:].T for x in range(4) # 11 x 10
            ]
        for i in range(yield_array.shape[1]) : # For each compound as test
            for j, indiv_yield_array in enumerate(yield_array_list) : 
                test_rank = yield_to_ranking(indiv_yield_array[i, :])
                train_row_inds = [x for x in range(indiv_yield_array.shape[0]) if x!=i]
                X_train = cfp_array[train_row_inds, :]
                y_train = indiv_yield_array[train_row_inds,:]
                rank_train = yield_to_ranking(y_train)
                if parser.baseline :
                    pred_rank = yield_to_ranking(np.mean(y_train, axis=0))
                    evaluate_lr_alg(test_rank, pred_rank, n_rxns, performance_dict_list[j], i, "baseline")
                    
                if parser.rpc :
                    std = StandardScaler()
                    train_X_std = std.fit_transform(X_train)
                    test_X_std = std.transform(cfp_array[i, :].reshape(1,-1))
                    rpc_lr = RPC(
                        base_learner=LogisticRegression(C=1), cross_validator=None
                    )
                    rpc_lr.fit(train_X_std, rank_train)
                    rpc_pred_rank = rpc_lr.predict(test_X_std)
                    evaluate_lr_alg(test_rank, rpc_pred_rank, n_rxns, performance_dict_list[j], i, "RPC")

                if parser.lrrf:
                    lrrf = LabelRankingRandomForest(n_estimators=50)
                    lrrf.fit(X_train, rank_train)
                    lrrf_pred_rank = lrrf.predict(cfp_array[i, :].reshape(1,-1))
                    evaluate_lr_alg(test_rank, lrrf_pred_rank, n_rxns, performance_dict_list[j], i, "LRRF")

                if parser.lrt:
                    lrt = DecisionTreeLabelRanker(
                        random_state=42, min_samples_split=rank_train.shape[1] * 2
                    )
                    lrt.fit(X_train, rank_train)
                    lrt_pred_rank = lrt.predict(cfp_array[i, :].reshape(1,-1))
                    evaluate_lr_alg(test_rank, lrt_pred_rank, n_rxns, performance_dict_list[j], i, "LRT")

                if parser.ibm :
                    ibm = IBLR_M(n_neighbors=3, metric="euclidean")
                    ibm.fit(X_train, rank_train)
                    ibm_pred_rank = ibm.predict(cfp_array[i, :].reshape(1,-1))
                    evaluate_lr_alg(test_rank, ibm_pred_rank, n_rxns, performance_dict_list[j], i, "IBM")
                    
                if parser.ibpl :
                    ibpl = IBLR_PL(n_neighbors=3, metric="euclidean")
                    ibpl.fit(X_train, rank_train)
                    ibpl_pred_rank = ibpl.predict(cfp_array[i, :].reshape(1,-1))
                    evaluate_lr_alg(test_rank, ibpl_pred_rank, n_rxns, performance_dict_list[j], i, "IBPL")

    # Training a random forest regressor
    if parser.rfr :
        # Preparing arrays
        X_arrays = []
        y_arrays = []
        for i, (_, yield_vals) in enumerate(informer_df.items()):
            y_arrays.append(yield_vals.to_numpy())
            X_arrays.append(np.hstack((
                np.tile(cfp_array[i,:], (40, 1)),
                desc_df.to_numpy()
            )))
        X_array = np.vstack(tuple(X_arrays))
        y_array = np.concatenate(tuple(y_arrays))

        if "amine_ratio" in parser.test_component :
            X_list = [
                X_array[[x for x in range(X_array.shape[0]) if x%8 < 4], :], 
                X_array[[x for x in range(X_array.shape[0]) if x%8 >= 4], :]
            ]
            y_list = [
                y_array[[x for x in range(X_array.shape[0]) if x%8 < 4]],
                y_array[[x for x in range(X_array.shape[0]) if x%8 >= 4]]
            ]
            test_fold = np.repeat(np.arange(11), 20)
            inner_test_fold = np.repeat(np.arange(10), 20)
            print(X_list[0].shape, X_list[1].shape, np.unique(X_list[0][:,-1]))

        elif "catalyst_ratio" in parser.test_component:
            X_list = [
                X_array[[y for y in range(X_array.shape[0]) if y%4 == x],:] for x in range(4)
            ]
            y_list = [
                y_array[[y for y in range(len(y_array)) if y%4 == x]] for x in range(4)
            ]
            test_fold = np.repeat(np.arange(11), 10)
            inner_test_fold = np.repeat(np.arange(10), 10)

        for a, (X, y) in enumerate(zip(X_list, y_list)) :
            ps = PredefinedSplit(test_fold)
            for i, (train_ind, test_ind) in enumerate(ps.split()) :
                X_train, X_test = X[train_ind, :], X[test_ind]
                y_train, y_test = y[train_ind], y[test_ind]

                params = {
                    "n_estimators":[30,100,200],
                    "max_depth":[3,10,None]
                }
                gcv = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid=params,
                    scoring="r2",
                    n_jobs=-1,
                    cv=PredefinedSplit(inner_test_fold)
                )
                gcv.fit(X_train, y_train)
                y_pred = gcv.best_estimator_.predict(X_test)
                y_ranking = yield_to_ranking(y_test)
                kt = kendalltau(y_ranking, yield_to_ranking(y_pred).flatten()).statistic
                largest_yield_inds = np.argpartition(-1*y_pred, n_rxns)[:n_rxns]
                reciprocal_rank = 1/np.min(y_ranking[largest_yield_inds])
                mean_reciprocal_rank = np.mean(np.reciprocal(y_ranking[largest_yield_inds]))

                performance_dict_list[a]["kendall_tau"].append(kt)
                performance_dict_list[a]["reciprocal_rank"].append(reciprocal_rank)
                performance_dict_list[a]["test_compound"].append(i)
                performance_dict_list[a]["model"].append("rfr")
    if parser.save :
        joblib.dump(
            performance_dict_list, 
            f"performance_excels/informer_performance_dict_{parser.test_component[0]}.joblib"
        )