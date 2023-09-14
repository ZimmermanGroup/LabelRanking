import numpy as np
import pandas as pd
import argparse
import joblib
from rdkit import Chem
from itertools import combinations
from copy import deepcopy
from dataset_utils import *
from label_ranking import *
from sklr.tree import DecisionTreeLabelRanker
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kendalltau
from tqdm import tqdm

np.random.seed(42)
RAW_DATA = pd.read_excel(
    "datasets/natureHTE/natureHTE.xlsx",
    sheet_name="Report - Variable Conditions",
    usecols=["BB SMILES", "Chemistry", "Catalyst", "Base", "Rel. % Conv."],
)

AMINE_DATA = RAW_DATA[RAW_DATA["Chemistry"] == "Amine"]
SULFON_DATA = RAW_DATA[RAW_DATA["Chemistry"] == "Sulfonamide"].reset_index()
AMIDE_DATA = RAW_DATA[RAW_DATA["Chemistry"] == "Amide"].reset_index()
REAGENT_DATA = {}
base_descriptors = pd.read_excel(
    "datasets/natureHTE/reagent_desc.xlsx", sheet_name="Base"
)
cat_descriptors = pd.read_excel(
    "datasets/natureHTE/reagent_desc.xlsx", sheet_name="Catalyst"
)
for _, row in base_descriptors.iterrows():
    REAGENT_DATA.update({row[0]: row[1:].to_numpy()})
for _, row in cat_descriptors.iterrows():
    REAGENT_DATA.update({row[0]: row[1:].to_numpy()})

PERFORMANCE_DICT = {
    "kendall_tau": [],
    "reciprocal_rank": [],
    "mean_reciprocal_rank": [],
    "test_compound": [],
    "model": [],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument(
        "--subs_type",
        action="append",
        help="Which data to evaluate on. Should be either amine or sulfonamide or amide.",
    )
    parser.add_argument(
        "--adversarial",
        # action="append",
        help="Whether to perform adversarial controls. Should either be fp or onehot or random.",
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
        "--boost_lrt",
        action="store_true",
        help="Include boosting with LRT as base learner.",
    )
    parser.add_argument(
        "--boost_rpc",
        action="store_true",
        help="Include boosting with pairwise LR as base learner.",
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
    # if (
    #     args.lrrf is False
    #     and args.rpc is False
    #     and args.ibm is False
    #     and args.ibpl is False
    #     and args.baseline is False
    # ):
    #     parser.error("At least one model is required.")
    return args


def load_data(feature_type, output_type, dataset):
    """Prepares dataset for different algorithm types, feature inputs and labels.

    Parameters
    ----------
    feature_type : str {'onehot', 'random', 'fp'}
        Input format.

    output_type : str {'yield', 'ranking'}
        Type of algorithm the dataset will be used for.

    dataset : str {'amine', 'sulfonamide', 'amide'} or list of these strings.
        Considered only when output_type=='ranking'.
        Which component will be used as labels, which are subject to ranking.

    Returns
    -------
    X, y : np.2darray and 1darray of shape (n_samples, n_features) and (n_samples,)
        Arrays ready to be used for algorithms.
    """
    data_dict = {"amide": AMIDE_DATA, "amine": AMINE_DATA, "sulfonamide": SULFON_DATA}
    # Prepare the dataset dataframe.
    if type(dataset) == str:
        full_dataset = data_dict[dataset]
    elif type(dataset) == list:
        full_dataset = pd.concat([data_dict[x] for x in dataset])
    else:
        print("Unsupported type for dataset.")
        return None

    # Preparing inputs
    substrate_smiles_list = list(full_dataset["BB SMILES"].unique())
    n_subs = len(substrate_smiles_list)
    bases = list(full_dataset["Base"].unique())
    catalysts = list(full_dataset["Catalyst"].unique())

    if feature_type == "fp":
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        fp_array = np.zeros((n_subs, 1024))
        for i, smiles in enumerate(substrate_smiles_list):
            fp_array[i] = mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))

        if output_type == "ranking":
            X = deepcopy(fp_array)

        if output_type == "yield":
            row_array = []
            for i, row in full_dataset.iterrows():
                fp_portion = fp_array[
                    substrate_smiles_list.index(row["BB SMILES"])
                ].reshape(1, -1)
                cat_descriptors = REAGENT_DATA[row["Catalyst"]].reshape(1, -1)
                base_descriptors = REAGENT_DATA[row["Base"]].reshape(1, -1)
                row_array.append(
                    np.hstack((fp_portion, cat_descriptors, base_descriptors))
                )
            X = np.vstack(tuple(row_array))

    elif feature_type == "onehot":
        if output_type == "ranking":
            X = np.identity(n_subs)
        elif output_type == "yield":
            X = np.zeros(
                (
                    full_dataset.shape[0],
                    len(substrate_smiles_list) + len(bases) + len(catalysts),
                )
            )
            for i, row in full_dataset.iterrows():
                X[
                    i,
                    [
                        substrate_smiles_list.index(row["BB SMILES"]),
                        n_subs + catalysts.index(row["Catalyst"]),
                        n_subs + len(catalysts) + bases.index(row["Base"]),
                    ],
                ] = 1

    elif feature_type == "random":
        if output_type == "ranking":
            X = np.random.randint(3, size=(n_subs, 1024))
        elif output_type == "yield":
            substrate_random_fp = np.random.randint(
                3, size=(n_subs, 1024)
            )  # to simulate count FP
            reagent_random_desc = np.random.rand(4, 12)  # there are 12 descriptors
            X = np.zeros((full_dataset.shape[0], 1036))
            for i, row in full_dataset.iterrows():
                X[i] = np.concatenate(
                    (
                        substrate_random_fp[
                            substrate_smiles_list.index(row["BB SMILES"])
                        ].flatten(),
                        reagent_random_desc[
                            2 * catalysts.index(row["Catalyst"])
                            + bases.index(row["Base"])
                        ].flatten(),
                    )
                )

    # Preparing outputs
    if output_type == "ranking":
        y = np.zeros((n_subs, len(bases) * len(catalysts)))
        for i, row in full_dataset.iterrows():
            y[
                substrate_smiles_list.index(row["BB SMILES"]),
                2 * catalysts.index(row["Catalyst"]) + bases.index(row["Base"]),
            ] = row["Rel. % Conv."]
        y = yield_to_ranking(y)
    elif output_type == "yield":
        y = full_dataset["Rel. % Conv."].values.flatten()
    return X, y


def update_perf_dict(perf_dict, kt, rr, mrr, comp, model):
    perf_dict["kendall_tau"].append(kt)
    perf_dict["reciprocal_rank"].append(rr)
    perf_dict["mean_reciprocal_rank"].append(mrr)
    perf_dict["test_compound"].append(comp)
    perf_dict["model"].append(model)


def evaluate_lr_alg(test_rank, pred_rank, n_rxns, perf_dict, comp, model):
    kt = kendalltau(test_rank, pred_rank).statistic
    predicted_highest_yield_inds = np.argpartition(pred_rank.flatten(), n_rxns)[:n_rxns]
    rr = 1 / np.min(test_rank[predicted_highest_yield_inds])
    mrr = np.mean(
        np.reciprocal(test_rank[predicted_highest_yield_inds], dtype=np.float32)
    )
    print(f"Test compound {comp}")
    print(f"    kendall-tau={kt} // reciprocal_rank={rr} // mrr={mrr}")
    print()

    update_perf_dict(PERFORMANCE_DICT, kt, rr, mrr, comp, model)


def lr_eval(feature_type, substrate_type, parser, n_rxns):
    X, y = load_data(feature_type, "ranking", substrate_type)
    if parser.baseline:
        _, y_yields = load_data(feature_type, "yield", substrate_type)
        y_yields = y_yields.reshape(y.shape)

    test_fold = np.arange(X.shape[0])  ## LOOCV
    print(test_fold.shape)
    ps = PredefinedSplit(test_fold)
    for i, (train_ind, test_ind) in enumerate(ps.split()):
        X_train, X_test = X[train_ind, :], X[test_ind, :]
        rank_train, test_rank = y[train_ind, :], y[test_ind, :]
        if test_rank.shape[0] == 1:
            test_rank = test_rank.flatten()
        if len(np.unique(test_rank)) > 1:
            if parser.baseline:
                pred_rank = yield_to_ranking(np.mean(y_yields[train_ind, :], axis=0))
                evaluate_lr_alg(
                    test_rank, pred_rank, n_rxns, PERFORMANCE_DICT, i, "baseline"
                )

            if parser.rpc or parser.boost_rpc or parser.ibm or parser.ibpl:
                std = StandardScaler()
                train_X_std = std.fit_transform(X_train)
                test_X_std = std.transform(X_test)
                if parser.rpc:
                    rpc_lr = RPC(
                        base_learner=LogisticRegression(C=1), cross_validator=None
                    )
                    rpc_lr.fit(train_X_std, rank_train)
                    rpc_pred_rank = rpc_lr.predict(test_X_std)
                    print(test_rank)
                    evaluate_lr_alg(
                        test_rank, rpc_pred_rank, n_rxns, PERFORMANCE_DICT, i, "RPC"
                    )
                if parser.ibm:
                    ibm = IBLR_M(n_neighbors=3, metric="euclidean")
                    ibm.fit(X_train, rank_train)
                    ibm_pred_rank = ibm.predict(test_X_std)
                    evaluate_lr_alg(
                        test_rank, ibm_pred_rank, n_rxns, PERFORMANCE_DICT, i, "IBM"
                    )
                if parser.ibpl:
                    ibpl = IBLR_PL(n_neighbors=3, metric="euclidean")
                    ibpl.fit(X_train, rank_train)
                    ibpl_pred_rank = ibpl.predict(test_X_std)
                    evaluate_lr_alg(
                        test_rank, ibpl_pred_rank, n_rxns, PERFORMANCE_DICT, i, "IBPL"
                    )

            if parser.lrrf:
                lrrf = LabelRankingRandomForest(n_estimators=50)
                lrrf.fit(X_train, rank_train)
                lrrf_pred_rank = lrrf.predict(X_test)
                evaluate_lr_alg(
                    test_rank, lrrf_pred_rank, n_rxns, PERFORMANCE_DICT, i, "LRRF"
                )

            if parser.lrt:
                lrt = DecisionTreeLabelRanker(
                    random_state=42, min_samples_split=rank_train.shape[1] * 2
                )
                lrt.fit(X_train, rank_train)
                lrt_pred_rank = lrt.predict(X_test)
                evaluate_lr_alg(
                    test_rank, lrt_pred_rank, n_rxns, PERFORMANCE_DICT, i, "LRT"
                )


def rfr_eval(
    feature_type,
    substrate_type,
    n_rxns,
    params={"n_estimators": [30, 100, 200], "max_depth": [5, 10, None]},
    random_state=42,
):
    X, y = load_data(feature_type, "yield", substrate_type)
    X_ohe, y_ohe = load_data("onehot", "yield", substrate_type)
    assert np.all(y == y_ohe)

    test_fold = np.argmax(X_ohe, axis=1)
    ps = PredefinedSplit(test_fold)
    print("Evaluating RandomForestRegressor....")
    for i, (train_ind, test_ind) in tqdm(enumerate(ps.split())):
        X_train, X_test = X[train_ind, :], X[test_ind]
        X_train_ohe = X_ohe[train_ind, :]
        y_train, y_test = y[train_ind], y[test_ind]
        if len(np.unique(y_test)) > 1:
            groups = np.argmax(X_train_ohe, axis=1).flatten()
            gfk = GroupKFold(n_splits=len(np.unique(groups)))
            # inner_test_fold
            gcv = GridSearchCV(
                RandomForestRegressor(random_state=random_state),
                param_grid=params,
                scoring="r2",
                n_jobs=-1,
                cv=gfk.split(X_train, y_train, groups=groups),
            )
            gcv.fit(X_train, y_train)
            y_pred = gcv.best_estimator_.predict(X_test)
            # print(f"    Test compound {i} - RMSE={round(mean_squared_error(y_test, y_pred, squared=False), 1)}, R2={round(r2_score(y_test, y_pred), 2)}")
            y_ranking = yield_to_ranking(y_test)
            kt = kendalltau(y_ranking, yield_to_ranking(y_pred).flatten()).statistic
            largest_yield_inds = np.argpartition(-1 * y_pred, n_rxns)[:n_rxns]
            reciprocal_rank = 1 / np.min(y_ranking[largest_yield_inds])
            mean_reciprocal_rank = np.mean(
                np.reciprocal(y_ranking[largest_yield_inds], dtype=np.float32)
            )
            update_perf_dict(
                PERFORMANCE_DICT, kt, reciprocal_rank, mean_reciprocal_rank, i, "rfr"
            )
            print(f"Test compound {i}")
            print(
                f"    kendall-tau={kt} // reciprocal_rank={reciprocal_rank} // mrr={mean_reciprocal_rank}"
            )
            print()
    print()


if __name__ == "__main__":
    parser = parse_args()
    feature = parser.adversarial
    subs_type = parser.subs_type

    if (
        parser.lrrf
        or parser.rpc
        or parser.ibm
        or parser.ibpl
        or parser.lrt
        or parser.baseline
    ):
        lr_eval(feature, subs_type, parser, n_rxns=1)

    if parser.rfr:
        rfr_eval(feature, subs_type, 1)

    # Saving the results
    if parser.save:
        joblib.dump(
            PERFORMANCE_DICT,
            f"performance_excels/natureHTE/{subs_type[0]}_{feature}.joblib",
        )
