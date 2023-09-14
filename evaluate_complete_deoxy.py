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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kendalltau
import torch
import torch.nn.functional as nnf

PERFORMANCE_DICT = {
    "kendall_tau": [],
    "reciprocal_rank": [],
    "mean_reciprocal_rank": [],
    "test_compound": [],
    "model": [],
}
SUBSTRATE_SMILES = [
    "OCCCCC1=CC=CC=C1",
    "OC(C)CCC1=CC=CC=C1",
    "OC(C)(C)CCC1=CC=CC=C1",
    "OCCCC1=NC(C2=CC=CC=C2)=C(C3=CC=CC=C3)O1",
    "O=C(C(N(C)C=N1)=C1N2C)N(CCCCC(O)C)C2=O",
    "O=C(C1=CC=CC=C1)N(C(C2=CC=CC=C2)=O)C3=NC=NC4=C3N=CN4[C@H]5[C@H](OC(C6=CC=CC=C6)=O)[C@H](OC(C7=CC=CC=C7)=O)[C@@H](CO)O5",
    "ClC1=CC([N+]([O-])=O)=CC=C1/N=N/C2=CC=C(N(CC)CCO)C=C2",
    "OCC1N(CCCC2=CC=CC=C2)CCCC1",
    "O[C@@H]1C[C@H](OCC2=CC=CC=C2)C1",
    "O=C(OC)[C@H]1N(C(OC(C)(C)C)=O)C[C@H](O)C1",
    "O=C(OC)[C@H]1N(C(OC(C)(C)C)=O)C[C@@H](O)C1",
    "CC1(C)OC[C@H]([C@@H]2[C@@H](O)[C@@H](OC(C)(C)O3)[C@@H]3O2)O1",
    "O[C@@H]1CO[C@]2([H])[C@@H](OC(C)=O)CO[C@@]21[H]",
    "O[C@@H]1CC[C@@H](N2C(C(C=CC=C3)=C3C2=O)=O)CC1",
    "O[C@H]1CC[C@@H](N2C(C(C=CC=C3)=C3C2=O)=O)CC1",
    "O[C@]1([H])C[C@H](CC2)N(C(OC(C)(C)C)=O)[C@H]2C1",
    "OCC(C=C1)=CC=C1C2=CC=CC=C2",
    "OCC1=CC=C(S(N(CCC)CCC)(=O)=O)C=C1",
    "ClC(N=C1CCCC)=C(CO)N1CC(C=C2)=CC=C2C3=C(C4=NN=NN4C(C5=CC=CC=C5)(C6=CC=CC=C6)C7=CC=CC=C7)C=CC=C3",
    "OCN1C(C)=CC(C)=N1",
    "COC1=C(OCCCC(OC)=O)C=C([N+]([O-])=O)C(C(O)C)=C1",
    "O[C@@H](C1=CC=CC=C1)[C@H](C)N2CCCC2",
    "C/C(C)=C/CC/C(C)=C/CC/C(C)=C/CO",
    "CCCCCCCC(O)C=C",
    "C/C(C)=C/CCC(O)(C=C)C",
    "O=C(C1=CC=C(Cl)C=C1)N2C3=CC=C(OC)C=C3C(CCO)=C2C",
    "CC1=NC=C([N+]([O-])=O)N1CCO",
    "O[C@@H](C1)CC[C@@]2(C)C1=CC[C@]3([H])[C@]2([H])CC[C@@]4(C)[C@@]3([H])CC=C4C5=CN=CC=C5",
    "O=C1C(O)CCO1",
    "O=C1C=C[C@@]2(C)C(CC[C@]([C@@](CC[C@@]3(C(CO)=O)O)([H])[C@]3(C)C4)([H])[C@]2([H])C4=O)=C1",
    "OC[C@@H](C(OC)=O)NC(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "C[C@@]12CC[C@]3([H])[C@]4(OO2)[C@@](O[C@H](O)[C@H](C)C4CC[C@H]3C)([H])O1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument(
        "--label_component",
        action="append",
        help="Which reaction components to consider as 'labels'. Should be either base or sulfonyl_fluoride.",
    )
    parser.add_argument(
        "--adversarial",
        # action="append",
        help="Whether to perform adversarial controls. Should either be desc or fp or onehot.",
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
        "--plr",
        action="store_true",
        help="Include logistic regression that optimizes upon precision@k."
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


def load_data(feature_type, output_type, label_component):
    """Prepares dataset for different algorithm types, feature inputs and labels.

    Parameters
    ----------
    feature_type : str {'onehot', 'desc', 'fp'}
        Input format.
        When fp, it is only for the substrate - we use descriptors for base and sulfonyl fluorides.

    output_type : str {'yield', 'ranking'}
        Type of algorithm the dataset will be used for.

    label_component : str {'sulfonyl_fluoride', 'base', 'both'}
        Considered only when output_type=='ranking'.
        Which component will be used as labels, which are subject to ranking.

    Returns
    -------
    X, y : np.2darray and 1darray of shape (n_samples, n_features) and (n_samples,)
        Arrays ready to be used for algorithms.
    """
    if feature_type == "desc":
        raw_descriptors = pd.read_csv(
            "datasets/deoxyfluorination/descriptor_table.csv"
        ).to_numpy()
        n_substrate_desc = 19
        n_base_desc = 1
    elif feature_type == "onehot":
        raw_descriptors = pd.read_csv(
            "datasets/deoxyfluorination/descriptor_table-OHE.csv"
        ).to_numpy()
        n_substrate_desc = 32
        n_base_desc = 4
    elif feature_type == "fp":
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        fp_array = np.zeros((len(SUBSTRATE_SMILES), 1024))
        for i, smiles in enumerate(SUBSTRATE_SMILES):
            fp_array[i] = mfpgen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
        raw_descriptors = np.hstack(
            (
                np.repeat(fp_array, 20, axis=0),
                pd.read_csv(
                    "datasets/deoxyfluorination/descriptor_table.csv"
                ).to_numpy()[:, 19:],
            )
        )
        n_substrate_desc = 1024
        n_base_desc = 1

    raw_yields = (
        pd.read_csv("datasets/deoxyfluorination/observed_yields.csv", header=None)
        .to_numpy()
        .flatten()
    )

    if output_type == "ranking":
        if label_component == "both":
            X = raw_descriptors[[20 * x for x in range(32)], :n_substrate_desc]
            y = yield_to_ranking(raw_yields.reshape(32, 20))
        elif label_component == "sulfonyl_fluoride":
            X = np.hstack(
                (
                    np.repeat(
                        raw_descriptors[[20 * x for x in range(32)], :n_substrate_desc],
                        4,
                        axis=0,
                    ),  # 4 = number of bases
                    np.tile(
                        raw_descriptors[
                            [5 * x for x in range(4)],
                            n_substrate_desc : n_substrate_desc + n_base_desc,
                        ],
                        (32, 1),
                    ),  # base descriptors
                )
            )
            y = yield_to_ranking(raw_yields.reshape(32 * 4, 5))
        elif label_component == "base":
            X = np.hstack(
                (
                    np.repeat(
                        raw_descriptors[[20 * x for x in range(32)], :n_substrate_desc],
                        5,
                        axis=0,
                    ),  # 5 = number of sulfonyl_fluoride
                    np.tile(
                        raw_descriptors[
                            [x for x in range(5)], n_substrate_desc + n_base_desc :
                        ],
                        (32, 1),
                    ),
                )
            )
            y = yield_to_ranking(raw_yields.reshape(32 * 5, 4))

    elif output_type == "yield":
        X = raw_descriptors
        y = raw_yields
    print("X array shape:", X.shape, "y array shape", y.shape)
    return X, y


def update_perf_dict(perf_dict, kt, rr, mrr, comp, model):
    if type(kt) != list:
        perf_dict["kendall_tau"].append(kt)
        perf_dict["reciprocal_rank"].append(rr)
        perf_dict["mean_reciprocal_rank"].append(mrr)
        perf_dict["test_compound"].append(comp)
        perf_dict["model"].append(model)
    elif type(kt) == list:
        assert len(kt) == len(rr)
        assert len(rr) == len(mrr)
        perf_dict["kendall_tau"].extend(kt)
        perf_dict["reciprocal_rank"].extend(rr)
        perf_dict["mean_reciprocal_rank"].extend(mrr)
        if type(comp) == list:
            perf_dict["test_compound"].extend(comp)
        elif type(comp) == int:
            perf_dict["test_compound"].extend([comp] * len(kt))
        if type(model) == "list":
            perf_dict["model"].extend(model)
        elif type(model) == str:
            perf_dict["model"].extend([model] * len(kt))


def evaluate_lr_alg(test_rank, pred_rank, n_rxns, perf_dict, comp, model):
    if label_component == "both":
        kt = kendalltau(test_rank, pred_rank).statistic
        predicted_highest_yield_inds = np.argpartition(pred_rank.flatten(), n_rxns)[
            :n_rxns
        ]
        rr = 1 / np.min(test_rank[predicted_highest_yield_inds])
        mrr = np.mean(
            np.reciprocal(test_rank[predicted_highest_yield_inds], dtype=np.float32)
        )
    else:
        kt = [
            kendalltau(test_rank[i, :], pred_rank[i, :]).statistic
            for i in range(test_rank.shape[0])
        ]
        rr = [1 / test_rank[a, x] for a, x in enumerate(np.argmin(pred_rank, axis=1))]
        mrr = rr
    print(f"Test compound {comp}")
    print(f"    kendall-tau={kt} // reciprocal_rank={rr} // mrr={mrr}")
    print()

    update_perf_dict(PERFORMANCE_DICT, kt, rr, mrr, comp, model)


def rfr_eval(
    feature_type,
    label_component,
    n_rxns,
    params={"n_estimators": [30, 100, 200], "max_depth": [5, 10, None]},
    random_state=42,
):
    X, y = load_data(feature_type, "yield", label_component)
    # print(y.shape)
    # For now let's use whole dataset. TODO divide datasets so that the non-label-component values are fixed.
    test_fold = np.repeat(np.arange(32), 20)
    inner_test_fold = np.repeat(np.arange(31), 20)
    ps = PredefinedSplit(test_fold)
    print("Evaluating RandomForestRegressor....")
    for i, (train_ind, test_ind) in enumerate(ps.split()):
        X_train, X_test = X[train_ind, :], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        gcv = GridSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_grid=params,
            scoring="r2",
            n_jobs=-1,
            cv=PredefinedSplit(inner_test_fold),
        )
        gcv.fit(X_train, y_train)
        y_pred = gcv.best_estimator_.predict(X_test)
        if label_component == "both":
            # print(f"    Test compound {i} - RMSE={round(mean_squared_error(y_test, y_pred, squared=False), 1)}, R2={round(r2_score(y_test, y_pred), 2)}")
            y_ranking = yield_to_ranking(y_test)
            kt = kendalltau(y_ranking, yield_to_ranking(y_pred).flatten()).statistic
            largest_yield_inds = np.argpartition(-1 * y_pred, n_rxns)[:n_rxns]
            reciprocal_rank = 1 / np.min(y_ranking[largest_yield_inds])
            mean_reciprocal_rank = np.mean(
                np.reciprocal(y_ranking[largest_yield_inds], dtype=np.float32)
            )
        else:
            if label_component == "base":
                y_pred = yield_to_ranking(np.reshape(y_pred, (4, 5)).T)
                y_ranking = yield_to_ranking(np.reshape(y_test, (4, 5)).T)
            elif label_component == "sulfonyl_fluoride":
                y_pred = yield_to_ranking(np.reshape(y_pred, (4, 5)))
                y_ranking = yield_to_ranking(np.reshape(y_test, (4, 5)))
            # print("Y_ranking", y_ranking)
            kt = [
                kendalltau(y_ranking[i, :], y_pred[i, :]).statistic
                for i in range(y_pred.shape[0])
            ]
            reciprocal_rank = [
                1 / y_ranking[a, x] for a, x in enumerate(np.argmin(y_pred, axis=1))
            ]
            mean_reciprocal_rank = reciprocal_rank

        update_perf_dict(
            PERFORMANCE_DICT, kt, reciprocal_rank, mean_reciprocal_rank, i, "rfr"
        )
    print()


def lr_eval(feature_type, label_component, parser, n_rxns):
    X, y = load_data(feature_type, "ranking", label_component)
    if parser.baseline:
        _, y_yields = load_data(feature_type, "yield", label_component)
        y_yields = y_yields.reshape(y.shape)

    test_fold = np.repeat(np.arange(32), int(y.shape[0] // 32))
    # print("RPC Test fold shape:", test_fold.shape)
    ps = PredefinedSplit(test_fold)
    for i, (train_ind, test_ind) in enumerate(ps.split()):
        X_train, X_test = X[train_ind, :], X[test_ind, :]
        rank_train, test_rank = y[train_ind, :], y[test_ind, :]
        if label_component == "both":
            test_rank = test_rank.flatten()
        # print("Test rank shape:", test_rank.shape)

        if parser.baseline:
            pred_rank = yield_to_ranking(np.mean(y_yields[train_ind, :], axis=0))
            # print("Actual rank:", test_rank)
            if label_component == "both":
                evaluate_lr_alg(
                    test_rank, pred_rank, n_rxns, PERFORMANCE_DICT, i, "baseline"
                )
            elif label_component == "base":
                evaluate_lr_alg(
                    test_rank,
                    np.tile(pred_rank, (5, 1)),
                    n_rxns,
                    PERFORMANCE_DICT,
                    i,
                    "baseline",
                )
            elif label_component == "sulfonyl_fluoride":
                evaluate_lr_alg(
                    test_rank,
                    np.tile(pred_rank, (4, 1)),
                    n_rxns,
                    PERFORMANCE_DICT,
                    i,
                    "baseline",
                )

        if parser.rpc or parser.boost_rpc or parser.ibm or parser.ibpl:
            std = StandardScaler()
            train_X_std = std.fit_transform(X_train)
            test_X_std = std.transform(X_test)
            if parser.rpc:
                rpc_lr = RPC(base_learner=LogisticRegression(C=1), cross_validator=None)
                rpc_lr.fit(train_X_std, rank_train)
                rpc_pred_rank = rpc_lr.predict(test_X_std)
                # print("RPC predicted rank:", rpc_pred_rank)
                # print("Actual rank:", test_rank)
                evaluate_lr_alg(
                    test_rank, rpc_pred_rank, n_rxns, PERFORMANCE_DICT, i, "RPC"
                )
            if parser.boost_rpc:
                boost_rpc = BoostLR(
                    RPC(base_learner=LogisticRegression(C=1), cross_validator=None)
                )
                boost_rpc.fit(train_X_std, rank_train)
                boost_rpc_pred_rank = boost_rpc.predict(test_X_std)
                evaluate_lr_alg(
                    test_rank,
                    boost_rpc_pred_rank,
                    n_rxns,
                    PERFORMANCE_DICT,
                    i,
                    "BoostRPC",
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

        if parser.boost_lrt:
            boost_lrt = BoostLR(
                DecisionTreeLabelRanker(min_samples_split=rank_train.shape[1] * 2)
            )
            boost_lrt.fit(X_train, rank_train)
            blrt_pred_rank = boost_lrt.predict(X_test)
            evaluate_lr_alg(
                test_rank, blrt_pred_rank, n_rxns, PERFORMANCE_DICT, i, "BoostLRT"
            )


def precision_lr_eval(
        feature_type,
        label_component,
        n_rxns,
    ):
    if feature_type == "desc" :
        n_subs_desc = 19
    elif feature_type == "onehot" :
        n_subs_desc = 32
    elif feature_type == "fp" :
        n_subs_desc = 1024

    if label_component == "both" :
        X, y = load_data(feature_type, "yield", label_component)
        X_subs, _ = load_data(feature_type, "ranking", label_component)
        X_cond = X[:20, n_subs_desc:]
        y_yields = y.reshape(32, 20)
    else :
        X_subs, y = load_data(feature_type, "ranking", label_component)
        components = ["sulfonyl_fluoride", "base"]
        other_comp = components[1 - components.index(label_component)]
        X_cond, _ = load_data(feature_type, "ranking", other_comp)
        X_cond = X_cond[:int(20//int(X_subs.shape[0]//32)), n_subs_desc:]
        y_yields = 100 - y
        
    X_cond_std = StandardScaler().fit_transform(X_cond)

    # Transforming the y_yields array such that the n_rxns number of highest yielding reactions become positive labels.
    highest_conds = np.argpartition(-1*y_yields, n_rxns)[:, :n_rxns]
    y_multilabel = np.zeros_like(y_yields)
    for i, col in enumerate(highest_conds) :
        y_multilabel[i, col] = 1
        assert sum(y_multilabel[i,:]) == n_rxns

    # Evaluation
    test_fold = np.repeat(np.arange(32), int(y_yields.shape[0] // 32))
    ps = PredefinedSplit(test_fold)
    for i, (train_ind, test_ind) in enumerate(ps.split()):
        X_train_subs, X_test_subs = X_subs[train_ind, :], X_subs[test_ind, :]
        y_ml_train = y_multilabel[train_ind]
        std = StandardScaler()
        X_train_subs_std = std.fit_transform(X_train_subs)
        X_test_subs_std = std.transform(X_test_subs)
        # Remove single valued columns
        cols_to_keep = []
        for j in range(X_train_subs_std.shape[1]) :
            if len(np.unique(X_train_subs_std[:,j])) > 1 :
                cols_to_keep.append(j)
        X_train_subs_std = X_train_subs_std[:, cols_to_keep]
        X_test_subs_std = X_test_subs_std[:, cols_to_keep]

        grc = GridSearchCV(
            estimator=MLPClassifier(solver="lbfgs", random_state=42+i, max_iter=10000),
            param_grid = {
                "hidden_layer_sizes":[
                    (10), (20,)   #for both, was 30
                ],
                "activation":["logistic","relu"],  # "tanh",
                "alpha":[0.001,0.003,0.01] #[0.0001,0.0003,0.001] # 
            },
            n_jobs=-1
        )
        grc.fit(X_train_subs_std, y_ml_train)
        pred_ranking = yield_to_ranking(grc.predict_proba(X_test_subs_std))

        
        test_ranking = yield_to_ranking(y_yields[test_ind, :])
        if label_component == "both":
            test_ranking = test_ranking.flatten()
        evaluate_lr_alg(
            test_ranking,
            pred_ranking,
            n_rxns,
            PERFORMANCE_DICT,
            i,
            "PrecisionOpt",
        )
    

if __name__ == "__main__":
    parser = parse_args()
    if len(parser.label_component) == 2:
        label_component = "both"
        n_rxns = 4
    elif len(parser.label_component) == 1:
        label_component = parser.label_component[0]
        n_rxns = 1

    # Training label ranking algorithms
    if (
        parser.lrrf
        or parser.rpc
        or parser.ibm
        or parser.ibpl
        or parser.lrt
        or parser.baseline
    ):
        print("Evaluating label rankers.")
        lr_eval(parser.adversarial, label_component, parser, n_rxns)

    # Training a random forest regressor
    if parser.rfr:
        rfr_eval(parser.adversarial, label_component, n_rxns)

    # Training a precision optimizer
    if parser.plr :
        precision_lr_eval(
            parser.adversarial,
            label_component,
            n_rxns,
        )
        print(PERFORMANCE_DICT)
    
    # Saving the results
    if parser.save:
        joblib.dump(
            PERFORMANCE_DICT,
            f"performance_excels/deoxy_performance_dict_{label_component}_{parser.adversarial}_precision_nofeature.joblib",
        )
