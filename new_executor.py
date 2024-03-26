import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold
import os
from dataloader import *
from new_evaluator import *
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
N_EVALS = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Specify the evaluation to run.")
    parser.add_argument(
        "--dataset",
        help="Which dataset to use. Should be either 'deoxy', 'natureHTE', 'scienceMALDI', 'informer', 'ullmann', 'borylation'.",
    )
    parser.add_argument(
        "--feature",
        help="Which feature to use. Should be either 'desc', 'fp', 'onehot', 'random'",
    )
    parser.add_argument(
        "--label_component",
        action="append",
        help="Which reaction components to consider as 'labels'. For the natureHTE and scienceMALDI datasets will use as the substrate to test.",
    )
    parser.add_argument(
        "--train_together",
        action=argparse.BooleanOptionalAction,
        help="Whether the non-label reaction component should be treated altogether or as separate datasets. Is not utilized in the paper.",
    )
    parser.add_argument(
        "--rfr", action="store_true", help="Include Random Forest Regressor."
    )
    parser.add_argument(
        "--lrrf", action="store_true", help="Include Label Ranking RF as in Qiu, 2018"
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
        "--rfc",
        action="store_true",
        help="Include random forest classifier",
    ),
    parser.add_argument("--lr", action="store_true", help="Include logistic regressor"),
    parser.add_argument("--knn", action="store_true", help="Include kNN classifier"),
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline models - avg_yield.",
    )
    parser.add_argument(
        "--n_missing_reaction",
        default=0,
        type=int,
        help="Number of reactions missing from each substrate.",
    )
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="If n_missing_reaction > 0, by specifying this keyword as true triggers the training dataset to be randomly selected substrates with all reaction conditions."
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Whether to save resulting scores in an excel file.",
    )
    args = parser.parse_args()
    return args


def parse_algorithms(parser, inner_ps=None):
    """For label ranking and classifier algorithm, goes through the parser
    to prepare a list of algorithm names to conduct.

    Parameters
    ----------
    parser: argparse object.
    inner_ps : PredefinedSplit object.
        Splits to train the algorithms.

    Returns
    -------
    lr_algorithms, classifiers : list of str
        Name of algorithms in each class to execute.
    """
    # Listing Label Ranking algorithms
    lr_algorithms = []
    if parser.rpc:
        lr_algorithms.append("RPC")
    if parser.lrrf:
        lr_algorithms.append("LRRF")
    if parser.ibm:
        lr_algorithms.append("IBM")
    if parser.ibpl:
        lr_algorithms.append("IBPL")
    # Listing Conventional Classifiers
    classifiers = []
    if parser.rfc:
        classifiers.append("RFC")
    if parser.lr:
        classifiers.append("LR")
    if parser.knn:
        classifiers.append("KNN")
    return lr_algorithms, classifiers


def prepare_stratified_kfold_by_top_condition(X, y_ranking, n_splits):
    """ Prepares stratified kfold for cross validation by splitting by the top reaction condition.
    
    Parameters
    ----------
    dataset : dataloader.Dataset object.
        Dataset to work on.
    n_splits : int
        Number of splits to make.

    Returns
    -------
    fold_array : np.ndarray of shape (n_substrates)
        Which test fold each substrate is alloted to. Used for modifying for regressor.
    outer_ps : PredefinedSplit object
        Predefined split to be used for evaluation.
    """
    # nonzero_rows, top_condition = np.where(y_ranking == 1)
    nonzero_rows, top_condition = [], []
    for i, row in enumerate(y_ranking) :
        if min(row) < y_ranking.shape[1] + 1 :
            nonzero_rows.append(i)
            top_condition.append(np.argmin(row))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    outer_ps_array = -1 * np.ones(X.shape[0])
    for fold, (_, test) in enumerate(skf.split(X[nonzero_rows], top_condition)):
        outer_ps_array[test] = fold
    outer_ps = PredefinedSplit(outer_ps_array)
    return outer_ps_array, outer_ps


def lr_names_to_model_objs(lr_names, inner_ps):
    """Changes list of algorithm names into objects to train.

    Parameters
    ----------
    lr_names : list of str
        Names of label ranking algorithms to consider.
    inner_ps : PredefinedSplit object or int
        How to split dataset for training

    Returns
    -------
    lr_objs : list of GridSearchCV objects.
    """
    convert_dict = {
        # "RPC": GridSearchCV(
        #     RPC(),
        #     param_grid={
        #         "C": [0.1, 0.3, 1, 3, 10], 
        #         "penalty": ["l1", "l2"],
        #     },
        #     scoring=kt_score,
        #     cv=inner_ps,
        #     n_jobs=-1,
        # ),
        "RPC" : GridSearchCV(
            RPC(),
            param_grid={
                "n_estimators":[10,25,50,100],
                "max_depth":[2,4,None]
            },
            scoring=kt_score,
            cv=inner_ps,
            n_jobs=-1,
        ),
        "LRRF": GridSearchCV(
            LabelRankingRandomForest(),
            param_grid={
                "n_estimators": [25, 50, 100],
                "max_depth": [4, 6, 8],
            },  # 25,,100  4,6,
            scoring=kt_score,
            cv=inner_ps,
            n_jobs=-1,
        ),
        "IBM": GridSearchCV(
            IBLR_M(),
            param_grid={"n_neighbors": [3, 5, 10]},  # ,5,10
            scoring=kt_score,
            cv=inner_ps,
            n_jobs=-1,
        ),
        "IBPL": GridSearchCV(
            IBLR_PL(),
            param_grid={"n_neighbors": [3, 5, 10]},  # ,5,10
            scoring=kt_score,
            cv=inner_ps,
            n_jobs=-1,
        ),
    }
    return [convert_dict[x] for x in lr_names]


def run_informer(parser):
    """Runs model evaluations on the informer dataset as defined by the parser.

    Parameters
    ----------
    parser: argparse object.

    Returns
    -------
    perf_dicts : dict
        key : model type
        val : list of (or a single) performance dictionaries
    """
    # Initialization
    label_component = parser.label_component[0]
    if label_component == "amine_ratio":
        n_rxns = 2
        n_other_component = 4
    elif label_component == "catalyst_ratio":
        n_rxns = 4
        n_other_component = 2  # may need to multiply by 5
    if parser.n_missing_reaction > 0:
        n_evals = N_EVALS
    else:
        n_evals = 1

    lr_algorithms, classifiers = parse_algorithms(parser)
    regressor_dataset = InformerDataset(True, label_component, parser.train_together, n_rxns)
    # Evaluations
    perf_dicts = []
    if parser.rfr:
        outer_ps = [PredefinedSplit(np.repeat(np.arange(11), 40/n_other_component))] * n_other_component
        inner_ps = PredefinedSplit(
            np.repeat(np.arange(10), 40/n_other_component - parser.n_missing_reaction)
        )

        evaluator = RegressorEvaluator(
            regressor_dataset,
            parser.feature,
            n_rxns,
            [
                GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid={
                        "n_estimators": [30, 100, 200],
                        "max_depth": [5, 10, None],
                    },
                    scoring="r2",
                    n_jobs=-1,
                    cv=inner_ps,
                )
            ],
            ["RFR"],
            outer_ps,
            parser.n_missing_reaction,
            n_evals,
        ).train_and_evaluate_models()
        perf_dicts.append(evaluator.perf_dict)

    if parser.baseline or len(lr_algorithms) > 0 or len(classifiers) > 0:
        dataset = InformerDataset(False, label_component, parser.train_together, n_rxns)
        ps = [PredefinedSplit(np.arange(11))] * n_other_component
        inner_ps = PredefinedSplit(np.arange(10))

        if parser.baseline:
            baseline_evaluator = BaselineEvaluator(
                dataset, n_rxns, ps, parser.n_missing_reaction, n_evals
            ).train_and_evaluate_models()
            perf_dicts.append(baseline_evaluator.perf_dict)
        if len(lr_algorithms) > 0:
            lr_algorithm_objs = lr_names_to_model_objs(lr_algorithms, inner_ps)
            label_ranking_evaluator = LabelRankingEvaluator(
                dataset,
                parser.feature,
                n_rxns,
                lr_algorithms,
                lr_algorithm_objs,
                ps,
                parser.n_missing_reaction,
                n_evals,
            ).train_and_evaluate_models()
            perf_dicts.append(label_ranking_evaluator.perf_dict)
        if len(classifiers) > 0:
            if n_rxns > 1:
                classifier_evaluator = MultilabelEvaluator(
                    dataset,
                    parser.feature,
                    n_rxns,
                    classifiers,
                    ps,
                    parser.n_missing_reaction,
                    n_evals,
                ).train_and_evaluate_models()
            perf_dicts.append(classifier_evaluator.perf_dict)

    return perf_dicts


def run_evaluation(parser):
    """ Runs model evaluations on datasets except the informer dataset.
    
    Parameters
    ----------
    parser : argparse object.
    
    Returns
    -------
    perf_dicts : dict
        key : model type
        val : val : list of (or a single) performance dictionaries
    """
    # Initialization
    if parser.label_component is not None :
        label_component = parser.label_component[0]
    else :
        label_component = [1] # Necessary for consistency sake
    lr_algorithms, classifiers = parse_algorithms(parser)
    multidataset = False
    if parser.dataset == "deoxy" :
        n_rxns = 1
        n_outer_splits = 5
        multidataset = True
        ranking_dataset, regressor_dataset = DeoxyDataset(False, "base", False, n_rxns), DeoxyDataset(True, "base", False, n_rxns)
    elif parser.dataset == "natureHTE" :
        n_rxns = 1
        n_outer_splits = 5
        ranking_dataset, regressor_dataset = NatureDataset(False, label_component, n_rxns), NatureDataset(True, label_component, n_rxns)
    elif parser.dataset == "scienceMALDI" :
        n_rxns = 1
        n_outer_splits = 4
        ranking_dataset, regressor_dataset = ScienceDataset(False, label_component, n_rxns), ScienceDataset(True, label_component, n_rxns)
    elif parser.dataset == "ullmann" :
        n_rxns = 4
        n_outer_splits = 4
        ranking_dataset, regressor_dataset = UllmannDataset(False, n_rxns), UllmannDataset(True, n_rxns)
    elif parser.dataset == "borylation" :
        n_rxns = 3
        n_outer_splits = 5
        ranking_dataset, regressor_dataset = BorylationDataset(False, n_rxns), BorylationDataset(True, n_rxns)

    if parser.n_missing_reaction > 0:
        n_evals = N_EVALS
    else:
        n_evals = 1
    
    perf_dicts = []
    ## Preparing CV splits
    inner_ps = 4
    if not multidataset :
        # FP only for CV purposes. Feature actually used is determined by parser.feature as in second argument of Evaluator()
        outer_ps_array, outer_ps = prepare_stratified_kfold_by_top_condition(ranking_dataset.X_fp, ranking_dataset.y_ranking, n_outer_splits)
    else :
        outer_ps_array = []
        outer_ps = []
        for X_array, y_array in zip(ranking_dataset.X_fp, ranking_dataset.y_ranking) :
            a, b = prepare_stratified_kfold_by_top_condition(X_array, y_array, n_outer_splits)
            outer_ps_array.append(a)
            outer_ps.append(b)
        

    if parser.rfr :
        if type(outer_ps_array) == np.ndarray :
            rfr_ps = PredefinedSplit(np.repeat(outer_ps_array, ranking_dataset.y_ranking.shape[1]))
        elif type(outer_ps_array) == list :
            rfr_ps = [PredefinedSplit(np.repeat(x, ranking_dataset.n_rank_component)) for x in outer_ps_array]
            
        evaluator = RegressorEvaluator(
            regressor_dataset, 
            parser.feature,
            n_rxns,
            [
                GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid={
                        "n_estimators": [30, 100, 200],
                        "max_depth": [5, 10, None],
                    },
                    scoring="r2",
                    n_jobs=-1,
                    cv=inner_ps,
                )
            ],
            ["RFR"],
            rfr_ps,
            parser.n_missing_reaction,
            n_evals,
            use_all_conditions=parser.all_conditions
        ).train_and_evaluate_models()
        perf_dicts.append(evaluator.perf_dict)
    
    if parser.baseline or len(lr_algorithms) > 0 or len(classifiers) > 0:
        if parser.baseline:
            baseline_evaluator = BaselineEvaluator(
                ranking_dataset, n_rxns, outer_ps, parser.n_missing_reaction, n_evals, use_all_conditions=parser.all_conditions
            )
            baseline_CV = baseline_evaluator.train_and_evaluate_models()
            perf_dicts.append(baseline_CV.perf_dict)

        if len(lr_algorithms) > 0:
            lr_names = deepcopy(lr_algorithms)
            lr_algorithms = lr_names_to_model_objs(lr_algorithms, inner_ps)
            label_ranking_evaluator = LabelRankingEvaluator(
                ranking_dataset,
                parser.feature,
                n_rxns,
                lr_names,
                lr_algorithms,
                outer_ps,
                parser.n_missing_reaction,
                n_evals,
                use_all_conditions=parser.all_conditions
            )
            label_ranking_CV = label_ranking_evaluator.train_and_evaluate_models()
            perf_dicts.append(label_ranking_CV.perf_dict)

        if len(classifiers) > 0:
            if n_rxns == 1 :
                classifier_evaluator = MulticlassEvaluator(
                    ranking_dataset,
                    parser.feature,
                    n_rxns,
                    classifiers,
                    outer_ps,
                    parser.n_missing_reaction,
                    n_evals,
                    use_all_conditions=parser.all_conditions
                ).train_and_evaluate_models()
                classifier_CV = classifier_evaluator.train_and_evaluate_models()
                perf_dicts.append(classifier_CV.perf_dict)
            else :
                classifier_evaluator = MultilabelEvaluator(
                    ranking_dataset,
                    parser.feature,
                    n_rxns,
                    classifiers,
                    outer_ps,
                    parser.n_missing_reaction,
                    n_evals,
                    use_all_conditions=parser.all_conditions
                ).train_and_evaluate_models()
                classifier_CV = classifier_evaluator.train_and_evaluate_models()
                perf_dicts.append(classifier_CV.perf_dict)

    return perf_dicts


def parse_perf_dicts(parser, perf_dicts):
    """
    Process the performance dicts.

    Parameters
    ----------
    save : bool
        Whether the processed performance log should be saved.
        if True : saves an excel file
        if False : prints out average scores
    perf_dicts : list of dictionaries or list of lists of dictionaries

    Returns
    -------
    None
    """

    def print_perf_df(perf_df, model):
        print(
            model,
            round(
                perf_df[perf_df["model"] == model]["reciprocal_rank"].mean(),
                3,
            ),
            round(perf_df[perf_df["model"] == model]["kendall_tau"].mean(), 3),
            round(perf_df[perf_df["model"] == model]["regret"].mean(), 3),
        )

    save = parser.save
    if type(perf_dicts[0]) == list:
        full_perf_df = []
        for i in range(len(perf_dicts[0])):
            sub_perf_dict = pd.concat([pd.DataFrame(x[i]) for x in perf_dicts])
            full_perf_df.append(sub_perf_dict)
        for perf_df in full_perf_df:
            for model in perf_df["model"].unique():
                print_perf_df(perf_df, model)
        full_perf_df = pd.concat(full_perf_df)
    elif type(perf_dicts[0]) == dict:
        full_perf_df = pd.concat([pd.DataFrame(x) for x in perf_dicts])
        for model in full_perf_df["model"].unique():
            print_perf_df(full_perf_df, model)
    if save:
        if not os.path.exists(f"performance_excels/{parser.dataset}"):
            os.mkdir(f"performance_excels/{parser.dataset}")
        if parser.label_component is None :
            comp = "None"
        elif len(parser.label_component) == 1:
            comp = parser.label_component[0]
        elif len(parser.label_component) == 2:
            comp = "both"

        if parser.n_missing_reaction == 0:
            filename = f"performance_excels/{parser.dataset}/{parser.feature}_{comp}_{parser.train_together}.xlsx"
        elif parser.all_conditions:
            filename = f"performance_excels/{parser.dataset}/{parser.feature}_{comp}_{parser.train_together}_rem{parser.n_missing_reaction}rxns_ALLCONDS.xlsx"
        else:
            filename = f"performance_excels/{parser.dataset}/{parser.feature}_{comp}_{parser.train_together}_rem{parser.n_missing_reaction}rxns.xlsx"
        # To append to previously existing file
        if os.path.exists(filename) :
            prev_df = pd.read_excel(filename)
            concat_df = pd.concat([prev_df, full_perf_df])
            concat_df.to_excel(filename)
        else :
            full_perf_df.to_excel(filename)


def main(parser):
    if parser.dataset != "informer":
        perf_dicts = run_evaluation(parser)
    else :
        perf_dicts = run_informer(parser)
    parse_perf_dicts(parser, perf_dicts)


if __name__ == "__main__":
    parser = parse_args()
    main(parser)