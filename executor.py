import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklr.tree import DecisionTreeLabelRanker

from dataloader import *
from evaluator import *
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N_EVALS = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Specify the evaluation to run.")
    parser.add_argument(
        "--dataset",
        help="Which dataset to use. Should be either 'deoxy', 'natureHTE', 'informer'.",
    )
    parser.add_argument(
        "--feature",
        help="Which feature to use. Should be either 'desc', 'fp', 'onehot', 'random'",
    )
    parser.add_argument(
        "--label_component",
        action="append",
        help="Which reaction components to consider as 'labels'. For the natureHTE dataset will use as the substrate to test.",
    )
    parser.add_argument(
        "--train_together",
        action=argparse.BooleanOptionalAction,
        help="Whether the non-label reaction component should be treated altogether or as separate datasets. Is not utilized in the natureHTE datset.",
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
    if parser.lrt:
        lr_algorithms.append("LRT")
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


def lr_names_to_model_objs(lr_names, inner_ps):
    """ Changes list of algorithm names into objects to train.
    
    Parameters
    ----------
    lr_names : list of str
        Names of label ranking algorithms to consider.
    
    Returns
    -------
    lr_objs : list of GridSearchCV objects.
    """
    convert_dict = {
        "RPC":GridSearchCV(
                RPC(),
                param_grid = {
                    "C":[0.1,0.3,1,3,10], #0.1,0.3,,3,10
                    "penalty":["l1","l2"] #"l1",
                },
                scoring=kt_score,
                cv=inner_ps,
                n_jobs=-1
            ),
        "LRT":DecisionTreeLabelRanker(
                random_state=42, min_samples_split=4 * 2  # might need to change
            ),
        "LRRF":GridSearchCV(
                LabelRankingRandomForest(),
                param_grid = {"n_estimators":[25,50,100], "max_depth":[4,6,8]}, #25,,100  4,6,
                scoring=kt_score,
                cv=inner_ps,
                n_jobs=-1
            ),
        "IBM":GridSearchCV(
                IBLR_M(),
                param_grid={"n_neighbors":[3,5,10]}, #,5,10
                scoring=kt_score,
                cv=inner_ps,
                n_jobs=-1
            ),
        "IBPL":GridSearchCV(
                IBLR_PL(),
                param_grid={"n_neighbors":[3,5,10]}, #,5,10
                scoring=kt_score,
                cv=inner_ps,
                n_jobs=-1
            )
    }
    return [convert_dict[x] for x in lr_names]


def run_nature(parser):
    """Runs model evaluations on the natureHTE dataset as defined by the parser.

    Parameters
    ----------
    parser: argparse object.

    Returns
    -------
    perf_dicts : idct
        key : model type
        val : list of (or a single) performance dictionaries
    """
    # Initialization
    # This dataset has only four reaction conditions.
    # So uses the label_component of parser as the dataset to test upon.
    label_component = parser.label_component[0]
    lr_algorithms, classifiers = parse_algorithms(parser)
    n_rxns = 1  # Since there are only 4 reaction condition candidates
    if parser.n_missing_reaction > 0 :
        n_evals = N_EVALS
    else :
        n_evals = 1
    # Evaluations
    perf_dicts = []
    if parser.rfr:
        dataset = NatureDataset(True, label_component, n_rxns)
        onehot_array = dataset.X_onehot
        # print("ONEHOT SHAPE", dataset.X_onehot.shape[0])
        outer_ps = PredefinedSplit(
            np.repeat(np.arange(int(onehot_array.shape[0] // 4)), 4)
        )
        inner_ps = PredefinedSplit(
            np.repeat(np.arange(int(onehot_array.shape[0] // 4 - 1)), 4 - parser.n_missing_reaction)
        )
        evaluator = RegressorEvaluator(
            dataset,
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
            n_evals
        ).train_and_evaluate_models()
        # regressor_validation = evaluator.external_validation()
        perf_dicts.append(evaluator.perf_dict)
        # perf_dicts.append(regressor_validation.perf_dict)
    if parser.baseline or len(lr_algorithms) > 0 or len(classifiers) > 0:
        dataset = NatureDataset(False, label_component, n_rxns)
        onehot_array = dataset.X_onehot
        ps = PredefinedSplit(np.arange(onehot_array.shape[0]))
        if parser.baseline:
            baseline_evaluator = BaselineEvaluator(dataset, n_rxns, ps, parser.n_missing_reaction, n_evals)
            baseline_CV = baseline_evaluator.train_and_evaluate_models()
            # baseline_validation = baseline_evaluator.external_validation()
            perf_dicts.append(baseline_CV.perf_dict)
            # perf_dicts.append(baseline_validation.valid_dict)
        if len(lr_algorithms) > 0:
            inner_ps = PredefinedSplit(np.arange(onehot_array.shape[0]-1))
            lr_names = deepcopy(lr_algorithms)
            lr_algorithms = lr_names_to_model_objs(lr_algorithms, inner_ps)
            label_ranking_evaluator = LabelRankingEvaluator(
                dataset,
                parser.feature,
                n_rxns,
                lr_names,
                lr_algorithms,
                ps,
                parser.n_missing_reaction,
                n_evals
            )
            label_ranking_CV = label_ranking_evaluator.train_and_evaluate_models()
            # label_ranking_validation = label_ranking_evaluator.external_validation()
            perf_dicts.append(label_ranking_CV.perf_dict)
            # perf_dicts.append(label_ranking_validation.valid_dict)
        if len(classifiers) > 0:
            classifier_evaluator = MulticlassEvaluator(
                    dataset, 
                    parser.feature, 
                    n_rxns, 
                    classifiers, 
                    ps,
                    parser.n_missing_reaction,
                    n_evals
                ).train_and_evaluate_models()
            classifier_CV = classifier_evaluator.train_and_evaluate_models()
            # classifier_validation = classifier_evaluator.external_validation()
            perf_dicts.append(classifier_CV.perf_dict)
            # perf_dicts.append(classifier_validation.valid_dict)
    return perf_dicts


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
    if parser.n_missing_reaction > 0 :
        n_evals = N_EVALS
    else :
        n_evals = 1

    lr_algorithms, classifiers = parse_algorithms(parser)
    # Evaluations
    perf_dicts = []
    if parser.rfr:
        if parser.train_together:
            outer_ps = PredefinedSplit(np.repeat(np.arange(11), 40))
            inner_ps = PredefinedSplit(np.repeat(np.arange(10), 40 - parser.n_missing_reaction))

        elif label_component == "amine_ratio":
            outer_ps = [PredefinedSplit(np.repeat(np.arange(11), 10))] * 4
            inner_ps = PredefinedSplit(np.repeat(np.arange(10), 10 - parser.n_missing_reaction))

        elif label_component == "catalyst_ratio":
            outer_ps = [PredefinedSplit(np.repeat(np.arange(11), 20))] * 2
            inner_ps = PredefinedSplit(np.repeat(np.arange(10), 20 - parser.n_missing_reaction))

        evaluator = RegressorEvaluator(
            InformerDataset(True, label_component, parser.train_together, n_rxns),
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
            n_evals
        ).train_and_evaluate_models()
        perf_dicts.append(evaluator.perf_dict)

    if parser.baseline or len(lr_algorithms) > 0 or len(classifiers) > 0:
        dataset = InformerDataset(False, label_component, parser.train_together, n_rxns)
        if parser.train_together:
            ps = PredefinedSplit(np.repeat(np.arange(11), n_other_component))
        else:
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
                n_evals
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
                    n_evals
                ).train_and_evaluate_models()
            perf_dicts.append(classifier_evaluator.perf_dict)

    return perf_dicts


def run_deoxy(parser):
    """Runs model evaluations on the deoxyfluorination dataset as defined by the parser.

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
    if len(parser.label_component) == 2:
        n_rxns = 4
        label_component = "both"
    else:
        n_rxns = 1
        label_component = parser.label_component[0]
        if label_component == "sulfonyl_fluoride":
            n_other_component = 4
        elif label_component == "base":
            n_other_component = 5
    lr_algorithms, classifiers = parse_algorithms(parser)
    if parser.n_missing_reaction > 0 :
        n_evals = N_EVALS
    else :
        n_evals = 1

    # Evaluations
    perf_dicts = []
    if parser.rfr:
        if parser.train_together or label_component == "both":
            print("Training together or ranking all conditions")
            inner_ps = PredefinedSplit(np.repeat(np.arange(31), 20 - parser.n_missing_reaction))
            outer_ps = PredefinedSplit(np.repeat(np.arange(32), 20))
        elif label_component == "base" and not parser.train_together:
            print("Not training together, ranking bases")
            inner_ps = PredefinedSplit(np.repeat(np.arange(31), 4 - parser.n_missing_reaction))
            outer_ps = [PredefinedSplit(np.repeat(np.arange(32), 4))] * 5
        elif label_component == "sulfonyl_fluoride" and not parser.train_together:
            inner_ps = PredefinedSplit(np.repeat(np.arange(31), 5 - parser.n_missing_reaction))
            outer_ps = [PredefinedSplit(np.repeat(np.arange(32), 5))] * 4

        evaluator = RegressorEvaluator(
            DeoxyDataset(True, label_component, parser.train_together, n_rxns),
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
            n_evals
        ).train_and_evaluate_models()
        perf_dicts.append(evaluator.perf_dict)

    if parser.baseline or len(lr_algorithms) > 0 or len(classifiers) > 0:
        dataset = DeoxyDataset(False, label_component, parser.train_together, n_rxns)
        if parser.train_together:
            if label_component != "both":
                ps = PredefinedSplit(np.repeat(np.arange(32), n_other_component))
            else:
                ps = PredefinedSplit(np.arange(32))
        else:
            ps = [PredefinedSplit(np.arange(32))] * n_other_component
            inner_ps = PredefinedSplit(np.arange(31))
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
                n_evals
            ).train_and_evaluate_models()
            perf_dicts.append(label_ranking_evaluator.perf_dict)
        if len(classifiers) > 0:
            if n_rxns == 1:
                classifier_evaluator = MulticlassEvaluator(
                    dataset, 
                    parser.feature, 
                    n_rxns, 
                    classifiers, 
                    ps,
                    parser.n_missing_reaction,
                    n_evals
                ).train_and_evaluate_models()
            perf_dicts.append(classifier_evaluator.perf_dict)
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
        if len(parser.label_component) == 1:
            comp = parser.label_component[0]
        elif len(parser.label_component) == 2:
            comp = "both"
        if parser.n_missing_reaction == 0 :
            full_perf_df.to_excel(
                f"performance_excels/{parser.dataset}/{parser.feature}_{comp}_{parser.train_together}.xlsx"
            )
        else : 
            full_perf_df.to_excel(
                f"performance_excels/{parser.dataset}/{parser.feature}_{comp}_{parser.train_together}_rem{parser.n_missing_reaction}rxns.xlsx"
            )



def main(parser):
    if parser.dataset == "deoxy":
        perf_dicts = run_deoxy(parser)
    elif parser.dataset == "informer":
        perf_dicts = run_informer(parser)
    elif parser.dataset == "natureHTE":
        perf_dicts = run_nature(parser)
    parse_perf_dicts(parser, perf_dicts)


if __name__ == "__main__":
    parser = parse_args()
    main(parser)
