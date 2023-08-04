import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr
from label_ranking import *
from rank_aggregation import *
from sklr.tree import DecisionTreeLabelRanker
from dataset_utils import *
from tqdm import tqdm
from time import time
import argparse
import os


# N_ITERS = 10

### Will consider as substrates
ARYL_HALIDE_DESC = pd.read_csv(
    "datasets/doyle_data/aryl_halide_DFT.csv"
)  # check SI of original paper to remove highly correlated descriptors
ADDITIVE_DESC = pd.read_csv("datasets/doyle_data/additive_DFT.csv")
### Consider as conditions
LIGAND_DESC = pd.read_csv("datasets/doyle_data/ligand_DFT.csv")
BASE_DESC = pd.read_csv("datasets/doyle_data/base_DFT.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument(
        "--rfr", action="store_true", help="Include Random Forest Regressor."
    )
    parser.add_argument(
        "--lrrf", action="store_true", help="Include Label Ranking RF as in Qiu, 2018"
    )
    parser.add_argument(
        "--lrt", action="store_true", help="Include Label Ranking Tree as in Hüllermeier, 2008"
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
        action="append",
        help="Include baseline models - use one of the keywords avg_yield, modal or borda aggregated."
    )
    parser.add_argument(
        "-n", "--n_iter",
        default=1,
        type=int,
        help="Number of random initializations to run."
    )
    parser.add_argument(
        "-s", "--save",
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


def prep_full_doyle_df():
    """Prepares dataframe for the Doyle dataset.
    Replace those with any nan yield values to 0%.
    Removes reactions that involve an isoxazole without given DFT descriptors.
    Outputs an excel pivot table of the organized dataset.

    Parameters
    ----------
    None

    Returns
    -------
    org_data : pd.DataFrame
        pivoted dataframe with halide+additive combinations as index, ligand+base combinations as columns.
    """
    raw_data = pd.read_csv(
        "datasets/doyle_data/Doyle_raw_data.csv",
        usecols=["aryl_halide", "ligand", "base", "additive", "yield"],
    )
    org_data = pd.pivot_table(
        raw_data,
        values="yield",
        index=["aryl_halide", "additive"],
        columns=["ligand", "base"],
    )
    # Putting 0% yield as results for those missing results
    org_data.fillna(0, inplace=True)

    # Removing additive without DFT descriptors
    isox_names = ADDITIVE_DESC["name"].unique()
    no_desc_isox = [
        x for x in org_data.index.unique(level=1).tolist() if x not in isox_names
    ]
    org_data.drop(no_desc_isox, level=1, axis=0, inplace=True)
    org_data.to_excel("organized_doyle.xlsx")

    return org_data


def split_dataset(
    normal_df,
    label_ranking_df,
    n_halides_to_leave_out,
    n_additive_to_leave_out,
    list_halide_names,
    list_additive_names,
    random_state=42,
):
    """Separates out a test set such that a specific number of reactants are leaved out.
    This test set is then separated into three parts - halide OOS, isoxazole OOS, both reactants OOS.

    Parameters
    ----------
    normal_df : pd.DataFrame
        Df where each row corresponds to a single reaction.

    label_ranking_df : pd.DataFrame
        Df where reactants are rows, each reaction condition is a column.

    n_reactants_to_leave_out : int
        Number of reactants to leave out as test set.

    list_halide_names, list_additive_names : list of str
        Names of each reactants.

    random_state : int
        To seed np.random.seed()

    Returns
    -------
    list_of_split_dfs : list of pd.DataFrames
        for each df in list_of_dfs_to_split, a list of splitted dfs are given in the order of
        [training, halide_OOS, additive_OOS, both_OOS]
    """
    # np.random.seed(random_state)

    def random_choice_n_test_comps(list_of_names, n_reactants_to_leave_out):
        inds = np.random.choice(
            len(list_of_names), n_reactants_to_leave_out, replace=False
        )
        return [list_of_names[x] for x in inds]

    test_halide_names = random_choice_n_test_comps(
        list_halide_names, n_halides_to_leave_out
    )
    test_additive_names = random_choice_n_test_comps(
        list_additive_names, n_additive_to_leave_out
    )

    # Splitting df for label ranking
    both_OOS_lr_df = label_ranking_df[
        (label_ranking_df.index.isin(test_halide_names, level=0))
        & (label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    halide_OOS_lr_df = label_ranking_df[
        (label_ranking_df.index.isin(test_halide_names, level=0))
        & (~label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    additive_OOS_lr_df = label_ranking_df[
        (~label_ranking_df.index.isin(test_halide_names, level=0))
        & (label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    training_lr_df = label_ranking_df[
        (~label_ranking_df.index.isin(test_halide_names, level=0))
        & (~label_ranking_df.index.isin(test_additive_names, level=1))
    ]

    # Splitting the normal df
    both_OOS_df = normal_df[
        (normal_df["aryl_halide"].isin(test_halide_names))
        & (normal_df["additive"].isin(test_additive_names))
    ]
    halide_OOS_df = normal_df[
        (normal_df["aryl_halide"].isin(test_halide_names))
        & (~normal_df["additive"].isin(test_additive_names))
    ]
    additive_OOS_df = normal_df[
        (~normal_df["aryl_halide"].isin(test_halide_names))
        & (normal_df["additive"].isin(test_additive_names))
    ]
    training_df = normal_df[
        (~normal_df["aryl_halide"].isin(test_halide_names))
        & (~normal_df["additive"].isin(test_additive_names))
    ]
    return (
        test_halide_names,
        test_additive_names,
        [
            [training_df, halide_OOS_df, additive_OOS_df, both_OOS_df],
            [training_lr_df, halide_OOS_lr_df, additive_OOS_lr_df, both_OOS_lr_df],
        ],
    )


def name_ranking_df_to_features(name_df):
    """Converts a ranking dataframe with compound names into corresponding numpy array of descriptor values.
    Do not use for normal problem settings.

    Parameters
    ----------
    name_df :  pd.DataFrame
        Dataframe with reactant names in row, reaction condition components in each column.

    Returns
    -------
    desc_array : np.ndarray of shape (n_samples, n_features)
        Array of descriptor values to train models on.
    ranking_array : np.ndarray of shape (n_samples, n_labels)
        Array of rankings.
    list_of_reactants : list of tuples (str, str)
        Pairs of reactant names for each desc and y array.
    """
    desc_array, list_of_reactants = [], []
    for i, row in name_df.iterrows():
        list_of_desc_arrays = []
        list_of_desc_arrays.append(
            ARYL_HALIDE_DESC[ARYL_HALIDE_DESC["name"] == row.name[0]].values[0][:-1]
        )
        list_of_desc_arrays.append(
            ADDITIVE_DESC[ADDITIVE_DESC["name"] == row.name[1]].values[0][:-1]
        )
        desc_array.append(np.concatenate(tuple(list_of_desc_arrays)))
        list_of_reactants.append(i)
    desc_array = np.vstack(tuple(desc_array))
    ranking_array = yield_to_ranking(name_df.to_numpy())
    # print(ranking_array.shape)
    return desc_array, ranking_array, list_of_reactants


def name_df_to_features(name_df):
    """Converts a dataframe with compound names into corresponding numpy array of descriptor values.
    Do not use for ranking dfs.

    Parameters
    ----------
    name_df : pd.DataFrame
        Dataframe with compound names in each row.

    Returns
    -------
    list_of_desc_arrays : np.ndarray of shape (n_samples, n_features)
        Array of descriptor values to train models on.
    list_of_y_arrays : np.ndarray of shape (n_samples,) if output_type.lower().startswith()=="y" or (n_samples, n_labels), if  output_type.lower().startswith()=="r".
        Array of continuous-value yields or rankings, depending on output_type
    list_of_reactants : list of tuples (str, str)
        Pairs of reactant names for each desc and y array.
    """
    reactant_pairs = pd.unique(
        [
            (hal, add)
            for (hal, add) in list(
                name_df[["aryl_halide", "additive"]].itertuples(index=False, name=None)
            )
        ]
    )
    list_of_desc_arrays, list_of_y_arrays, list_of_reactants = [], [], []

    for reactant in reactant_pairs:
        list_of_reactants.append(reactant)
        sub_df = name_df[
            (name_df["aryl_halide"] == reactant[0])
            & (name_df["additive"] == reactant[1])
        ]
        row_to_full = []
        for i, row in sub_df.iterrows():
            copy_arrays_to_concat = []
            copy_arrays_to_concat.append(
                ARYL_HALIDE_DESC[ARYL_HALIDE_DESC["name"] == row["aryl_halide"]].values[
                    0
                ][:-1]
            )
            copy_arrays_to_concat.append(
                ADDITIVE_DESC[ADDITIVE_DESC["name"] == row["additive"]].values[0][:-1]
            )
            copy_arrays_to_concat.append(
                LIGAND_DESC[LIGAND_DESC["name"] == row["ligand"]].values[0][:-1]
            )
            copy_arrays_to_concat.append(
                BASE_DESC[BASE_DESC["name"] == row["base"]].values[0][:-1]
            )
            row_to_full.append(np.concatenate(tuple(copy_arrays_to_concat)))
        list_of_desc_arrays.append(np.vstack(tuple(row_to_full)))
        list_of_y_arrays.append(sub_df.loc[:, "value"].to_numpy())

    return list_of_desc_arrays, list_of_y_arrays, list_of_reactants


def train_usual_models(
    model_instance, param_dict, train_X, train_y, predefined_split, scoring
):
    """Trains usual models using grid search cross-validation.

    Parameters
    ----------
    model_instance: RandomForestClassifier, RandomForestRegressor and models like these.
        Object of a type of model to train.
    param_dict : dict
        Parameters to screen through with grid search.
    train_X, train_y : np.ndarrays of shape (n_samples, n_features), (n_samples,)
        Arrays to train the model on.
    predefined_split : sklearn predefined_split object.
        Split to conduct Leave-one-halide out CV.
    scoring : str
        Metric to evaluate CV with.

    Returns
    -------
    trained_model
        Model fit on full training data after conducting grid-search CV.
    """
    grid = GridSearchCV(
        model_instance, param_dict, cv=predefined_split, scoring=scoring, n_jobs=-1
    )
    grid.fit(train_X, train_y)
    return grid.best_estimator_


def evaluate_and_update_score_dict(
    score_dict,
    model,
    list_of_test_X,
    list_of_test_y,
    list_test_reactant_pair,
    y_is_ranking,
    train_halides,
    train_additives,
    num_test_reactants,
    model_type,
    list_which_OOS=["halide", "additive", "both"],
    top_k=1,
):
    """
    Evaluates the predictions of model.

    Parameters
    ----------
    score_dict : dict
        Dictionary used to keep track of all predictions.
    model : Regressor, LabelRanker etc.
        Trained model.
    list_of_test_X : list of np.ndarrays of shape (n_samples, n_features) or (n_samples, n_conditions)
        Input array.
    list_of_test_y : list of np.ndarray of shape (n_samples)
        Ground truth array of continuous yields.
    list_test_reactant_pair : list of tuples (halide, additive)
        Reactants used for X and y.
    y_is_ranking : bool
        Whether the output of the model are ranking values.
        Assumes that list_of_test_y has the same type of arrays.
    train_halides, train_additives : tuples of str
        Names of halides / additives in training set.
    metric : str {'kendall', 'reciprocal_rank', 'p_at_k'}
        Metric to evaluate the models on.
    remaining are strings used for updating the score_dict.

    Returns
    -------
    None : score_dict will be updated.
    """
    # Need to evaluate on single substrate pairs.

    for ind, (list_test_X, list_test_y, list_oos) in enumerate(
        zip(list_of_test_X, list_of_test_y, list_test_reactant_pair)
    ):
        for test_X, test_y, oos in zip(list_test_X, list_test_y, list_oos):
            pred_y = model.predict(test_X)
            if not y_is_ranking:
                pred_y = yield_to_ranking(pred_y).flatten()
                ranking_y = yield_to_ranking(test_y)  # actual ranking
            else:
                pred_y = pred_y.flatten()
                ranking_y = test_y
            print()
            print(pred_y)
            print(ranking_y)
            top_rank_retrieved = np.min(ranking_y[np.argpartition(pred_y, kth=top_k)[:top_k]])
            print(ranking_y[np.argpartition(pred_y, kth=top_k)[:top_k]])
            kendall = kendalltau(ranking_y, pred_y).statistic
            # Top rank retrieved when top-k suggested reactions are conducted.

            score_dict["Number of test reactants"].append(num_test_reactants)
            score_dict["Train halides"].append(train_halides)
            score_dict["Train additives"].append(train_additives)
            score_dict["Model"].append(model_type)
            score_dict["OOS"].append(list_which_OOS[ind])
            score_dict["Kendall"].append(kendall)
            score_dict["Top Rank"].append(top_rank_retrieved)
            score_dict["Test halide"].append(oos[0])
            score_dict["Test additive"].append(oos[1])


if __name__ == "__main__":
    parser = parse_args()
    print(parser)
    if not os.path.exists("organized_doyle.xlsx") :
        print("Organized data file does not exist. Preparing...")
        org_doyle_df = prep_full_doyle_df()
    else :
        print("Loading previously organized dataset.")
        org_doyle_df = pd.read_excel("organized_doyle.xlsx")
    print()
    # Below line unpivots the table, for use in normal classification / regression
    cleaned_raw_data = org_doyle_df.reset_index().melt(
        id_vars=["aryl_halide", "additive"]
    )
    halides = org_doyle_df.index.unique(level=0).tolist()  # 15 halides
    additives = org_doyle_df.index.unique(level=1).tolist()  # 22 isoxazoles

    performance_dict = {
        "Number of test reactants": [],
        "Train halides": [],
        "Train additives": [],
        "Model": [],
        "OOS": [],
        "Kendall": [],
        "Top Rank": [],
        "Test halide": [],
        "Test additive": [],
    }
    np.random.seed(42)
    if parser.rfr :
        regressor_performance_dict = deepcopy(performance_dict)
    if parser.lrrf or parser.rpc or parser.ibm or parser.ibpl or parser.lrt :
        rpc_performance_dict = deepcopy(performance_dict)
    if parser.baseline :
        baseline_performance_dict = deepcopy(performance_dict)

    for n_reactants_as_test in range(13, 14):  
        for iter in tqdm(range(parser.n_iter)):
            # Split arrays for Label Ranking algorithms
            test_halide_names, test_additive_names, df_list = split_dataset(
                cleaned_raw_data,
                org_doyle_df,
                n_reactants_as_test,
                n_reactants_as_test + 5,
                halides,
                additives,
                42 + iter,
            )
            train_halide_names = tuple(
                sorted([x for x in halides if x not in test_halide_names])
            )
            train_additive_names = tuple(
                sorted([x for x in additives if x not in test_additive_names])
            )
            print("Training halides:", train_halide_names)
            print("Training additives:", train_additive_names)
            print()
            
            if parser.rfr : 
                # Prepare descriptor arrays
                training_normal_df = df_list[0][0]
                train_X, train_y_cont, train_reactants = name_df_to_features(
                    training_normal_df
                )
                train_X = np.vstack(tuple(train_X))
                train_y_cont = np.concatenate(tuple(train_y_cont))
                print(
                    "Training array shapes:",
                    train_X.shape,
                    train_y_cont.shape,
                    len(train_reactants),
                )

                (
                    list_halide_OOS_X,
                    list_halide_OOS_y_cont,
                    list_OOS_halide_pairs,
                ) = name_df_to_features(df_list[0][1])
                print(
                    "OOS Halide array shapes:",
                    len(list_halide_OOS_X),
                    list_halide_OOS_X[0].shape,
                    len(list_halide_OOS_y_cont),
                    list_halide_OOS_y_cont[0].shape,
                    len(list_OOS_halide_pairs),
                )

                (
                    list_additive_OOS_X,
                    list_additive_OOS_y_cont,
                    list_OOS_additive_pairs,
                ) = name_df_to_features(df_list[0][2])
                print(
                    "OOS Additive array shapes:",
                    len(list_additive_OOS_X),
                    list_additive_OOS_X[0].shape,
                    len(list_additive_OOS_y_cont),
                    list_additive_OOS_y_cont[0].shape,
                    len(list_OOS_additive_pairs),
                )

                list_both_OOS_X, list_both_OOS_y_cont, list_OOS_pairs = name_df_to_features(
                    df_list[0][3]
                )
                print(
                    "Both OOS array shapes:",
                    len(list_both_OOS_X),
                    list_both_OOS_X[0].shape,
                    len(list_both_OOS_y_cont),
                    list_both_OOS_y_cont[0].shape,
                    len(list_OOS_pairs),
                )
                # print()
                # print("Finished preparing all arrays.")
                # print()

                # Prepare one-halide-out CV split
                training_halides = [x for x in halides if x not in test_halide_names]
                test_fold = [
                    training_halides.index(x) for x in training_normal_df["aryl_halide"]
                ]
                assert len(test_fold) == training_normal_df.shape[0]
                ps = PredefinedSplit(test_fold)

                # Training normal regressor
                start = time()
                rfr = train_usual_models(
                    RandomForestRegressor(random_state=42 + iter, max_features="sqrt"),
                    {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 5, None],
                    },
                    train_X,
                    train_y_cont,
                    ps,
                    "r2",
                )

                # # Evaluating normal regressor
                evaluate_and_update_score_dict(
                    regressor_performance_dict,
                    rfr,
                    [list_halide_OOS_X, list_additive_OOS_X, list_both_OOS_X],
                    [
                        list_halide_OOS_y_cont,
                        list_additive_OOS_y_cont,
                        list_both_OOS_y_cont,
                    ],
                    [list_OOS_halide_pairs, list_OOS_additive_pairs, list_OOS_pairs],
                    False,
                    train_halide_names,
                    train_additive_names,
                    n_reactants_as_test,
                    "RandomForestRegressor",
                    list_which_OOS=["halide", "additive", "both"],
                    top_k=2,
                )
                end = time()
                print(
                    f"Finished training and evaluating RFR. Took {round(end-start, 2)} sec."
                )
                if parser.save :
                    regressor_performance_df = pd.DataFrame(regressor_performance_dict)
                    regressor_performance_df.to_excel(f"performance_excels/regressor_performance_{parser.n}.xlsx")
                    
                print("Moving onto label ranking algorithms.")
                print()
                            
            if parser.lrrf or parser.rpc or parser.ibm or parser.ibpl or parser.lrt or parser.baseline :
                # # Preparing arrays for label ranking models.
                training_lr_df = df_list[1][0]
                train_X, train_y_rank, train_reactants = name_ranking_df_to_features(
                    training_lr_df
                )
                train_X = np.vstack(tuple(train_X))
                scaler = StandardScaler()
                train_X_std = scaler.fit_transform(train_X)
                print(
                    "Training array shapes:",
                    train_X.shape,
                    train_y_rank.shape,
                    len(train_reactants),
                )

                (
                    list_halide_OOS_X,
                    list_halide_OOS_y_rank,
                    list_OOS_halide_pairs,
                ) = name_ranking_df_to_features(df_list[1][1])
                list_std_halide_OOS_X = [
                    scaler.transform(x.reshape(1, -1)) for x in list_halide_OOS_X
                ]
                std_halide_OOS_X = np.vstack(tuple(list_std_halide_OOS_X))

                (
                    list_additive_OOS_X,
                    list_additive_OOS_y_rank,
                    list_OOS_additive_pairs,
                ) = name_ranking_df_to_features(df_list[1][2])
                list_std_additive_OOS_X = [
                    scaler.transform(x.reshape(1, -1)) for x in list_additive_OOS_X
                ]
                std_additive_OOS_X = np.vstack(tuple(list_std_additive_OOS_X))

                (
                    list_both_OOS_X,
                    list_both_OOS_y_rank,
                    list_OOS_pairs,
                ) = name_ranking_df_to_features(df_list[1][3])
                list_std_both_OOS_X = [
                    scaler.transform(x.reshape(1, -1)) for x in list_both_OOS_X
                ]
                std_both_OOS_X = np.vstack(tuple(list_std_both_OOS_X))

                if parser.lrrf or parser.rpc or parser.ibm or parser.ibpl or parser.lrt :
                    print()
                    print("Finished preparing all arrays.")
                    label_ranking_models = []
                    label_ranking_names = []
                    if parser.rpc : 
                        print()
                        print("Fitting RPC-LR")
                        rpc_lr = RPC(base_learner=LogisticRegression(C=1), cross_validator=None)
                        rpc_lr.fit(train_X_std, train_y_rank)
                        label_ranking_models.append(rpc_lr)
                        label_ranking_names.append("Pairwise")
                        print("Complete")
                    if parser.lrrf : 
                        print()
                        print("Fitting LRRF")
                        lrrf = LabelRankingRandomForest(n_estimators=200)
                        lrrf.fit(train_X_std, train_y_rank)
                        label_ranking_models.append(lrrf)
                        label_ranking_names.append("LRRF")
                        print("Complete")
                    if parser.lrt : 
                        print()
                        print("Fitting LRT")
                        lrt = DecisionTreeLabelRanker(
                            random_state=42, min_samples_split=train_y_rank.shape[1] * 2
                        )
                        lrt.fit(train_X_std, train_y_rank)
                        label_ranking_models.append(lrt)
                        label_ranking_names.append("LRT")
                        print("Complete")
                    print()
                    for model, name in zip(label_ranking_models, label_ranking_names):
                        evaluate_and_update_score_dict(
                            rpc_performance_dict,
                            model,
                            [
                                list_std_halide_OOS_X,
                                list_std_additive_OOS_X,
                                list_std_both_OOS_X,
                            ],
                            [
                                list_halide_OOS_y_rank,
                                list_additive_OOS_y_rank,
                                list_both_OOS_y_rank,
                            ],
                            [list_OOS_halide_pairs, list_OOS_additive_pairs, list_OOS_pairs],
                            True,
                            train_halide_names,
                            train_additive_names,
                            n_reactants_as_test,
                            name,
                            list_which_OOS=["halide", "additive", "both"],
                            top_k=2,
                        )
                    if parser.save :
                        rpc_performance_df = pd.DataFrame(rpc_performance_dict)
                        rpc_performance_df.to_excel(f"performance_excels/label_ranking_performance_{parser.n}.xlsx")

                if parser.baseline :
                    print("Evaluating baseline models.")
                    baseline_models = []
                    for baseline_type in parser.baseline :
                        if baseline_type == "avg_yield" :
                            avg_yield_base = Baseline()
                            avg_yield_base.fit("", training_lr_df.to_numpy())
                            baseline_models.append(avg_yield_base)
                            print("Top 2 predicted through average yield in training data:", avg_yield_base.predict(np.zeros((1,4)))[0][:2])
                        elif baseline_type == "modal":
                            modal_base = Baseline(criteria="modal")
                            modal_base.fit("", train_y_rank)
                            baseline_models.append(modal_base)
                        elif baseline_type == "borda" :
                            borda_base = Baseline(criteria="borda")
                            borda_base.fit("", train_y_rank)
                            baseline_models.append(borda_base)
                            print("Top 2 predicted through Borda count in training data:", borda_base.predict(np.zeros((1,4)))[0][:2])
                            print()
                        else :
                            print("This is not a supported baseline type. Please use one of avg_yield, modal, borda.")
                            break
                    for model, name in zip(baseline_models, parser.baseline):
                        evaluate_and_update_score_dict(
                            baseline_performance_dict,
                            model,
                            [
                                list_std_halide_OOS_X,
                                list_std_additive_OOS_X,
                                list_std_both_OOS_X,
                            ],
                            [
                                list_halide_OOS_y_rank,
                                list_additive_OOS_y_rank,
                                list_both_OOS_y_rank,
                            ],
                            [list_OOS_halide_pairs, list_OOS_additive_pairs, list_OOS_pairs],
                            True,
                            train_halide_names,
                            train_additive_names,
                            n_reactants_as_test,
                            name,
                            list_which_OOS=["halide", "additive", "both"],
                            top_k=2,
                        )
                    if parser.save :
                        baseline_performance_df = pd.DataFrame(baseline_performance_dict)
                        baseline_performance_df.to_excel(f"performance_excels/baseline_performance_{parser.n_iter}.xlsx")
                        

