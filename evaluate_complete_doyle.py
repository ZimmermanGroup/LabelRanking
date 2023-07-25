import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr
from label_ranking import *
from rank_aggregation import *
from tqdm import tqdm
from time import time
import argparse


N_ITERS = 10

### Will consider as substrates
ARYL_HALIDE_DESC = pd.read_csv("datasets/doyle_data/aryl_halide_DFT.csv") # check SI of original paper to remove highly correlated descriptors
ADDITIVE_DESC = pd.read_csv("datasets/doyle_data/additive_DFT.csv")
### Consider as conditions
LIGAND_DESC = pd.read_csv("datasets/doyle_data/ligand_DFT.csv")
BASE_DESC = pd.read_csv("datasets/doyle_data/base_DFT.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Select the models to evaluate.")
    parser.add_argument("--lrrf", default=False, help="Include Label Ranking RF as in Qiu, 2018")
    parser.add_argument("--rpc", default=False, help="Include Pairwise label ranking as in Hüllermeier, 2008")
    parser.add_argument("--ibm", default=False, help="Include Instance-based label ranking with Mallows model as in Hüllermeier, 2009")
    parser.add_argument("--ibpl", default=False, help="Include Instance-based label ranking with Plackett=Luce model as in Hüllermeier, 2010")
    parser.add_argument("--save", default=True, help="Whether to save resulting scores in an excel file.")
    args = parser.parse_args()
    if args.lrrf is False and args.rpc is False and args.ibm is False and args.ibpl is False :
        parser.error("At least one model is required.")
    return args


def prep_full_doyle_df():
    """ Prepares dataframe for the Doyle dataset.
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
    raw_data = pd.read_csv("datasets/doyle_data/Doyle_raw_data.csv", usecols=["aryl_halide", "ligand", "base", "additive", "yield"])
    org_data = pd.pivot_table(
        raw_data,
        values="yield",
        index=["aryl_halide", "additive"],
        columns=["ligand","base"]
    )
    # Putting 0% yield as results for those missing results
    org_data.fillna(0, inplace=True)
    
    # Removing additive without DFT descriptors
    isox_names = ADDITIVE_DESC["name"].unique()
    no_desc_isox = [x for x in org_data.index.unique(level=1).tolist() if x not in isox_names]
    org_data.drop(no_desc_isox, level=1, axis=0, inplace=True)
    org_data.to_excel("organized_doyle.xlsx")

    return org_data


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
        raw_rank = yield_array.shape[1]-np.argsort(np.argsort(yield_array, axis=1), axis=1)
        for i, row in enumerate(yield_array) : 
            raw_rank[i, np.where(row==0)[0]] = len(row > 0)
        print("Raw rank", raw_rank.shape)
    elif len(yield_array.shape) == 1:
        raw_rank = len(yield_array) - np.argsort(np.argsort(yield_array))
        raw_rank[np.where(raw_rank == 0)[0]] = len(raw_rank)
    return raw_rank


def split_dataset(normal_df, label_ranking_df, n_halides_to_leave_out, n_additive_to_leave_out,
                  list_halide_names, list_additive_names, random_state=42):
    """ Separates out a test set such that a specific number of reactants are leaved out.
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
    np.random.seed(random_state)

    def random_choice_n_test_comps(list_of_names, n_reactants_to_leave_out):
        inds = np.random.choice(len(list_of_names), n_reactants_to_leave_out, replace=False)
        return [list_of_names[x] for x in inds]

    test_halide_names = random_choice_n_test_comps(list_halide_names, n_halides_to_leave_out)
    test_additive_names = random_choice_n_test_comps(list_additive_names, n_additive_to_leave_out)

    # Splitting df for label ranking
    both_OOS_lr_df = label_ranking_df[
        (label_ranking_df.index.isin(test_halide_names, level=0)) &\
        (label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    halide_OOS_lr_df = label_ranking_df[
        (label_ranking_df.index.isin(test_halide_names, level=0)) &\
        (~label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    additive_OOS_lr_df = label_ranking_df[
        (~label_ranking_df.index.isin(test_halide_names, level=0)) &\
        (label_ranking_df.index.isin(test_additive_names, level=1))
    ]
    training_lr_df = label_ranking_df[
        (~label_ranking_df.index.isin(test_halide_names, level=0)) &\
        (~label_ranking_df.index.isin(test_additive_names, level=1))
    ]

    # Splitting the normal df
    both_OOS_df = normal_df[
        (normal_df["aryl_halide"].isin(test_halide_names)) &\
        (normal_df["additive"].isin(test_additive_names))
    ]
    halide_OOS_df = normal_df[
        (normal_df["aryl_halide"].isin(test_halide_names)) &\
        (~normal_df["additive"].isin(test_additive_names))
    ]
    additive_OOS_df = normal_df[
        (~normal_df["aryl_halide"].isin(test_halide_names)) &\
        (normal_df["additive"].isin(test_additive_names))
    ]
    training_df = normal_df[
        (~normal_df["aryl_halide"].isin(test_halide_names)) &\
        (~normal_df["additive"].isin(test_additive_names))
    ]
    return test_halide_names, test_additive_names, [[training_df, halide_OOS_df, additive_OOS_df, both_OOS_df],
            [training_lr_df, halide_OOS_lr_df, additive_OOS_lr_df, both_OOS_lr_df]]


def name_ranking_df_to_features(name_df):
    """ Converts a ranking dataframe with compound names into corresponding numpy array of descriptor values.
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
        list_of_desc_arrays.append(ARYL_HALIDE_DESC[ARYL_HALIDE_DESC["name"]==row.name[0]].values[0][:-1])
        list_of_desc_arrays.append(ADDITIVE_DESC[ADDITIVE_DESC["name"]==row.name[1]].values[0][:-1])
        desc_array.append(np.concatenate(tuple(list_of_desc_arrays)))
        list_of_reactants.append(i)
    desc_array = np.vstack(tuple(desc_array))
    ranking_array = yield_to_ranking(name_df.to_numpy())
    # print(ranking_array.shape)
    return desc_array, ranking_array, list_of_reactants

def name_df_to_features(name_df):
    """ Converts a dataframe with compound names into corresponding numpy array of descriptor values.
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
        [(hal, add) for (hal, add) in list(name_df[["aryl_halide", "additive"]].itertuples(index=False, name=None))]
    )
    list_of_desc_arrays, list_of_y_arrays, list_of_reactants = [], [], []

    for reactant in reactant_pairs :
        list_of_reactants.append(reactant)
        sub_df = name_df[
            (name_df["aryl_halide"]==reactant[0]) &\
            (name_df["additive"]==reactant[1])
        ]
        row_to_full = []
        for i, row in sub_df.iterrows():
            copy_arrays_to_concat = []
            copy_arrays_to_concat.append(ARYL_HALIDE_DESC[ARYL_HALIDE_DESC["name"]==row["aryl_halide"]].values[0][:-1])
            copy_arrays_to_concat.append(ADDITIVE_DESC[ADDITIVE_DESC["name"] == row["additive"]].values[0][:-1])
            copy_arrays_to_concat.append(LIGAND_DESC)[LIGAND_DESC["name"] == row["ligand"]].values[0][:-1]
            copy_arrays_to_concat.append(BASE_DESC[BASE_DESC["name"] == row["base"]].values[0][:-1])
            row_to_full.append(np.concatenate(tuple(copy_arrays_to_concat)))
        list_of_desc_arrays.append(np.vstack(tuple(row_to_full)))
        list_of_y_arrays.append(sub_df.loc[:,"value"].to_numpy())

    return list_of_desc_arrays, list_of_y_arrays, list_of_reactants


def train_usual_models(model_instance, param_dict, train_X, train_y, predefined_split, scoring) :
    """ Trains usual models using grid search cross-validation.
    
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
    grid = GridSearchCV(model_instance, param_dict, cv=predefined_split, scoring=scoring, n_jobs=-1)
    grid.fit(train_X, train_y)
    return grid.best_estimator_


def evaluate_and_update_score_dict(
        score_dict, model, list_of_test_X, list_of_test_y, list_test_reactant_pair,
        y_is_ranking, 
        num_test_reactants, model_type, 
        list_which_OOS=["halide","additive","both"], top_k=1
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
    metric : str {'kendall', 'reciprocal_rank', 'p_at_k'}
        Metric to evaluate the models on.
    remaining are strings used for updating the score_dict.

    Returns
    -------
    None : score_dict will be updated.
    """
    # Need to evaluate on single substrate pairs.

    for ind, (list_test_X, list_test_y, list_oos) in enumerate(zip(list_of_test_X, list_of_test_y, list_test_reactant_pair)):
        for test_X, test_y, oos in zip(list_test_X, list_test_y, list_oos):
            pred_y = model.predict(test_X)
            if not y_is_ranking :
                pred_y = yield_to_ranking(pred_y).flatten()
                ranking_y = yield_to_ranking(test_y) #actual ranking
                top_rank_retrieved = np.min(ranking_y[np.argsort(pred_y)[:top_k]])
            else :
                ranking_y = test_y
                ranks_retrieved = np.argsort(pred_y, axis=1)[:,:top_k] + 1
                top_rank_retrieved = np.min(ranks_retrieved)
            kendall = kendalltau(ranking_y, pred_y).statistic
            # Top rank retrieved when top-k suggested reactions are conducted.
            
            score_dict["Number of test reactants"].append(num_test_reactants)
            score_dict["Model"].append(model_type)
            score_dict["OOS"].append(list_which_OOS[ind])
            score_dict["Kendall"].append(kendall)
            score_dict["Top Rank"].append(top_rank_retrieved)
            score_dict["Test halide"].append(oos[0])
            score_dict["Test additive"].append(oos[1])


if __name__ == "__main__" :
    # parser = parse_args()
    org_doyle_df = prep_full_doyle_df()
    # Below line unpivots the table, for use in normal classification / regression
    cleaned_raw_data = org_doyle_df.reset_index().melt(id_vars=["aryl_halide", "additive"])
    halides = org_doyle_df.index.unique(level=0).tolist() # 15 halides
    additives = org_doyle_df.index.unique(level=1).tolist() # 22 isoxazoles

    performance_dict = {
        "Number of test reactants":[],
        "Model":[],
        "OOS":[],
        "Kendall":[],
        "Top Rank":[],
        "Test halide":[],
        "Test additive":[]
    }
    regressor_performance_dict = deepcopy(performance_dict)
    rpc_performance_dict = deepcopy(performance_dict)
    for n_reactants_as_test in range(13,14) : # need to change 3 to 6 or larger
        for iter in range(1) :
            np.random.seed(42+iter)
            # Split arrays for Label Ranking algorithms
            test_halide_names, test_additive_names, df_list = split_dataset(
                cleaned_raw_data, org_doyle_df, n_reactants_as_test, n_reactants_as_test+5,
                halides, additives, 42+iter
            )
            # # Prepare descriptor arrays
            # training_normal_df = df_list[0][0]
            # train_X, train_y_cont, train_reactants = name_df_to_features(training_normal_df, "yield")
            # train_X = np.vstack(tuple(train_X))
            # train_y_cont = np.concatenate(tuple(train_y_cont))
            # print("Training array shapes:", train_X.shape, train_y_cont.shape, len(train_reactants))
            
            # list_halide_OOS_X, list_halide_OOS_y_cont, list_OOS_halide_pairs = name_df_to_features(df_list[0][1], "yield")
            # print("OOS Halide array shapes:", len(list_halide_OOS_X), list_halide_OOS_X[0].shape, 
            #       len(list_halide_OOS_y_cont), list_halide_OOS_y_cont[0].shape, len(list_OOS_halide_pairs))

            # list_additive_OOS_X, list_additive_OOS_y_cont, list_OOS_additive_pairs = name_df_to_features(df_list[0][2], "yield")
            # print("OOS Additive array shapes:", len(list_additive_OOS_X), list_additive_OOS_X[0].shape, 
            #       len(list_additive_OOS_y_cont), list_additive_OOS_y_cont[0].shape, len(list_OOS_additive_pairs))

            # list_both_OOS_X, list_both_OOS_y_cont, list_OOS_pairs = name_df_to_features(df_list[0][3], "yield")
            # print("Both OOS array shapes:", len(list_both_OOS_X), list_both_OOS_X[0].shape, 
            #       len(list_both_OOS_y_cont), list_both_OOS_y_cont[0].shape, len(list_OOS_pairs))
            # # print()
            # # print("Finished preparing all arrays.")
            # # print()

            # # # Prepare one-halide-out CV split
            # training_halides = [x for x in halides if x not in test_halide_names]
            # test_fold = [training_halides.index(x) for x in training_normal_df["aryl_halide"]]
            # assert len(test_fold) == training_normal_df.shape[0]
            # ps = PredefinedSplit(test_fold)

            # # # Training normal regressor
            # start = time()
            # rfr = train_usual_models(
            #     RandomForestRegressor(random_state=42+iter, max_features="sqrt"), 
            #     {"n_estimators":[50,100,200], "max_depth":[3,5,None],}, 
            #     train_X, train_y_cont, ps, "r2"
            # )
            
            # # # Evaluating normal regressor
            # evaluate_and_update_score_dict(
            #     regressor_performance_dict, rfr, 
            #     [list_halide_OOS_X, list_additive_OOS_X, list_both_OOS_X], 
            #     [list_halide_OOS_y_cont, list_additive_OOS_y_cont, list_both_OOS_y_cont],
            #     [list_OOS_halide_pairs, list_OOS_additive_pairs, list_OOS_pairs],
            #     False, 
            #     n_reactants_as_test, "RandomForestRegressor", 
            #     list_which_OOS=["halide","additive","both"], top_k=2
            # )
            # end = time()
            # print(f"Finished training and evaluating RFR. Took {round(end-start, 2)} sec.")
            # print("Moving onto label ranking algorithms.")
            # print()
            
            # regressor_performance_df = pd.DataFrame(regressor_performance_dict)
            # regressor_performance_df.to_excel("regressor_performance.xlsx")
            
            # # Preparing arrays for label ranking models.
            training_lr_df = df_list[1][0]
            train_X, train_y_rank, train_reactants = name_ranking_df_to_features(training_lr_df)
            train_X = np.vstack(tuple(train_X))
            scaler = StandardScaler()
            train_X_std = scaler.fit_transform(train_X)
            print("Training array shapes:", train_X.shape, train_y_rank.shape, len(train_reactants))
            
            list_halide_OOS_X, list_halide_OOS_y_rank, list_OOS_halide_pairs = name_ranking_df_to_features(df_list[1][1])
            list_std_halide_OOS_X = [scaler.transform(x.reshape(1,-1)) for x in list_halide_OOS_X]
            std_halide_OOS_X = np.vstack(tuple(list_std_halide_OOS_X))
            # halide_OOS_y_ranking = np.vstack(tuple(list_halide_OOS_y_rank))
            # print("OOS Halide array shapes:", std_halide_OOS_X.shape, halide_OOS_y_ranking.shape, len(list_OOS_halide_pairs))
            
            list_additive_OOS_X, list_additive_OOS_y_rank, list_OOS_additive_pairs = name_ranking_df_to_features(df_list[1][2])
            list_std_additive_OOS_X = [scaler.transform(x.reshape(1,-1)) for x in list_additive_OOS_X]
            std_additive_OOS_X = np.vstack(tuple(list_std_additive_OOS_X))
            # additive_OOS_y_ranking = np.vstack(tuple(list_additive_OOS_y_rank))
            # print("OOS Additive array shapes:",std_additive_OOS_X.shape, additive_OOS_y_ranking.shape, len(list_OOS_additive_pairs))

            list_both_OOS_X, list_both_OOS_y_rank, list_OOS_pairs = name_ranking_df_to_features(df_list[1][3])
            list_std_both_OOS_X = [scaler.transform(x.reshape(1,-1)) for x in list_both_OOS_X]
            std_both_OOS_X = np.vstack(tuple(list_std_both_OOS_X))
            # both_OOS_y_ranking = np.vstack(tuple(list_both_OOS_y_rank))
            # print("Both OOS array shapes:", std_both_OOS_X.shape, both_OOS_y_ranking.shape, len(list_OOS_pairs))

            print()
            print("Finished preparing all arrays.")
            print()
            # Instance-based Mallows Model
            start = time()
            print("Fitting IBLR-M")
            ibm = IBLR_M(n_neighbors=5, metric="euclidean")
            ibm.fit(train_X_std, train_y_rank)
            print("Complete")
            print()
            print("Fitting IBLR-PL")
            ibpl = IBLR_PL(n_neighbors=5, metric="euclidean")
            ibpl.fit(train_X_std, train_y_rank)
            print("Complete")
            print()
            print("Fitting RPC-LR")
            rpc_lr = RPC(
                base_learner=LogisticRegression(C=1),
                cross_validator=None
                # cross_validator=GridSearchCV(
                #     estimator=LogisticRegression(solver="liblinear"),
                #     param_grid = {
                #         "C":[0.03,0.1,0.3,1,3,10,30],
                #         "penalty":["l1","l2"]
                #     },
                #     cv=2,
                #     n_jobs=-1
                # )
            )
            rpc_lr.fit(train_X_std, train_y_rank)
            print("Complete")
            print()
            print("Fitting LBRF")
            lbrf = LabelRankingRandomForest(n_estimators=200)
            lbrf.fit(train_X_std, train_y_rank)
            
            for model, name in zip([rpc_lr, lbrf], ["Pairwise", "LBRF"]): #ibm, ibpl,  "InstanceBased-Mallows", "InstanceBased-PlackettLuce", 
                evaluate_and_update_score_dict(
                    rpc_performance_dict, model, 
                    [list_std_halide_OOS_X, list_std_additive_OOS_X, list_std_both_OOS_X], 
                    [list_halide_OOS_y_rank, list_additive_OOS_y_rank, list_both_OOS_y_rank],
                    [list_OOS_halide_pairs, list_OOS_additive_pairs, list_OOS_pairs],
                    True, 
                    n_reactants_as_test, name, 
                    list_which_OOS=["halide","additive","both"], top_k=2
                )
            # end = time()
            # print(f"Finished training and evaluating IB-M with GridSearchCV. Took {round(end-start, 2)} sec.")
            rpc_performance_df = pd.DataFrame(rpc_performance_dict)
            rpc_performance_df.to_excel("rpc_performance.xlsx")
            
            # Training normal classifier
            # rfc = train_usual_models(
            #     RandomForestClassifier(random_state=42+iter), 
            #     {"n_estimators":[50,100,200], "max_depth":[3,5,None],}, 
            #     train_X, train_y, ps, "roc_auc_score"
            # )


