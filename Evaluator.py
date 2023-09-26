from copy import deepcopy
from scipy.stats import kendalltau
import numpy as np
from Dataloader import yield_to_ranking
from abc import ABC, abstractmethod
from label_ranking import *
from sklearn.linear_model import LogisticRegression
from sklr.tree import DecisionTreeLabelRanker

PERFORMANCE_DICT = {
    "kendall_tau": [],
    "reciprocal_rank": [],
    "mean_reciprocal_rank": [],
    "regret": [],
    "test_compound": [],
    "model": [],
}


class Evaluator(ABC):
    """Base class for evaluators for various types of algorithsm.

    Parameters
    ----------
    dataset : Dataset object from Dataloader.py
        Dataset to utilize.
    n_rxns : int
        Number of reactions that is simulated to be selected and conducted.
    outer_cv : sklearn split object or list of them
        Cross-validation scheme to 'evaluate' the algorithms.
    """

    def __init__(self, dataset, n_rxns, outer_cv):
        self.dataset = dataset
        self.n_rxns = n_rxns
        self.outer_cv = outer_cv

    def _update_perf_dict(self, perf_dict, kt, rr, mrr, regret, comp, model):
        """Updates the dictionary keeping track of performances.

        Parameters
        ----------
        perf_dict : dictionary
            Has 6 keys as  you can see below.
        kt : list or float
            Kendall-tau ranking correlation scores.
            This being a list comes from the case when multiple rows are evaluated simultaneously.
        rr : list or float
            Reciprocal (ground-truth) rank of the best selection by the model.
        mrr : list or float
            Mean reciprocal (ground-truth) rank of all selections by the model.
        regret : list or float
            Ground-truth best yield - ground-truth yield of best selection by model.
        comp : int
            Compound index of the test compound.
        model : str
            Name of the model

        Returns
        -------
        None
            The perf_dict input is updated in-place.
        """
        if type(kt) != list:
            perf_dict["kendall_tau"].append(kt)
            perf_dict["reciprocal_rank"].append(rr)
            perf_dict["mean_reciprocal_rank"].append(mrr)
            perf_dict["regret"].append(regret)
            perf_dict["test_compound"].append(comp)
            perf_dict["model"].append(model)
        elif type(kt) == list:
            assert len(kt) == len(rr)
            assert len(rr) == len(mrr)
            perf_dict["kendall_tau"].extend(kt)
            perf_dict["reciprocal_rank"].extend(rr)
            perf_dict["mean_reciprocal_rank"].extend(mrr)
            perf_dict["regret"].extend(regret)
            if type(comp) == list:
                perf_dict["test_compound"].extend(comp)
            elif type(comp) == int:
                perf_dict["test_compound"].extend([comp] * len(kt))
            if type(model) == "list":
                perf_dict["model"].extend(model)
            elif type(model) == str:
                perf_dict["model"].extend([model] * len(kt))

    def _evaluate_alg(self, perf_dict, test_yield, pred_rank, comp, model):
        """Evaluates algorithms on the following metrics
        • 'reciprocal rank' : of the selected n_rxns, the reciprocal of the best ground-truth-rank.
        • 'mean reciprocal rank' : mean of the reciprocal of ground-truth-ranks of all rxns selected by model.
        • 'regret' : the numerical difference between the ground-truth highest yield and
                     the best yield retrieved by model.
        and updates the performance dictionary.
        If the yield array from which predicted_rank is derived from has ties at the top values, that means
        under the constraint that we can choose only one reaction we randomly choose one of them.
        For convenience we select the one that shows up the earliest in the vector.

        Parameters
        ----------
        test_yield : np.ndarray of shape (n_test_rxns, ) or (n_test_substrates, n_rxn_conditions)
            Array of numerical yields from the test set.
        pred_rank : np.ndarray of shape (n_test_substrates, n_rxn_conditions)
            Array of predicted ranks under each reaction condition
        comp : int
            Index of test compound.
        model : str
            Name of the algorithm that is evaluated.

        Returns
        -------
        self (with updated perf_dict)
        """
        test_rank = yield_to_ranking(test_yield)
        if np.ndim(pred_rank) == 1:
            kt = kendalltau(test_rank, pred_rank).statistic
            predicted_highest_yield_inds = np.argpartition(
                pred_rank.flatten(), self.n_rxns
            )[: self.n_rxns]
            best_retrieved_yield = np.max(test_yield[predicted_highest_yield_inds])
            actual_inds_with_that_yield = np.where(test_yield == best_retrieved_yield)[
                0
            ]
            rr = 1 / np.min(test_rank[actual_inds_with_that_yield])
            mrr = np.mean(
                np.reciprocal(test_rank[predicted_highest_yield_inds], dtype=np.float32)
            )
            regret = max(test_yield) - max(test_yield[predicted_highest_yield_inds])

        elif np.ndim(pred_rank) == 2:
            kt = [
                kendalltau(test_rank[i, :], pred_rank[i, :]).statistic
                for i in range(pred_rank.shape[0])
            ]
            predicted_highest_yield_inds = np.argpartition(
                pred_rank, self.n_rxns, axis=1
            )[:, : self.n_rxns]
            best_retrieved_yield = [
                np.max(test_yield[i, row])
                for i, row in enumerate(predicted_highest_yield_inds)
            ]
            actual_inds_with_that_yield = [
                np.where(test_yield[i, :] == best_y)[0]
                for i, best_y in enumerate(best_retrieved_yield)
            ]
            rr = [
                1 / np.min(test_rank[a, x])
                for a, x in enumerate(actual_inds_with_that_yield)
            ]
            mrr = [
                np.mean(np.reciprocal(test_rank[a, row]))
                for a, row in enumerate(predicted_highest_yield_inds)
            ]
            raw_regret = np.max(test_yield, axis=1) - np.max(
                np.vstack(
                    tuple(
                        [
                            test_yield[i, row]
                            for i, row in enumerate(predicted_highest_yield_inds)
                        ]
                    )
                ),
                axis=1,
            )
            regret = list(raw_regret.flatten())
        print(rr)
        self._update_perf_dict(perf_dict, kt, rr, mrr, regret, comp, model)

    @abstractmethod
    def train_and_evaluate_models(self):
        pass

    def _CV_loops(self, perf_dict, cv, X, y, further_action_before_logging, y_yield=None):
        """ 
        Implements multiple CV loops that are utilized across all algorithms.
        
        Parameters
        ----------
        perf_dict : dict 
            Dictionary to keep track of performance measurements.
        cv : sklearn split object
            Cross-validation scheme to execute.
        X : np.ndarray of shape (n_reactions, n_features)
            Input features.
        y : np.ndarray of shape (n_reactions, ) or (n_substrates, n_reaction_conditions)
            Array the model needs to train upon!
            For label rankers or classifiers, this should not be yields!
        further_action_before_logging : function
            Parameters: trained model, X_test, test_YIELD_ARRAY
        y_yield : np.ndarray of shape (n_reactions, )
            For label rankers and classifiers that does not use raw yields.
        
        Returns
        -------
        self
            perf_dict is updated in-place.
        """
        for a, (train_ind, test_ind) in enumerate(cv.split()):
            # for j, array in enumerate(y) :
            y_train, y_test = y[train_ind], y[test_ind]
            if X is not None :
                X_train, X_test = X[train_ind, :], X[test_ind]
                # print("X TRAIN SHAPE", X_train.shape)
            else :
                X_test = y_train
            if y_yield is not None :
                y_yield_test = y_yield[test_ind]
                # print("Y TEST SHAPE", y_yield_test.shape)
            print("Compound", a)
            for b, (model, model_name) in enumerate(
                zip(self.list_of_algorithms, self.list_of_names)
            ):
                if X is not None :
                    model.fit(X_train, y_train)
                if y_yield is None : 
                    processed_y_test, processed_pred_rank = further_action_before_logging(model, X_test, y_test)
                else : 
                    processed_y_test, processed_pred_rank = further_action_before_logging(model, X_test, y_yield_test)
                self._evaluate_alg(
                    perf_dict,
                    processed_y_test,
                    processed_pred_rank,
                    a,
                    model_name,
                )
        return self


class BaselineEvaluator(Evaluator):
    """Evaluates the baseline model of selecting based on average yield in the training dataset.

    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    n_rxns : int
        Number of reactions that is simulated to be conducted.

    Attributes
    ----------
    perf_dict : dict or list of dicts
        Records of model performances, measured by rr, mrr, regret and kendall tau, over each test compound.
    """
    def __init__(self, dataset, n_rxns, outer_cv):
        super().__init__(dataset, n_rxns, outer_cv)
        self.list_of_algorithms = ["Baseline"]
        self.list_of_names = ["Baseline"]

    def _processing_before_logging(self, model, y_train, y_test):
        processed_pred_rank = np.tile(
            yield_to_ranking(np.mean(y_train, axis=0)),
            (y_test.shape[0], 1),
        )
        return y_test, processed_pred_rank

    def train_and_evaluate_models(self):
        y = self.dataset.y_yield
        print("LENGTH Y", len(y))
        if type(self.outer_cv) == list:
            print("LIST CV")
            self.perf_dict = []
            for i, (array, cv) in enumerate(zip(y, self.outer_cv)):
                # print(i, array[:3, :])
                print(cv)
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(perf_dict, cv, None, array, self._processing_before_logging)
                self.perf_dict.append(perf_dict)
        else :
            self.perf_dict = PERFORMANCE_DICT
            self._CV_loops(self.perf_dict, self.outer_cv, None, y, self._processing_before_logging)
        return self

class MulticlassEvaluator(Evaluator):
    """ Evaluates multiclass classifiers.
    
    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    feature_type : str {'desc', 'fp', 'onehot', 'random'}
        Which representation to use as inputs.
    n_rxns : int
        Number of reactions that is simulated to be conducted.
    outer_cv : sklearn split object
        Cross-validation scheme to 'evaluate' the algorithms.

    Attributes
    ----------
    
    """

class LabelRankingEvaluator(Evaluator):
    """Evaluates multiple label ranking algorithms.

    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    feature_type : str {'desc', 'fp', 'onehot', 'random'}
        Which representation to use as inputs.
    n_rxns : int
        Number of reactions that is simulated to be conducted.
    outer_cv : sklearn split object
        Cross-validation scheme to 'evaluate' the algorithms.
    list_of_names : list of str {'RPC', 'LRT', 'LRRF', 'IBM', 'IBPL'}
        List of label ranking algorithm names.

    Attributes
    ----------
    perf_dict : dict or list of dicts
        Records of model performances, measured by rr, mrr, regret and kendall tau, over each test compound.

    """
    def __init__(
        self,
        dataset,
        feature_type,
        n_rxns,
        list_of_names,
        outer_cv,
    ):
        super().__init__(dataset, n_rxns, outer_cv)
        self.feature_type = feature_type
        self.list_of_names = list_of_names

        self.list_of_algorithms = []
        for name in self.list_of_names :
            if name == "RPC":
                self.list_of_algorithms.append(
                    RPC(base_learner=LogisticRegression(C=1))
                )
            elif name == "LRT" :
                self.list_of_algorithms.append(
                    DecisionTreeLabelRanker(
                        random_state=42, min_samples_split=4 * 2 # might need to change
                    )
                )
            elif name == "LRRF":
                self.list_of_algorithms.append(
                    LabelRankingRandomForest(
                        n_estimators=50
                    )
                )
            elif name == "IBM":
                self.list_of_algorithms.append(
                    IBLR_M(n_neighbors=3, metric="euclidean")
                )
            elif name == "IBPL":
                self.list_of_algorithms.append(
                    IBLR_PL(n_neighbors=3, metric="euclidean")
                )


    def _processing_before_logging(self, model, X_test, y_yield_test):
        if self.dataset.component_to_rank == "base":
            if self.dataset.train_together : 
                y_test_reshape = y_yield_test.reshape(4, 5).T
                pred_rank_reshape = model.predict(X_test).reshape(4, 5).T
            else : 
                y_test_reshape = y_yield_test.flatten()
                pred_rank_reshape = model.predict(X_test).flatten()
        elif self.dataset.component_to_rank == "sulfonyl_fluoride":
            if self.dataset.train_together : 
                y_test_reshape = y_yield_test.reshape(4, 5)
                pred_rank_reshape = model.predict(X_test).reshape(4, 5)
            else : 
                y_test_reshape = y_yield_test.flatten()
                pred_rank_reshape = model.predict(X_test).flatten()
        elif self.dataset.component_to_rank == "both":
            y_test_reshape = y_yield_test
            pred_rank_reshape = model.predict(X_test)
        return y_test_reshape, pred_rank_reshape


    def train_and_evaluate_models(self):
        if self.feature_type == "desc":
            X = self.dataset.X_desc
        elif self.feature_type == "fp":
            X = self.dataset.X_fp
        elif self.feature_type == "onehot":
            X = self.dataset.X_onehot
        elif self.feature_type == "random":
            X = self.dataset.X_random
        y = self.dataset.y_ranking
        y_yield = self.dataset.y_yield
        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but separating the datasets
            self.perf_dict = []
            for i, (X_array, array, yield_array, cv) in enumerate(zip(X, y, y_yield, self.outer_cv)): # X is not included as it remains the same across different reagents
                perf_dict = deepcopy(PERFORMANCE_DICT)
                print("X array shape:", X_array.shape)
                # print("ARRAY SHAPE:", array.shape)
                self._CV_loops(perf_dict, cv, X_array, array, self._processing_before_logging, y_yield=yield_array)
                self.perf_dict.append(perf_dict)
        else : 
            perf_dict = deepcopy(PERFORMANCE_DICT)
            self._CV_loops(perf_dict, self.outer_cv, X, y, self._processing_before_logging, y_yield=y_yield)
            self.perf_dict = perf_dict
        return self

class RegressorEvaluator(Evaluator):
    """Evaluates regressors for a specific dataset.

    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    feature_type : str {'desc', 'fp', 'onehot', 'random'}
        Which representation to use as inputs.
    n_rxns : int
        Number of reactions that is simulated to be conducted.
    outer_cv : sklearn split object
        Cross-validation scheme to 'evaluate' the algorithms.
    list_of_algorithms : list of GridSearchCV objects
        Regressors to train.

    Attributes
    ----------
    perf_dict : dict or list of dicts
        Records of model performances, measured by rr, mrr, regret and kendall tau, over each test compound.
    models : list of regressors
        Models resulting from each CV split.
    cv_scores : list of floats
        Training-CV scores measured in R2 from each test CV split.
    """

    def __init__(
        self,
        regressor_dataset,
        feature_type,
        n_rxns,
        list_of_algorithms,
        list_of_names,
        outer_cv,
    ):
        super().__init__(regressor_dataset, n_rxns, outer_cv)
        self.feature_type = feature_type
        self.list_of_algorithms = list_of_algorithms
        self.list_of_names = list_of_names

    def train_and_evaluate_models(self):
        """
        Trains models.

        Parameters
        ----------
        """
        if self.feature_type == "desc":
            X = self.dataset.X_desc
        elif self.feature_type == "fp":
            X = self.dataset.X_fp
        elif self.feature_type == "onehot":
            X = self.dataset.X_onehot
        elif self.feature_type == "random":
            X = self.dataset.X_random
        y = self.dataset.y_yield

        self.models = [[] for _ in range(len(self.list_of_algorithms))]
        self.cv_scores = [[] for _ in range(len(self.list_of_algorithms))]

        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but separating the datasets
            print("LIST CV", type(X))
            self.perf_dict = []
            for i, (array, cv) in enumerate(zip(y, self.outer_cv)):
                perf_dict = deepcopy(PERFORMANCE_DICT)
                print("X array shape:", X.shape)
                print("ARRAY SHAPE:", array.shape)
                print(cv)
                for j, (train_ind, test_ind) in enumerate(cv.split()):
                    X_train, X_test = X[train_ind, :], X[test_ind]
                    y_train, y_test = array[train_ind], array[test_ind]
                    for k, (model, model_name) in enumerate(
                        zip(self.list_of_algorithms, self.list_of_names)
                    ):
                        model.fit(X_train, y_train)
                        self.models[k].append(model.best_estimator_)
                        self.cv_scores[k].append(model.best_score_)
                        self._evaluate_alg(
                            perf_dict,
                            y_test,
                            yield_to_ranking(model.predict(X_test)),
                            j,
                            model_name,
                        )
                self.perf_dict.append(perf_dict)
            print(len(self.perf_dict))

        else:  # cases when both reaction components are ranked + one component but training together
            self.perf_dict = PERFORMANCE_DICT
            for i, (train_ind, test_ind) in enumerate(self.outer_cv.split()):
                # for j, array in enumerate(y) :
                X_train, X_test = X[train_ind, :], X[test_ind]
                y_train, y_test = y[train_ind], y[test_ind]
                print("Compound", i)
                for k, (model, model_name) in enumerate(
                    zip(self.list_of_algorithms, self.list_of_names)
                ):
                    model.fit(X_train, y_train)
                    self.models[k].append(model.best_estimator_)
                    self.cv_scores[k].append(model.best_score_)
                    if self.dataset.component_to_rank == "base":
                        y_test_reshape = y_test.reshape(4, 5).T
                        pred_rank_reshape = yield_to_ranking(
                            model.predict(X_test).reshape(4, 5).T
                        )
                    elif self.dataset.component_to_rank == "sulfonyl_fluoride":
                        y_test_reshape = y_test.reshape(4, 5)
                        pred_rank_reshape = yield_to_ranking(
                            model.predict(X_test).reshape(4, 5)
                        )
                    elif self.dataset.component_to_rank == "both":
                        y_test_reshape = y_test
                        pred_rank_reshape = yield_to_ranking(model.predict(X_test))
                    self._evaluate_alg(
                        self.perf_dict,
                        y_test_reshape,
                        pred_rank_reshape,
                        i,
                        model_name,
                    )

        return self
