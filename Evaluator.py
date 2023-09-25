from scipy.stats import kendalltau
import numpy as np
from Dataloader import yield_to_ranking
from abc import ABC, abstractmethod

PERFORMANCE_DICT = {
    "kendall_tau": [],
    "reciprocal_rank": [],
    "mean_reciprocal_rank": [],
    "regret":[],
    "test_compound": [],
    "model": [],
}


class Evaluator(ABC) :
    """ Base class for evaluators for various types of algorithsm.
    
    Parameters
    ----------
    dataset : Dataset object from Dataloader.py
        Dataset to utilize.
    n_rxns : int
        Number of reactions that is simulated to be selected and conducted.
    outer_cv : sklearn split object
        Cross-validation scheme to 'evaluate' the algorithms.
    """
    def __init__(
        self,
        dataset,
        n_rxns,
        outer_cv
    ):
        self.dataset = dataset
        self.n_rxns = n_rxns
        self.outer_cv = outer_cv
        self.perf_dict = PERFORMANCE_DICT


    def _update_perf_dict(perf_dict, kt, rr, mrr, regret, comp, model):
        """ Updates the dictionary keeping track of performances.
        
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


    def _evaluate_alg(self, test_yield, pred_rank, comp, model):
        """ Evaluates algorithms on the following metrics 
        • 'reciprocal rank' : of the selected n_rxns, the reciprocal of the best ground-truth-rank.
        • 'mean reciprocal rank' : mean of the reciprocal of ground-truth-ranks of all rxns selected by model.
        • 'regret' : the numerical difference between the ground-truth highest yield and
                     the best yield retrieved by model.
        and updates the performance dictionary.

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
        if np.ndim(pred_rank) == 1 :
            kt = kendalltau(test_rank, pred_rank).statistic
            predicted_highest_yield_inds = np.argpartition(pred_rank.flatten(), self.n_rxns)[:self.n_rxns]
            best_retrieved_yield = np.max(test_yield[predicted_highest_yield_inds])
            actual_inds_with_that_yield = np.where(test_yield == best_retrieved_yield)[0]
            rr = 1 / np.min(test_rank[actual_inds_with_that_yield])
            mrr = np.mean(
                np.reciprocal(test_rank[predicted_highest_yield_inds], dtype=np.float32)
            )
            regret = max(test_yield) - max(test_yield[predicted_highest_yield_inds])

        elif np.ndim(pred_rank) == 2 :
            kt = [
                kendalltau(test_rank[i, :], pred_rank[i, :]).statistic
                for i in range(pred_rank.shape[0])
            ]
            predicted_highest_yield_inds = np.argpartition(pred_rank, self.n_rxns, axis=1)[:, :self.n_rxns]
            best_retrieved_yield = [np.max(test_yield[i, row]) for i, row in enumerate(predicted_highest_yield_inds)]
            actual_inds_with_that_yield = [np.where(test_yield == best_y)[0] for best_y in best_retrieved_yield]
            rr = [
                1 / np.min(test_rank[a, x]) for a, x in enumerate(actual_inds_with_that_yield)
            ]
            mrr = [
                np.mean(np.reciprocal(test_rank[a, row])) for a, row in enumerate(predicted_highest_yield_inds)
            ]
            raw_regret = np.max(test_yield, axis=1) - np.max(np.vstack(tuple([test_yield[i, row] for i, row in enumerate(predicted_highest_yield_inds)])), axis=1)
            regret = list(raw_regret.flatten())
        self._update_perf_dict(self.perf_dict, kt, rr, mrr, regret, comp, model)

    @abstractmethod
    def train_and_evaluate_models(self) :
        pass


class BaselineEvaluator(Evaluator):
    """ Evaluates the baseline model of selecting based on average yield in the training dataset.
    
    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    n_rxns : int
        Number of reactions that is simulated to be conducted.
    """
    def __init__(
        self,
        dataset,
        n_rxns,
        outer_cv
    ) :
        super().__init__(dataset, n_rxns, outer_cv)

    def train_and_evaluate_models(self):  
        y = self.dataset.y_yield
        for i, (train_ind, test_ind) in enumerate(self.outer_cv.split(())) :
            if type(y) == list :
                self.perf_dict = [PERFORMANCE_DICT] * len(y)
                for j, array in enumerate(y) :
                    y_train, y_test = array[train_ind], array[test_ind]    

            else :
                y_train, y_test = y[train_ind], y[test_ind]
                self._evaluate_alg(
                    y_test,
                    np.tile(yield_to_ranking(np.mean(y_train, axis=0)), (y_test.shape[0], 1)),
                    i,
                    "baseline",
                )
        return self


class RegressorEvaluator(Evaluator):
    """ Evaluates regressors for a specific dataset.
    
    Parameters
    ----------
    dataset : Dataset object as in Dataloader.py
        Dataset to utilize.
    feature_type : str {'desc', 'fp', 'onehot', 'random'}
        Which representation to use as inputs.
    n_rxns : int
        Number of reactions that is simulated to be conducted.
    list_of_algorithms : list of GridSearchCV objects
        Regressors to train.
    outer_cv : sklearn split object
        Cross-validation scheme to 'evaluate' the algorithms.
    """
    def __init__(
        self,
        regressor_dataset,
        feature_type,
        n_rxns,
        list_of_algorithms,
        list_of_names,
        outer_cv
    ) :
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
        if self.feature_type == "desc" :
            X = self.regressor_dataset.X_desc
        elif self.feature_type == "fp" :
            X = self.regressor_dataset.X_fp
        elif self.feature_type == "onehot":
            X = self.regressor_dataset.X_onehot
        elif self.feature_type == "random":
            X = self.regressor_dataset.X_random
        y = self.regressor_dataset.y_yield

        self.models = [[] for _ in range(len(self.list_of_algorithms))]
        self.cv_scores = [[] for _ in range(len(self.list_of_algorithms))]

        for i, (train_ind, test_ind) in enumerate(self.outer_cv.split(())) :
            print(f"Evaluating on compound {i}")
            X_train, X_test = X[train_ind, :], X[test_ind, :]
            y_train, y_test = y[train_ind], y[test_ind]

            for j, (model, model_name) in enumerate(zip(self.list_of_algorithms, self.list_of_names)) :
                model.fit(X_train, y_train)
                best_model = model.best_estimator_
                print(f"    CV score: {round(model.best_score_, 3)}")
                self.models[j].append(best_model)
                self.cv_scores[j].append(model.best_score_)
                y_pred = best_model.predict(X_test)

                self._evaluate_alg(
                    y_test, 
                    yield_to_ranking(y_pred), 
                    i, 
                    model_name
                )
        return self