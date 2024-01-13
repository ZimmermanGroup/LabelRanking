from copy import deepcopy
from scipy.stats import kendalltau
import numpy as np
import dataloader
from dataloader import yield_to_ranking
from abc import ABC, abstractmethod
from label_ranking import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PERFORMANCE_DICT = {
    "kendall_tau": [],
    "reciprocal_rank": [],
    "mean_reciprocal_rank": [],
    "regret": [],
    "test_compound": [],
    "model": [],
}
np.random.seed(42)


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
    n_rxns_to_erase : int, default=0
        Number of reaction conditions to erase from each substrate pair.
    n_evaluations : int, default=1
        Number of evaluations to conduct.
    """

    def __init__(self, dataset, n_rxns, outer_cv, n_rxns_to_erase=0, n_evaluations=1):
        self.dataset = dataset
        self.n_rxns = n_rxns
        self.outer_cv = outer_cv
        self.n_rxns_to_erase = n_rxns_to_erase
        self.n_evaluations = n_evaluations

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
            else:
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
        print(rr, regret)
        self._update_perf_dict(perf_dict, kt, rr, mrr, regret, comp, model)

    def _load_X(self):
        """Prepares the input features based on the specified type."""
        if self.feature_type == "desc":
            X = self.dataset.X_desc
        elif self.feature_type == "fp":
            X = self.dataset.X_fp
        elif self.feature_type == "onehot":
            X = self.dataset.X_onehot
        elif self.feature_type == "random":
            X = self.dataset.X_random
        return X

    def _dist_array_train_test_split(self, dist_array, test_ind):
        """
        Splits a precomputed distance array into those only with training compounds and test compounds.
        We assume that only one test compound is left out.

        Parameters
        ----------
        dist_array : np.ndarray of shape (n_substrates, n_substrates)
            Precomputed Tanimoto distance array.
        test_ind : int [0, dist_array.shape[0]-1]
            Index of the test compound to separate out from the dist_array.

        Returns
        -------
        train_dists : np.ndarray of shape (n_substrates-1, n_substrates-1)
            Distances between the training compounds.
        test_dists : np.ndarray of shape (n_substrates-1)
            Distance of test compound to all other training compounds.
        """
        train_dists = np.vstack(
            tuple(
                [row for ind, row in enumerate(dist_array) if ind not in test_ind]
            )  # used to be != instead of not in
        )
        train_dists = train_dists[
            :, [x for x in range(train_dists.shape[1]) if x not in test_ind]
        ]
        print("DIST ARRAY SHAPE", dist_array.shape)
        if len(test_ind) == 1:
            test_dists = dist_array[
                test_ind, [x for x in range(dist_array.shape[1]) if x != test_ind]
            ]
        else:
            test_dists = np.vstack(
                tuple(
                    [
                        dist_array[
                            x,
                            [
                                a
                                for a in range(dist_array.shape[1])
                                if a not in test_ind
                            ],
                        ]
                        for x in test_ind
                    ]
                )
            )
            print("DIST SHAPES", train_dists.shape, test_dists.shape)
        return train_dists, test_dists

    @abstractmethod
    def train_and_evaluate_models(self):
        pass

    # TODO  NEED TO SOMEHOW MOVE UNDER __INIT__ SO THAT IT IS REPRODUCIBLE EACH TIME
    def _random_selection_of_inds_to_erase(self, n_training_substrates):
        """Prepares an array of random numbers between 0 and 1 to be used to
        erase the same reactions across regression and label ranking models.

        Parameters
        ----------
        n_training_substrates : int
            Number of training substrates to prepare for.
        
        Returns
        -------
        list_of_inds_to_cover : list of tuples
            array of row indices and array of column indices to remove.
        """
        list_of_inds_to_cover = []
        shape = (n_training_substrates, self.dataset.n_rank_component)
        for i in range(self.n_evaluations):
            random_numbers = np.random.rand(shape[0], shape[1])
            list_of_inds_to_cover.append(
                (
                    np.repeat(np.arange(shape[0]), self.n_rxns_to_erase).flatten(),
                    np.argpartition(random_numbers, kth=self.n_rxns_to_erase, axis=1)[
                        :, : self.n_rxns_to_erase
                    ].flatten(),
                )
            )
        return list_of_inds_to_cover

    def _CV_loops(
        self, perf_dict, cv, X, y, further_action_before_logging, y_yield=None
    ):
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
            y_train, y_test = y[train_ind], y[test_ind]
            print("Y TRAIN, TEST SHAPE", y.shape, y_train.shape, y_test.shape)
            if X is not None:
                X_train, X_test = X[train_ind, :], X[test_ind]
                std = StandardScaler()
                X_train = std.fit_transform(X_train)
                X_test = std.transform(X_test)
            else:
                X_test = y_train
            if y_yield is not None:
                y_yield_test = y_yield[test_ind]
                y_yield_train = y_yield[train_ind]
            print()
            print("CV FOLD", a)
            if self.n_rxns_to_erase >= 1:
                list_of_inds_to_erase = self._random_selection_of_inds_to_erase(len(train_ind))
            for eval_loop_num in range(self.n_evaluations):
                print("EVALUATION #", eval_loop_num)
                if self.n_rxns_to_erase >= 1:
                    # Need this block again so that we use the original arrays for different evaluations.
                    if X is not None:
                        X_train = std.transform(X[train_ind, :])
                    y_train, y_test = y[train_ind], y[test_ind]
                    inds_to_erase = list_of_inds_to_erase[eval_loop_num]
                    # For regressors
                    if type(self) == RegressorEvaluator:
                        inds_to_remove = []
                        inds_to_remove.extend(
                            [
                                self.dataset.n_rank_component * row_num + col_num
                                for row_num, col_num in zip(
                                    inds_to_erase[0], inds_to_erase[1]
                                )
                            ]
                        )
                        inds_to_keep = [
                            x for x in range(len(y_train)) if x not in inds_to_remove
                        ]
                        y_train = y_train[inds_to_keep]
                        X_train = X_train[inds_to_keep]
                    elif type(self) == MulticlassEvaluator:
                        y_train_missing = deepcopy(y_yield_train).astype(float)
                        y_train_missing[inds_to_erase] = np.nan
                        y_train = np.nanargmax(y_train_missing, axis=1)
                    # For label ranking
                    else:
                        y_train_missing = deepcopy(y_train).astype(float)
                        y_train_missing[inds_to_erase] = np.nan
                        if type(self) == LabelRankingEvaluator:
                            y_train = mstats.rankdata(
                                np.ma.masked_invalid(y_train_missing), axis=1
                            )
                            y_train[y_train == 0] = np.nan
                            # print("YTRAIN", y_train_missing)
                        elif type(self) == BaselineEvaluator:
                            y_train = np.nanmean(y_train_missing, axis=0)
                            X_train = y_train
                            X_test = y_train_missing
                        elif type(self) == MultilabelEvaluator:
                            y_yield_train = y_yield[train_ind]
                            y_train_missing = deepcopy(y_yield_train).astype(float)
                            y_train_missing[inds_to_erase] = np.nan
                            # print("YTRAIN", y_train_missing)
                            nonzero_inds = np.argpartition(
                                -1 * y_train_missing, self.n_rxns, axis=1
                            )[:, : self.n_rxns]
                            y_train = np.zeros_like(y_train_missing)
                            for row_num, col_nums in enumerate(nonzero_inds):
                                y_train[np.array([row_num] * self.n_rxns), col_nums] = 1
                            # print("MULTILABEL YTRAIN", y_train)

                for b, (model, model_name) in enumerate(
                    zip(self.list_of_algorithms, self.list_of_names)
                ):
                    ###### Model training phase ###### 
                    if X is not None:
                        # For multilabel LogReg
                        if (
                            type(model) == GridSearchCV
                            and type(model.estimator) == LogisticRegression
                            and type(self) == MultilabelEvaluator
                        ):
                            trained_models = []
                            if len(y_train.shape) == 1:
                                y_train = y_train.reshape(-1, 1)
                            for i in range(y_train.shape[1]):
                                if np.sum(y_train[:, i]) in [0, y_train.shape[0]]:
                                    trained_models.append(
                                        float(np.sum(y_train[:, i]) / y_train.shape[0])
                                    )
                                elif np.sum(y_train[:, i]) > 3:
                                    model.fit(X_train, y_train[:, i])
                                    trained_models.append(model)
                                else:
                                    lr = LogisticRegression(
                                        penalty="l1",
                                        solver="liblinear",  # lbfgs doesn't converge
                                        random_state=42,
                                    )
                                    lr.fit(X_train, y_train[:, i])
                                    trained_models.append(lr)
                            model = trained_models
                        # For nearest neighbor based models, if the substrate features are the only inputs
                        # to the model, use Tanimoto distances.
                        elif (
                            type(model) == GridSearchCV
                            and type(model.estimator)
                            in [KNeighborsClassifier, IBLR_M, IBLR_PL]
                        ) and not self.dataset.train_together:
                            dist_array = self.dataset.X_dist
                            train_dists, test_dists = self._dist_array_train_test_split(
                                dist_array, test_ind
                            )
                            model.fit(train_dists, y_train)
                        else:
                            print("Y TRAIN SHAPE", y_train.shape)
                            model.fit(X_train, y_train)
                    ###### EVALUATION PHASE ###### 
                    # For baseline
                    if y_yield is None:
                        (
                            processed_y_test,
                            processed_pred_rank,
                        ) = further_action_before_logging(model, X_test, y_test)
                    # Nearest neighbors based models require different input from other algorithms.
                    elif (
                        type(model) == GridSearchCV
                        and type(model.estimator)
                        in [KNeighborsClassifier, IBLR_M, IBLR_PL]
                        # and not self.dataset.train_together
                    ):
                        (
                            processed_y_test,
                            processed_pred_rank,
                        ) = further_action_before_logging(
                            model, test_dists, y_yield_test
                        )
                    else:
                        print("Y YIELD TEST", y_yield_test)
                        (
                            processed_y_test,
                            processed_pred_rank,
                        ) = further_action_before_logging(model, X_test, y_yield_test)
                    print("PROCESSED Y TEST", processed_y_test)
                    print("PROCESSED PREDICTED RANK", processed_pred_rank)
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

    def __init__(self, dataset, n_rxns, outer_cv, n_rxns_to_erase=0, n_evaluations=1):
        super().__init__(dataset, n_rxns, outer_cv, n_rxns_to_erase, n_evaluations)
        self.list_of_algorithms = ["Baseline"]
        self.list_of_names = ["Baseline"]

    def _processing_before_logging(self, model, y_train, y_test):
        processed_pred_rank = np.tile(
            yield_to_ranking(np.nanmean(y_train, axis=0)),
            (y_test.shape[0], 1),
        )
        return y_test, processed_pred_rank

    def train_and_evaluate_models(self):
        y = self.dataset.y_yield
        if type(self.outer_cv) == list:
            self.perf_dict = []
            for i, (array, cv) in enumerate(zip(y, self.outer_cv)):
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(
                    perf_dict, cv, None, array, self._processing_before_logging
                )
                self.perf_dict.append(perf_dict)
        else:
            self.perf_dict = deepcopy(PERFORMANCE_DICT)
            self._CV_loops(
                self.perf_dict, self.outer_cv, None, y, self._processing_before_logging
            )
        return self

    def external_validation(self):
        self.valid_dict = deepcopy(PERFORMANCE_DICT)
        y_train = self.dataset.y_yield
        y_valid = self.dataset.y_valid
        self._evaluate_alg(
            self.valid_dict,
            y_valid,
            np.tile(yield_to_ranking(np.mean(y_train, axis=0)), (y_valid.shape[0], 1)),
            "Validation",
            "Baseline",
        )
        return self


class MulticlassEvaluator(Evaluator):
    """Evaluates multiclass classifiers.
    Note that this doesn't process informer, ullmann and borylation datasets.

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
    list_of_algorithms : list
        GridSearchCV objects of algorithms specified by the user.
    perf_dict : dict
        Logging of performances of each test substrate.
    """

    def __init__(
        self,
        dataset,
        feature_type,
        n_rxns,
        list_of_names,
        outer_cv,
        n_rxns_to_erase=0,
        n_evaluations=1,
    ):
        super().__init__(dataset, n_rxns, outer_cv, n_rxns_to_erase, n_evaluations)
        self.feature_type = feature_type
        self.list_of_names = list_of_names
        self.list_of_algorithms = []
        if type(self.dataset) in [
            dataloader.DeoxyDataset,
            dataloader.InformerDataset,
            dataloader.ScienceDataset,
            dataloader.UllmannDataset,
            dataloader.BorylationDataset,
        ]:
            cv = 4
        elif type(self.dataset) == dataloader.NatureDataset:
            cv = 3
        for name in self.list_of_names:
            if name == "RFC":
                self.list_of_algorithms.append(
                    GridSearchCV(
                        RandomForestClassifier(random_state=42),
                        param_grid={
                            "n_estimators": [25, 50, 100],
                            "max_depth": [3, 5, None],
                        },
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                        cv=cv,
                    )
                )
            elif name == "LR":
                self.list_of_algorithms.append(
                    GridSearchCV(
                        LogisticRegression(
                            solver="liblinear",  # lbfgs doesn't converge
                            multi_class="auto",
                            random_state=42,
                        ),
                        param_grid={"penalty": ["l1", "l2"], "C": [0.1, 0.3, 1, 3, 10]},
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                        cv=cv,
                    )
                )
            elif name == "KNN":
                if self.dataset.train_together:
                    metric = "euclidean"
                else:
                    metric = "precomputed"
                self.list_of_algorithms.append(
                    GridSearchCV(
                        KNeighborsClassifier(metric=metric),
                        param_grid={"n_neighbors": [3,5,10]},
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                        cv=cv,
                    )
                )

    def _get_full_class_proba(self, pred_proba, model):
        """ " When the training set is only exposed to a subset of target classes,
        the predicted probability for those classes with the multiclass classifier is 0.
        To measure metrics, this needs to be accounted for.
        This function fills up the classes not in the training set.

        Parameters
        ----------
        pred_proba : list of n_classes number of np.ndarrays of shape (n_samples, )
            Predicted positive probability values.
        model : sklearn classifier object
            Trained model that gives the pred_proba array.

        Returns
        -------
        new_pred_proba :
            Updated pred_proba array with all classes.
        """
        if pred_proba.shape[0] == 1 :
            new_pred_proba_list = []
            for i in range(self.dataset.n_rank_component):
                if i not in model.classes_:
                    new_pred_proba.append(0)
                else:
                    new_pred_proba.append(pred_proba[0][list(model.classes_).index(i)])
            new_pred_proba = np.array(new_pred_proba_list)
        else :
            allzero_column_idx = []
            for i in range(self.dataset.n_rank_component):
                if i not in model.classes_:
                    allzero_column_idx.append(i)
            new_pred_proba = np.insert(pred_proba, tuple(allzero_column_idx), 0, axis=1)
                    
        return new_pred_proba

    def _processing_before_logging(self, model, X_test, y_yield_test):
        pred_proba = model.predict_proba(X_test)
        print("PRED PROBA", pred_proba)
        if len(pred_proba[0]) < self.dataset.n_rank_component:
            pred_proba = self._get_full_class_proba(pred_proba, model)
        pred_rank_reshape = yield_to_ranking(pred_proba)

        return y_yield_test, pred_rank_reshape

    def train_and_evaluate_models(self):
        X = self._load_X()
        y_rank = self.dataset.y_ranking
        y_yield = self.dataset.y_yield
        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but separating the datasets
            y = [
                np.argmin(y_sub_rank, axis=1) for y_sub_rank in y_rank
            ]  # transforming ranks into multiclass outputs
            self.perf_dict = []
            for i, (X_array, array, yield_array, cv) in enumerate(
                zip(X, y, y_yield, self.outer_cv)
            ):  # X is not included as it remains the same across different reagents
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(
                    perf_dict,
                    cv,
                    X_array,
                    array,
                    self._processing_before_logging,
                    y_yield=yield_array,
                )
                self.perf_dict.append(perf_dict)
        else:
            y = np.argmin(y_rank, axis=1)
            perf_dict = deepcopy(PERFORMANCE_DICT)
            self._CV_loops(
                perf_dict,
                self.outer_cv,
                X,
                y,
                self._processing_before_logging,
                y_yield=y_yield,
            )
            self.perf_dict = perf_dict
        return self

    def external_validation(self):
        self.valid_dict = deepcopy(PERFORMANCE_DICT)
        y_rank_train = self.dataset.y_label
        y_rank_valid = self.dataset.y_valid
        y_yield_train = self.dataset.y_yield
        y_yield_valid = self.dataset.y_valid
        for b, (model, model_name) in enumerate(
            zip(self.list_of_algorithms, self.list_of_names)
        ):
            if type(model.estimator) != KNeighborsClassifier:
                X_train = self._load_X()
                X_valid = self.dataset.X_valid
                model.fit(X_train, y_rank_train)
            else:
                X_train = self.dataset.X_dist
                X_valid = self.dataset.X_valid
                model.fit(X_train, y_rank_train)
            pred_proba = model.predict_proba(X_valid)
            if len(pred_proba[0]) < self.dataset.n_rank_component:
                pred_proba = self._get_full_class_proba(pred_proba, model)
            self._evaluate_alg(
                self.valid_dict,
                y_yield_valid,
                yield_to_ranking(pred_proba),
                "Validation",
                model_name,
            )
        return self


class MultilabelEvaluator(MulticlassEvaluator):
    """Evaluates multilabel classifiers.

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

    """
    def __init__(
        self,
        dataset,
        feature_type,
        n_rxns,
        list_of_names,
        outer_cv,
        n_rxns_to_erase=0,
        n_evaluations=1,
    ):
        super().__init__(
            dataset,
            feature_type,
            n_rxns,
            list_of_names,
            outer_cv,
            n_rxns_to_erase,
            n_evaluations,
        )

        self.list_of_algorithms = []
        for name in self.list_of_names:
            if name == "RFC":
                self.list_of_algorithms.append(
                    GridSearchCV(
                        RandomForestClassifier(random_state=42),
                        param_grid={
                            "n_estimators": [25, 50, 100],
                            "max_depth": [3, 5, None],
                        },
                        scoring="roc_auc",
                        n_jobs=-1,
                        cv=3,
                    )
                )
            elif name == "LR":
                self.list_of_algorithms.append(
                    GridSearchCV(
                        LogisticRegression(
                            solver="liblinear",  # lbfgs doesn't converge
                            random_state=42,
                        ),
                        param_grid={
                            "penalty": ["l1", "l2"],
                            "C": [0.1, 0.3, 1, 3, 10],
                        },
                        scoring="roc_auc",
                        n_jobs=-1,
                        cv=3,
                    )
                )
            elif name == "KNN":
                if self.dataset.train_together:
                    metric = "euclidean"
                else:
                    metric = "precomputed"
                self.list_of_algorithms.append(
                    GridSearchCV(
                        KNeighborsClassifier(metric=metric),
                        param_grid={"n_neighbors": [2, 4, 6]},
                        scoring="roc_auc",
                        n_jobs=-1,
                        cv=4,
                    )
                )

    def _processing_before_logging(self, model, X_test, y_yield_test):
        if type(model) == list:  # Only for multilabel Logistic Regressions
            pred_proba = []
            for lr in model:
                if type(lr) == float:
                    if not self.dataset.train_together:
                        pred_proba.append(np.array([[lr]]))
                    else:
                        pred_proba.append(lr * np.ones((X_test.shape[0], 2)))
                else:
                    pred_proba.append(lr.predict_proba(X_test))
            # print(pred_proba)
        else:
            # print(X_test)
            if X_test.ndim == 1 :
                pred_proba = model.predict_proba(X_test.reshape(1,-1))
            else :
                pred_proba = model.predict_proba(X_test)
        print("PRED PROBA SHAPE", pred_proba)
        if self.dataset.train_together or type(self.dataset) in [
            dataloader.UllmannDataset,
            dataloader.BorylationDataset,
        ]:
            arrays_to_stack = []
            for proba_array in pred_proba:
                if proba_array.shape[1] > 1:
                    arrays_to_stack.append(proba_array[:, 1].reshape(-1, 1))
                elif proba_array.shape[0] > 1:
                    arrays_to_stack.append(proba_array.flatten().reshape(-1, 1))
                else:
                    assert len(proba_array.flatten()) == 1
                    arrays_to_stack.append(
                        np.repeat(proba_array, X_test.shape[0], axis=0)
                    )
            pred_proba = np.hstack(tuple(arrays_to_stack))
            y_test_reshape = y_yield_test
        else:
            pred_proba = np.array(
                [x[0][1] if len(x[0]) == 2 else 1 - x[0][0] for x in pred_proba]
            )
            y_test_reshape = y_yield_test.flatten()
        pred_rank_reshape = yield_to_ranking(pred_proba)

        return y_test_reshape, pred_rank_reshape

    def train_and_evaluate_models(self):
        X = self._load_X()
        y_label = self.dataset.y_label
        y_yield = self.dataset.y_yield
        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but separating the datasets
            self.perf_dict = []
            for i, (X_array, array, yield_array, cv) in enumerate(
                zip(X, y_label, y_yield, self.outer_cv)
            ):  # X is not included as it remains the same across different reagents
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(
                    perf_dict,
                    cv,
                    X_array,
                    array,
                    self._processing_before_logging,
                    y_yield=yield_array,
                )
                self.perf_dict.append(perf_dict)
        else:
            perf_dict = deepcopy(PERFORMANCE_DICT)
            self._CV_loops(
                perf_dict,
                self.outer_cv,
                X,
                y_label,
                self._processing_before_logging,
                y_yield=y_yield,
            )
            self.perf_dict = perf_dict
        return self


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
        list_of_algorithms,
        outer_cv,
        n_rxns_to_erase=0,
        n_evaluations=1,
    ):
        super().__init__(dataset, n_rxns, outer_cv, n_rxns_to_erase, n_evaluations)
        self.feature_type = feature_type
        self.list_of_algorithms = list_of_algorithms
        self.list_of_names = list_of_names

    def _processing_before_logging(self, model, X_test, y_yield_test):
        if type(self.dataset) == dataloader.InformerDataset:
            y_test_reshape = y_yield_test.flatten()
            if X_test.ndim == 1 :
                pred_rank_reshape = model.predict(X_test.reshape(1,-1)).flatten()
            else :
                pred_rank_reshape = model.predict(X_test).flatten()
        else :
            y_test_reshape = y_yield_test
            pred_rank_reshape = model.predict(X_test)
        return y_test_reshape, pred_rank_reshape

    def train_and_evaluate_models(self):
        X = self._load_X()
        y = self.dataset.y_ranking
        y_yield = self.dataset.y_yield
        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but separating the datasets
            self.perf_dict = []
            for i, (X_array, array, yield_array, cv) in enumerate(
                zip(X, y, y_yield, self.outer_cv)
            ):  # X is not included as it remains the same across different reagents
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(
                    perf_dict,
                    cv,
                    X_array,
                    array,
                    self._processing_before_logging,
                    y_yield=yield_array,
                )
                self.perf_dict.append(perf_dict)
        else:
            perf_dict = deepcopy(PERFORMANCE_DICT)
            self._CV_loops(
                perf_dict,
                self.outer_cv,
                X,
                y,
                self._processing_before_logging,
                y_yield=y_yield,
            )
            self.perf_dict = perf_dict
        return self

    def external_validation(self):
        self.valid_dict = deepcopy(PERFORMANCE_DICT)
        X_train = self._load_X()
        X_valid = self.dataset.X_valid
        y_rank_train = self.dataset.y_ranking
        y_rank_valid = self.dataset.y_valid
        y_yield_train = self.dataset.y_yield
        y_yield_valid = self.dataset.y_valid
        for b, (model, model_name) in enumerate(
            zip(self.list_of_algorithms, self.list_of_names)
        ):
            model.fit(X_train, y_rank_train)
            pred_ranking = model.predict(X_valid)
            self._evaluate_alg(
                self.valid_dict,
                y_yield_valid,
                pred_ranking,
                "Validation",
                model_name,
            )
        print(self.valid_dict)
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
        n_rxns_to_erase=0,
        n_evaluations=1,
    ):
        super().__init__(
            regressor_dataset, n_rxns, outer_cv, n_rxns_to_erase, n_evaluations
        )
        self.feature_type = feature_type
        self.list_of_algorithms = list_of_algorithms
        self.list_of_names = list_of_names

    def _processing_before_logging(self, model, X_test, y_test):
        if type(self.dataset) in [
            dataloader.UllmannDataset,
            dataloader.BorylationDataset,
            dataloader.NatureDataset,
            dataloader.ScienceDataset,
            dataloader.DeoxyDataset
        ]:
            n_conds = self.dataset.n_rank_component
            y_test_reshape = y_test.reshape((len(y_test) // n_conds, n_conds))
            pred_rank_reshape = yield_to_ranking(
                model.predict(X_test).reshape((len(X_test) // n_conds, n_conds))
            )
        elif type(self.dataset) == dataloader.InformerDataset:
            y_test_reshape = y_test.flatten()
            pred_rank_reshape = yield_to_ranking(model.predict(X_test).flatten())
        return y_test_reshape, pred_rank_reshape

    def train_and_evaluate_models(self):
        X = self._load_X()
        y = self.dataset.y_yield
        
        self.models = [[] for _ in range(len(self.list_of_algorithms))]
        self.cv_scores = [[] for _ in range(len(self.list_of_algorithms))]

        if (
            type(self.outer_cv) == list
        ):  # When one component is ranked but the dataset is separated by the other component (deoxy, informer)
            self.perf_dict = []
            for i, (array, cv) in enumerate(zip(y, self.outer_cv)):
                perf_dict = deepcopy(PERFORMANCE_DICT)
                self._CV_loops(
                    perf_dict,
                    cv,
                    X,  # X_array,
                    array,
                    self._processing_before_logging,
                    y_yield=array,  # yield_array
                )
                self.perf_dict.append(perf_dict)
        # cases when both reaction components are ranked simultaneously
        else:
            perf_dict = PERFORMANCE_DICT
            print("y shape", y.shape, X.shape)
            self._CV_loops(
                perf_dict,
                self.outer_cv,
                X,
                y,
                self._processing_before_logging,
                y_yield=y,
            )
            self.perf_dict = perf_dict
        return self
