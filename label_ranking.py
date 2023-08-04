import numpy as np
from copy import deepcopy
from itertools import combinations
from scipy.optimize import line_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from rank_aggregation import *

### We deal with y as rankings, not scores or preferences. The smaller values, the better.


class RPC:
    """Reproduces the ranking by pairwise comparison method proposed in
    Hüllermeier et al. Artif. Intel. 2008, 1897. Label ranking by learning pairwise preferences.
    The paper uses logistic regression as their base learner, but this code fixes it such that any binary classifier is used.

    Parameters
    ----------
    base_learner: Binary classifier.
        Algorithm to use for predicting label preference.
    cross_validator : GridSearchCV object.
        Defines how to conduct cross validation for each predictor.
    vote_aggregator : TODO

    Attributes
    ----------

    """

    def __init__(
        self,
        base_learner,
        cross_validator=None,
        # vote_aggregator
    ):
        self.base_learner = base_learner
        self.cross_validator = cross_validator
        # self.vote_aggregator = vote_aggregator

    def fit(self, X, y):
        """Builds a binary classifier for all possible pairs of labels.

        Parameters
        ---------
        X : np.ndarray of shape (n_samples, n_features)
            Input array of descriptors.
        y : np.ndarray of shape (n_samples, n_labels)
            Output array of continuous yield values.
        """
        self.learner_by_column_pair = {}
        self.n_labels = y.shape[1]
        for column_combination in combinations(range(y.shape[1]), 2):
            sub_y = y[:, list(column_combination)]
            # np.argmin : Label with higher column index is assigned 1 if it is preferred (lower ranking value).
            # three conditions : insures that both labels are recorded with different continuous values.
            sub_preference = np.argmin(
                sub_y[
                    (~np.isnan(sub_y[:, 0]))
                    & (~np.isnan(sub_y[:, 1]))
                    & (sub_y[:, 0] != sub_y[:, 1])
                ],
                axis=1,
            )
            if len(sub_preference) > 0 and len(np.unique(sub_preference)) > 1:
                sub_X = X[
                    (~np.isnan(sub_y[:, 0]))
                    & (~np.isnan(sub_y[:, 1]))
                    & (sub_y[:, 0] != sub_y[:, 1])
                ]
                if self.cross_validator is None:
                    model = deepcopy(self.base_learner)
                    model.fit(sub_X, sub_preference)
                else:
                    self.cross_validator.fit(sub_X, sub_preference)
                    model = self.cross_validator.best_estimator_
                self.learner_by_column_pair.update({column_combination: model})
            # If there one label constantly is preferred over the other
            elif len(sub_preference) > 0:
                # Assigning the column label that is ALWAYS RANKED HIGHER
                self.learner_by_column_pair.update(
                    {column_combination: int(np.unique(sub_preference))}
                )
        return self

    def predict(self, X):
        """Builds a ranking for the case where labels are fully labeled.
        TODO: need to incorporate vote aggregator

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input array.

        Returns
        -------
        ranking_array : np.ndarray of shape (n_samples, n_labels)
        """
        score_array = np.zeros((X.shape[0], self.n_labels, self.n_labels))
        for column_combination, model in self.learner_by_column_pair.items():
            # assert column_combination[0] < column_combination[1]
            if type(model) == int:
                score_array[
                    :, column_combination[model], column_combination[1 - model]
                ] = 1

            else:
                proba = model.predict_proba(X)
                score_array[:, column_combination[1], column_combination[0]] = proba[
                    :, 1
                ]
                score_array[:, column_combination[0], column_combination[1]] = proba[
                    :, 0
                ]
        order = np.argsort(-np.sum(score_array, axis=2), axis=1)
        rank = np.argsort(order, axis=1) + np.ones_like(order)
        return rank


class IBLR_M:
    """Reproduces the instance based label ranking with the probabilistic mallows model, proposed in
    W. Cheng, W, Hühn, E. Hüllermeier, ICML, 2009. Decision Tree and Instance-Based Learning for Label Ranking.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors to consider.
    metric : str or callable function
        Metric to use for distance computation.
        For list of possible str, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
        If metric is a callable function, it takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors.

    """

    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1
        )
        neigh.fit(X)
        self.neigh = neigh
        self.y_train = y
        return self

    def _most_probable_extension(self, bold_sigma, pi_hat):
        """Finds the most probable extension of a ranking sigma_i, given local most probable ranking pi_hat.
        Follows Proposition 1.

        Parameters
        ----------
        bold_sigma : np.ndarray of shape (n_neighbors, n_labels)
            Ranking labels of neighbors, where unranked labels are np.nan

        pi_hat : np.ndarray of shape (n_labels,)
            Locally most probable ranking, found by borda count of sigma_i

        Returns
        --------
        sigma_star : np.ndarray of shape (n_neighbors, n_labels)
            Consistently extended rankings of neighbors.
        """
        sigma_star = np.zeros_like(bold_sigma)
        for a, sigma in enumerate(bold_sigma):
            if len(np.where(np.isnan(sigma))[0]) > 0:
                sigma_star_i = deepcopy(sigma)
                empty_labels = np.argwhere(np.isnan(sigma)).flatten()
                assigned_position_by_empty_label = np.zeros_like(empty_labels)
                ranked_labels = list(np.where(~np.isnan(sigma))[0])
                for i, empty_label in enumerate(empty_labels):
                    min_discordant_num = 9999999
                    position_to_assign = 9999999
                    for j in range(
                        len(ranked_labels) + 1
                    ):  # j is the position that the empty_label will be inserted into.
                        for k in ranked_labels:  # Going through all ranked labels
                            labels_before_j = np.where(sigma <= k)[0]
                            pi_k_greater_than_pi_i = np.where(
                                pi_hat > pi_hat[empty_label]
                            )

                            labels_after_j = np.where(sigma > k)[0]
                            pi_k_smaller_than_pi_i = np.where(
                                pi_hat < pi_hat[empty_label]
                            )

                            num_discordant = len(
                                np.intersect1d(labels_before_j, pi_k_greater_than_pi_i)
                            ) + len(
                                np.intersect1d(labels_after_j, pi_k_smaller_than_pi_i)
                            )
                            if num_discordant < min_discordant_num:
                                min_discordant_num = num_discordant
                                position_to_assign = k
                    assigned_position_by_empty_label[i] = position_to_assign
                for empty_label_ind, assigned_position in enumerate(
                    assigned_position_by_empty_label
                ):
                    sigma_star_i[empty_labels[empty_label_ind]] = (
                        assigned_position + 0.5
                    )

                # If there are multiple indices inserted at the same position, we put them in the same order as in pi.
                for val in np.unique(sigma_star_i):
                    if len(np.where(sigma_star_i == val)[0]) > 0 and int(val) != val:
                        inds_to_adjust = np.where(sigma_star_i == val)[0]
                        adjust_vals = [
                            x / (len(inds_to_adjust) + 1)
                            for x in range(1, len(inds_to_adjust) + 1)
                        ]
                        for a, ind in enumerate(np.argsort(pi_hat[inds_to_adjust])):
                            sigma_star_i[inds_to_adjust[ind]] += -0.5 + adjust_vals[a]
                sigma_star[a] = sigma_star_i
            else:
                sigma_star[a] = sigma
        order = np.argsort(sigma_star, axis=1)
        rank = np.argsort(order, axis=1) + np.ones_like(order)
        return rank

    def _get_theta_hat(self, bold_sigma, pi_hat):
        """Computes the maximum likelihood estimation of spread parameter theta_hat
        from the mean observed distance from pi_hat.
        Will implement later

        Parameter
        ---------
        bold_sigma : np.ndarray of shape (n_neighbors, n_labels)
            Ranking labels of neighbors, where unranked labels are np.nan

        pi_hat : np.ndarray of shape (n_labels,)
            Locally most probable ranking, found by borda count of sigma_i

        Returns
        --------
        theta_hat : float
            Estimated value of spread parameter which can be interpreted as model confidence.
        """
        pass

    def predict(self, X):
        """Predicts the rankings for test instances.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input array.

        Returns
        -------
        pi_hat : np.ndarray of shape (n_samples, n_labels)
            Estimated rankings for each sample.
        """
        # Line 1 : Find the k nearest neghbors of x in T
        neighbor_inds = self.neigh.kneighbors(
            X, self.n_neighbors, return_distance=False
        )
        # Line 2 : get neighbor rankings
        neighbor_rankings = np.zeros(
            (X.shape[0], self.y_train.shape[1], self.n_neighbors)
        )
        for i, row in enumerate(neighbor_inds):
            neighbor_rankings[i, :, :] = self.y_train[row, :].T
        # Line 3 : get generalized Borda count from neighbor rankings
        pi_hat = borda(neighbor_rankings)  # shape n_samples, n_labels
        pi = np.zeros_like(pi_hat)
        count = 0
        while np.any(pi != pi_hat):
            if count > 0:
                pi_hat = pi
            # Lines 4~8 : First M step - filling incomplete rankings so that they are the most probable extension
            all_sigma_star = np.zeros_like(neighbor_rankings)
            for i in range(X.shape[0]):
                sigma_star_by_instance = self._most_probable_extension(
                    neighbor_rankings[i, :, :].T, pi_hat
                )
                all_sigma_star[i, :, :] = sigma_star_by_instance.T
            # Line 9
            pi = borda(
                all_sigma_star
            )  # will need to fix this into completed_neighbor_rankings
            count += 1
        return pi_hat
        # np.ones_like(pi_hat) * pi_hat.shape[1] - pi_hat
        # theta_hat = np.zeros(neighbor_rankings.shape[0])
        # for i in range(neighbor_rankings.shape[0]):
        #     theta_hat[i] = self._get_theta_hat(neighbor_rankings[i,:,:].T, pi_hat)
        # return theta_hat, pi_hat


class IBLR_PL:
    """Reproduces the instance based label ranking with the probabilistic Plackett-Luce model, posposed in
    W. Cheng, K. Dembczyński, E. Hüllermeier, ICML, 2010. Label Ranking Methods Based on the Plackett-Luce Model.
    Currently, the kendall tau values are negative - need to figure out why but values seem high.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors to consider.
    metric : str or callabel function. Default: "euclidean"
        Metric to use for distance computation.
        For list of possible str, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
        If metric is a callable function, it takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors.
    """

    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        """Fits the nearest neighbor model and stores the ranking information.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input array.
        y : np.ndarray of shape (n_samples, n_labels)
            Ranking array.
        """
        neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1
        )
        neigh.fit(X)
        self.neigh = neigh
        self.y_train = y
        return self

    def _plackett_luce(self, rankings):
        """
        Modification of the orignial implementation in reference below.
        The matlab code which was found at http://sites.stat.psu.edu/~dhunter/code/btmatlab/plackmm.m
        was modified.

        Reference : Sections 5 and 6 starting on p.395 (p.12 in pdf) of
         Hunter, David R. MM algorithms for generalized Bradley-Terry models.
            Ann. Statist. 32 (2004), no. 1, 384--406. doi:10.1214/aos/1079120141.
            http://projecteuclid.org/euclid.aos/1079120141.

        Parameters
        ----------
        rankings : np.ndarray of shape (n_neighbors, n_labels)
            Ranking array. Missing labels = np.nan, minimum value = 1
        """
        a = np.zeros((rankings.shape[0] * rankings.shape[1], 3))
        row_count = 0
        for neighbor_num, row in enumerate(rankings):
            for label_num, rank in enumerate(row):
                a[row_count] = [label_num, neighbor_num, rank]
                row_count += 1
        assert row_count == a.shape[0]
        # a: nx3 input matrix with columns (label ID, neighbor ID, rank)
        M = rankings.shape[1]  # M = total # of labels
        N = rankings.shape[0]  # N = total # of neighbors

        w = np.zeros(M, dtype=int)  # w[i] = # times label i placed higher than last
        pp = np.count_nonzero(
            ~np.isnan(rankings), axis=1
        )  # pp[j] = # individuals in contest j

        for row in rankings:
            num = np.sum(~np.isnan(row))
            w += row < num

        gamma = np.ones(M)  # (unscaled) initial gamma vector
        dgamma = 1
        iterations = 0

        while np.linalg.norm(dgamma) > 1e-09:
            iterations += 1
            g = np.zeros_like(rankings, dtype=np.float32)
            for i, row in enumerate(rankings):
                for j in range(1, pp[i]):  # excluding lowest rank
                    g[i, j - 1] = np.reciprocal(np.sum(gamma[np.where(row <= j)[0]]))
            # at this point, g(i,j) should be reciprocal of the
            # sum of gamma's for places j and higher in i'th neighbor
            # except for j = last place

            ## To calculate the loglikelihood (not necessary):
            #    ell = w'*log(gamma)+sum(sum(log(g(g>0))));

            g = np.cumsum(g, axis=1)
            # Now g(i, j) should be the sum of all the denominators for jth place in ith neighbor.

            denominator = np.zeros(M)
            # Now for the denominator in gamma(i), we need to add up all g(j, r(i,j)) for nonzero r over all neighbors.
            for i in range(M):
                col = rankings[:, i].flatten()
                denominator[i] = np.sum(
                    g[np.where(~np.isnan(col))[0], col[~np.isnan(col)] - 1]
                )
            new_gamma = np.divide(w, denominator)
            dgamma = new_gamma - gamma
            gamma = new_gamma

        iterations = iterations
        out = gamma / np.sum(gamma)
        return out

    def predict(self, X):
        """Predicts the ranking for test cases.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input array.

        Returns
        -------
        pi_hat : np.ndarray of shape (n_samples, n_labels)
            Estimated rankings for each sample.
        """
        # Find the k nearest neghbors of x in T
        neighbor_inds = self.neigh.kneighbors(
            X, self.n_neighbors, return_distance=False
        )
        pl_param_array = np.zeros((X.shape[0], self.y_train.shape[1]))
        for i, row in enumerate(neighbor_inds):
            rankings = self.y_train[row]
            pl_param_array[i, :] = self._plackett_luce(rankings)
        # Sort the array such that the larger the parameter, better the ranking (i.e. lower rank value)
        return pl_param_array.shape[1] - np.argsort(
            np.argsort(pl_param_array, axis=1), axis=1
        )


class LabelRankingRandomForest:
    """Reproduces the label ranking random forest method proposed in
    Y. Zhou and G. Qiu, Expert Systems App. 2018, 99. Random forest for label ranking.
    This paper builds on random forests.
    Considers only top label in ranking as class.
    Log-loss or entropy criterion
    """

    def __init__(
        self,
        n_estimators=50,
        max_depth=8,
        cross_validator=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cross_validator = cross_validator

    def fit(self, X, y):
        """Builds a random forest classifier by TLAC (top label as class).
        Also stores the training instance that arrives at each node.

        Parameters
        ---------
        X : np.ndarray of shape (n_samples, n_features)
            Input array of descriptors.
        y : np.ndarray of shape (n_samples, n_labels)
            Output array of continuous yield values.
        """
        # Applies TLAC to y array
        tlac_y = np.argmin(y, axis=1)

        if self.cross_validator is not None:
            self.cross_validator.fit(X, tlac_y)
            model = self.cross_validator.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                criterion="log_loss",
                max_features="log2",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X, tlac_y)
        self.model = model
        self.n_labels = y.shape[1]

        borda_rank_by_tree_and_leaf = {}
        # Gets what training instances arrive at which node
        X_leaves = model.apply(X)
        for i in range(X_leaves.shape[1]):
            borda_rank_by_tree_and_leaf.update({i: {}})
            for leaf_num in np.unique(X_leaves[:, i]):
                inds = np.where(X_leaves[:, i] == leaf_num)[0]
                borda_rank = borda_count(np.expand_dims(y[inds, :].T, axis=0))
                borda_rank_by_tree_and_leaf[i].update({leaf_num: borda_rank})

        self.borda_by_tree_and_leaf = borda_rank_by_tree_and_leaf

        return self

    def predict(self, X):
        """Builds a ranking for the case where labels are fully labeled.
        TODO: need to incorporate vote aggregator

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input array.

        Returns
        -------
        ranking_array : np.ndarray of shape (n_samples, n_labels)
        """
        X_leaves = self.model.apply(X)
        collected_borda_array = np.zeros(
            (X_leaves.shape[0], self.n_labels, len(self.model.estimators_))
        )
        for i in range(X_leaves.shape[0]):
            row = X_leaves[i, :]
            for j, leaf_num in enumerate(row):
                collected_borda_array[i, :, j] = self.borda_by_tree_and_leaf[j][
                    leaf_num
                ]
        ranking_array = borda_count(collected_borda_array)
        return ranking_array


class Baseline:
    """ A baseline method that selects the top-k reaction conditions from the training dataset 
    by either highest average yield / borda aggregated ranking / modal aggregated ranking.

    Parameters
    ----------
    criteria : str {"avg_yield", "borda", "modal"}
        How to select the reactions.
            avg_yield : selects the reaction conditions that give highest average yield across substrate pairs in the training data.
                        If this is selected, we assume that y given for fit is an array of rankings.
            borda: selects the highest ranked conditions after borda aggregation.
            modal: selects the condition that has the most frequent number of highest ranks.
    
    k : int 
        Number of reactions to select.
    """

    def __init__(
        self,
        criteria="avg_yield"
    ):
        self.criteria = criteria

    def fit(self, X, y):
        """ Selects the reaction condition to return. 
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Unnecessary for baselines, but kept for consistency with other classes.
        y : np.ndarray of shape (n_samples, n_labels)
            Can be an array of either raw yield values or rankings.
        """
        if self.criteria == "avg_yield" :
            pred_y_inds = np.argsort(np.mean(y, axis=0))[::-1]
        elif self.criteria == "borda" :
            pred_y_inds = np.argsort(borda(np.expand_dims(y.T, axis=0)).flatten())
        elif self.criteria == "modal" :
            pred_y_inds = np.argsort(modal(np.expand_dims(y.T, axis=0)).flatten())
        self.pred_y_inds = pred_y_inds
        return self
    
    def predict(self, X) :
        """ Returns the baseline predictions for all test reactions. 
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test feature array.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples, self.k)
            Suggested top reactions.
        """
        return np.vstack(tuple([self.pred_y_inds]*X.shape[0]))