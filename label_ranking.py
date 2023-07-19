import numpy as np
from copy import deepcopy
from itertools import combinations
from scipy.optimize import line_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from rank_aggregation import *



class RPC():
    """ Reproduces the ranking by pairwise comparison method proposed in 
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
    ) :
        self.base_learner = base_learner
        self.cross_validator = cross_validator
        # self.vote_aggregator = vote_aggregator

    def fit(self, X, y):
        """ Builds a binary classifier for all possible pairs of labels.
         
        Parameters
        ---------
        X : np.ndarray of shape (n_samples, n_features)
            Input array of descriptors.
        y : np.ndarray of shape (n_samples, n_labels)
            Output array of continuous yield values.
        """
        self.learner_by_column_pair = {}
        self.n_labels = y.shape[1]
        for column_combination in combinations(range(y.shape[1]), 2) :
            sub_y = y[:, list(column_combination)]
            # np.argmax : Label with higher column index is assigned 1 if it is preferred.
            # three conditions : insures that both labels are recorded with different continuous values.
            sub_preference = np.argmax(
                sub_y[(~np.isnan(sub_y[:, 0])) & (~np.isnan(sub_y[:, 1])) & (sub_y[:, 0] != sub_y[:, 1])],
                axis=1
            )
            if len(sub_preference) > 0 and len(np.unique(sub_preference)) > 1 :
                sub_X = X[(~np.isnan(sub_y[:, 0])) & (~np.isnan(sub_y[:, 1])) & (sub_y[:, 0] != sub_y[:, 1])]
                if self.cross_validator is None : 
                    model = deepcopy(self.base_learner)
                    model.fit(sub_X, sub_preference) 
                else :
                    self.cross_validator.fit(sub_X, sub_preference)
                    model = self.cross_validator.best_estimator_
                self.learner_by_column_pair.update({column_combination:model})
            # If there one label constantly is preferred over the other 
            elif len(sub_preference) > 0 :
                # Assigning the column label that is ALWAYS RANKED HIGHER
                self.learner_by_column_pair.update({column_combination:int(np.unique(sub_preference))})
        return self

    def predict(self, X) :
        """ Builds a ranking for the case where labels are fully labeled. 
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
        for column_combination, model in self.learner_by_column_pair.items() :
            # assert column_combination[0] < column_combination[1]
            if type(model) == int :
                score_array[:,column_combination[model],column_combination[1-model]] = 1
                
            else : 
                proba = model.predict_proba(X)
                score_array[:,column_combination[1], column_combination[0]] = proba[:,1]
                score_array[:,column_combination[0], column_combination[1]] = proba[:,0]
        order = np.argsort(np.sum(score_array, axis=2), axis=1)
        rank = np.argsort(order, axis=1) + np.ones_like(order)
        return rank


class IBLR_M():
    """ Reproduces the instance based label ranking with the probabilistic mallows model, proposed in
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
    def __init__(
        self,
        n_neighbors,
        metric
    ) :
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y) :
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1)
        neigh.fit(X)
        self.neigh = neigh
        self.y_train = y
        return self
    
    def _most_probable_extension(self, bold_sigma, pi_hat):
        """ Finds the most probable extension of a ranking sigma_i, given local most probable ranking pi_hat.
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
        for a, sigma in enumerate(bold_sigma) :
            if len(np.where(np.isnan(sigma))[0])  > 0 :
                sigma_star_i = deepcopy(sigma)
                empty_labels = np.argwhere(np.isnan(sigma)).flatten()
                assigned_position_by_empty_label = np.zeros_like(empty_labels)
                ranked_labels = list(np.where(~np.isnan(sigma))[0])
                for i, empty_label in enumerate(empty_labels) :
                    min_discordant_num = 9999999
                    position_to_assign = 9999999
                    for j in range(len(ranked_labels) + 1) : # j is the position that the empty_label will be inserted into.
                        for k in ranked_labels : # Going through all ranked labels
                            labels_before_j = np.where(sigma <= k)[0]
                            pi_k_greater_than_pi_i = np.where(pi_hat > pi_hat[empty_label])

                            labels_after_j = np.where(sigma > k)[0]
                            pi_k_smaller_than_pi_i = np.where(pi_hat < pi_hat[empty_label])

                            num_discordant = len(np.intersect1d(labels_before_j, pi_k_greater_than_pi_i)) +\
                                len(np.intersect1d(labels_after_j, pi_k_smaller_than_pi_i))
                            if num_discordant < min_discordant_num :
                                min_discordant_num = num_discordant
                                position_to_assign = k
                    assigned_position_by_empty_label[i] = position_to_assign
                for empty_label_ind, assigned_position in enumerate(assigned_position_by_empty_label) :
                    sigma_star_i[empty_labels[empty_label_ind]] = assigned_position + 0.5
                
                # If there are multiple indices inserted at the same position, we put them in the same order as in pi.
                for val in np.unique(sigma_star_i) :
                    if len(np.where(sigma_star_i == val)[0]) > 0 and int(val)!=val :
                        inds_to_adjust = np.where(sigma_star_i == val)[0]
                        adjust_vals = [x/(len(inds_to_adjust)+1) for x in range(1,len(inds_to_adjust)+1)]
                        for a, ind in enumerate(np.argsort(pi_hat[inds_to_adjust])) :
                            sigma_star_i[inds_to_adjust[ind]] += (-0.5+adjust_vals[a]) 
                sigma_star[a] = sigma_star_i
            else :
                sigma_star[a] = sigma
        order = np.argsort(sigma_star, axis=1)
        rank = np.argsort(order, axis=1) + np.ones_like(order)
        return rank
    
    
    def _get_theta_hat(self, bold_sigma, pi_hat):
        """ Computes the maximum likelihood estimation of spread parameter theta_hat 
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



    def predict(self, X) : 
        """ Predicts the rankings for test instances.
        
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
        neighbor_inds = self.neigh.kneighbors(X, self.n_neighbors, return_distance=False)
        # Line 2 : get neighbor rankings
        neighbor_rankings = np.zeros((X.shape[0], self.y_train.shape[1], self.n_neighbors))
        for i, row in enumerate(neighbor_inds) :
            neighbor_rankings[i, :, :] = self.y_train[row, :].T
        # Line 3 : get generalized Borda count from neighbor rankings
        pi_hat = borda(neighbor_rankings) # shape n_samples, n_labels
        pi = np.zeros_like(pi_hat)
        count = 0
        while pi != pi_hat :
            if count > 0 :
                pi_hat = pi
            # Lines 4~8 : First M step - filling incomplete rankings so that they are the most probable extension
            all_sigma_star = np.zeros_like(neighbor_rankings)
            for i in range(X.shape[0]) :
                sigma_star_by_instance = self._most_probable_extension(neighbor_rankings[i,:,:].T, pi_hat)
                all_sigma_star[i,:,:] = sigma_star_by_instance.T
            # Line 9
            pi = borda(all_sigma_star) # will need to fix this into completed_neighbor_rankings
            count += 1
        return pi_hat
        # theta_hat = np.zeros(neighbor_rankings.shape[0])
        # for i in range(neighbor_rankings.shape[0]):
        #     theta_hat[i] = self._get_theta_hat(neighbor_rankings[i,:,:].T, pi_hat)
        # return theta_hat, pi_hat
        

class LabelRankingRandomForest():
    """ Reproduces the label ranking random forest method proposed in 
    Y. Zhou and G. Qiu, Expert Systems App. 2018, 99. Random forest for label ranking.
    This paper builds on random forests.
    Considers only top label in ranking as class.
    Log-loss or entropy criterion
    """
    def __init__(
        self,
        cross_validator=None,
    ) :
        self.cross_validator = cross_validator
        

    def fit(self, X ,y) :
        """ Builds a random forest classifier by TLAC (top label as class).
        Also stores the training instance that arrives at each node.
         
        Parameters
        ---------
        X : np.ndarray of shape (n_samples, n_features)
            Input array of descriptors.
        y : np.ndarray of shape (n_samples, n_labels)
            Output array of continuous yield values.
        """
        # Applies TLAC to y array
        tlac_y = np.argmax(y, axis=1)
        order = np.argsort(y, axis=1)
        y_rank = np.ones_like(order)*order.shape[1] - np.argsort(order, axis=1) 

        if self.cross_validator is not None :
            self.cross_validator.fit(X, tlac_y)
            model = self.cross_validator.best_estimator_
        else :
            model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=8, 
                criterion="log_loss", 
                max_features="log2", 
                n_jobs=-1,
                random_state=42
            )
            model.fit(X, tlac_y)
        self.model = model
        self.n_labels = y.shape[1]

        borda_count_by_tree_and_leaf = {}
        # Gets what training instances arrive at which node
        X_leaves = model.apply(X)
        for i in range(X_leaves.shape[1]) :
            borda_count_by_tree_and_leaf.update({i:{}})
            for leaf_num in np.unique(X_leaves[:,i]) :
                inds = np.where(X_leaves[:,i] == leaf_num)[0]
                borda_count = borda(np.expand_dims(y_rank[inds,:].T, axis=0))
                borda_count_by_tree_and_leaf[i].update({leaf_num:borda_count})
 
        self.borda_by_tree_and_leaf = borda_count_by_tree_and_leaf

        return self
    

    def predict(self, X) :
        """ Builds a ranking for the case where labels are fully labeled. 
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
        collected_borda_array = np.zeros((X_leaves.shape[0], self.n_labels, len(self.model.estimators_)))
        for i in range(X_leaves.shape[0]) :
            row = X_leaves[i,:]
            for j, leaf_num in enumerate(row) :
                collected_borda_array[i, :, j] = self.borda_by_tree_and_leaf[j][leaf_num]
        ranking_array = borda(collected_borda_array)
        return ranking_array
