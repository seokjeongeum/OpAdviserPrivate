# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)


import logging
import typing

import numpy as np
from pyrfr import regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

from autotune.optimizer.surrogate.base.base_model import  AbstractModel
from autotune.utils.constants import N_TREES



class RandomForestWithContexts(AbstractModel):

    """Random forest that takes instance features into account.

    Attributes
    ----------
    rf_opts : regression.rf_opts
        Random forest hyperparameter
    n_points_per_tree : int
    rf : regression.binary_rss_forest
        Only available after training
    hypers: list
        List of random forest hyperparameters
    unlog_y: bool
    seed : int
    types : np.ndarray
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(self, types: np.ndarray,
                 bounds: typing.List[typing.Tuple[float, float]],
                 current_context: np.ndarray=None,
                 context_pca_components: int=5,
                 log_y: bool=False,
                 num_trees: int=N_TREES,
                 do_bootstrapping: bool=True,
                 n_points_per_tree: int=-1,
                 ratio_features: float=5. / 6.,
                 min_samples_split: int=3,
                 min_samples_leaf: int=3,
                 max_depth: int=2**20,
                 eps_purity: float=1e-8,
                 max_num_nodes: int=2**20,
                 seed: int=42,
                 **kwargs):
        """
        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        log_y: bool
            y values (passed to this RF) are expected to be log(y) transformed;
            this will be considered during predicting
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(types, bounds, **kwargs)
        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

        self.current_context = current_context
        self.context_pca_components = context_pca_components
        self.contetx_n_feats = current_context.flatten().shape[0]
        if self.context_pca_components and self.contetx_n_feats > self.context_pca_components:
            self.context_pca = PCA(n_components=self.context_pca_components)
            self.context_scaler = MinMaxScaler()
            self.context_len = self.context_pca_components
            self.logger.info("Use PCA for context, convert dimension {} to {}".format(self.contetx_n_feats, self.context_pca_components))
        else:
            self.context_pca = None
            self.context_scaler = None
            self.context_len = self.contetx_n_feats

        self.types = np.append(self.types, np.zeros(self.context_len))
        self.bounds = np.vstack((self.bounds, (np.array([[0.0,1.1]]).repeat(self.context_len, axis=0))))
        self.log_y = log_y
        self.rng = regression.default_random_engine(42)


        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else \
            max(1, int(types.shape[0] * ratio_features))
        self.rf_opts.tree_opts.max_features = max_features
        self.rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self.rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.tree_opts.max_depth = max_depth
        self.rf_opts.tree_opts.epsilon_purity = eps_purity
        self.rf_opts.tree_opts.max_num_nodes = max_num_nodes
        self.rf_opts.compute_law_of_total_variance = False

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, 42]
        self.seed = 42



    def _train(self, X: np.ndarray, y: np.ndarray):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """
        contexts = self.contexts
        if self.context_pca and X.shape[0] > self.context_pca.n_components:
            # scale features
            X_feats = self.context_scaler.fit_transform(contexts)
            X_feats = np.nan_to_num(contexts)  # if features with max == min
            # PCA
            X_feats = self.context_pca.fit_transform(contexts)
            self.X = np.hstack((X, X_feats))
            '''if hasattr(self, "types"):
                # for RF, adapt types list
                # if X_feats.shape[0] < self.pca, X_feats.shape[1] ==
                # X_feats.shape[0]
                self.types = np.array(
                    np.hstack((self.types[:self.n_params], np.zeros((X_feats.shape[1])))),
                    dtype=np.uint,
                )'''
        else:
            self.X = np.hstack((X, contexts))

        self.y = y.flatten()

        if self.n_points_per_tree <= 0:
            self.rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            self.rf_opts.num_data_points_per_tree = self.n_points_per_tree
        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self._init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)
        return self

    def _init_data_container(self, X: np.ndarray, y: np.ndarray):
        """Fills a pyrfr default data container, s.t. the forest knows
        categoricals and bounds for continous data

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values

        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data

    def _predict(self, X: np.ndarray, current_context: np.ndarray=None) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' % (self.types.shape[0], X.shape[1]))

        if current_context is None:
            current_context = self.current_context
        else:
            self.current_context = current_context

        if self.context_pca:
            try:
                X_feats = current_context.reshape(1, -1).repeat(X.shape[0], axis=0)
                X_feats = self.context_scaler.transform(X_feats)
                X_feats = self.context_pca.transform(X_feats)
                self.logger.debug("Convert {} t0 {}".format(current_context, X_feats[0]))
                X = np.hstack((X, X_feats))
            except NotFittedError:
                pass  # PCA not fitted if only one training sample
        else:
            X = np.hstack((X, current_context.reshape(1, -1).repeat(X.shape[0], axis=0)))

        means, vars_ = [], []
        for row_X in X:
            if self.log_y:
                preds_per_tree = self.rf.all_leaf_values(row_X)
                means_per_tree = []
                for preds in preds_per_tree:
                    # within one tree, we want to use the
                    # arithmetic mean and not the geometric mean
                    means_per_tree.append(np.log(np.mean(np.exp(preds))))
                mean = np.mean(means_per_tree)
                var = np.var(means_per_tree) # variance over trees as uncertainty estimate
            else:
                mean, var = self.rf.predict_mean_var(row_X)
            means.append(mean)
            vars_.append(var)
        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized_over_instances(self, X: np.ndarray, current_context: np.ndarray=None):

        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if self.instance_features is None or \
                len(self.instance_features) == 0:

            mean, var = self.predict(X)
            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.bounds.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.bounds.shape[0],
                              X.shape[1]))

        if current_context is None:
            current_context = self.current_context
        else:
            self.current_context = current_context

        if self.context_pca:
            try:
                X_feats = current_context.reshape(1, -1).repeat(X.shape[0], axis=0)
                X_feats = self.context_scaler.transform(X_feats)
                X_feats = self.context_pca.transform(X_feats)
                X = np.hstack((X, X_feats))
            except NotFittedError:
                pass  # PCA not fitted if only one training sample
        else:
            X = np.hstack((X, current_context.reshape(1, -1).repeat(X.shape[0], axis=0)))

        mean = np.zeros(X.shape[0])
        var = np.zeros(X.shape[0])
        for i, x in enumerate(X):

            # marginalize over instances
            # 1. get all leaf values for each tree
            preds_trees = [[] for i in range(self.rf_opts.num_trees)]

            for feat in self.instance_features:
                x_ = np.concatenate([x, feat])
                preds_per_tree = self.rf.all_leaf_values(x_)
                for tree_id, preds in enumerate(preds_per_tree):
                    preds_trees[tree_id] += preds

            # 2. average in each tree
            for tree_id in range(self.rf_opts.num_trees):
                if self.log_y:
                    preds_trees[tree_id] = \
                        np.log(np.mean(np.exp(preds_trees[tree_id])))
                else:
                    preds_trees[tree_id] = np.mean(preds_trees[tree_id])

            # 3. compute statistics across trees
            mean_x = np.mean(preds_trees)
            var_x = np.var(preds_trees)
            if var_x < self.var_threshold:
                var_x = self.var_threshold

            var[i] = var_x
            mean[i] = mean_x

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var
