# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

from typing import List, Optional, Tuple, Union

from ConfigSpace import ConfigurationSpace
import numpy as np
import sklearn.gaussian_process.kernels

from autotune.optimizer.surrogate.base.base_model import AbstractModel
import autotune.optimizer.surrogate.base.gp_base_prior

from skopt.learning.gaussian_process.kernels import Kernel
from skopt.learning.gaussian_process import GaussianProcessRegressor


class BaseGP(AbstractModel):

    def __init__(
            self,
            configspace: ConfigurationSpace,
            types: List[int],
            bounds: List[Tuple[float, float]],
            seed: int,
            kernel: Kernel,
            instance_features: Optional[np.ndarray] = None,
            pca_components: Optional[int] = None,
    ):
        """
        Abstract base class for all Gaussian process models.
        """
        super().__init__(
            types=types,
            bounds=bounds,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.configspace = configspace
        self.rng = np.random.RandomState(42)
        self.kernel = kernel
        self.gp = self._get_gp()
        self.seed = 42

    def _get_gp(self) -> GaussianProcessRegressor:
        raise NotImplementedError()

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean unit standard deviation.

        Parameters
        ----------
        y : np.ndarray
            Targets for the Gaussian process

        Returns
        -------
        np.ndarray
        """
        self.mean_y_ = np.mean(y)
        self.std_y_ = np.std(y)
        if self.std_y_ == 0:
            self.std_y_ = 1
        return (y - self.mean_y_) / self.std_y_

    def _untransform_y(
            self,
            y: np.ndarray,
            var: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform zeromean unit standard deviation data into the regular space.

        This function should be used after a prediction with the Gaussian process which was trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on Tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_ ** 2
            return y, var
        return y

    def _get_all_priors(
            self,
            add_bound_priors: bool = True,
            add_soft_bounds: bool = False,
    ) -> List[List[autotune.optimizer.surrogate.base.gp_base_prior.Prior]]:
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        # to_visit.append(self.gp.kernel.k1)
        # to_visit.append(self.gp.kernel.k2)
        to_visit.append(self.gp.kernel)  # fix single kernel
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(
                                autotune.optimizer.surrogate.base.gp_base_prior.SoftTopHatPrior(
                                    lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng, exponent=2,
                                ))
                        else:
                            priors_for_hp.append(
                                autotune.optimizer.surrogate.base.gp_base_prior.TophatPrior(
                                    lower_bound=bounds[i][0], upper_bound=bounds[i][1], rng=self.rng,
                                ))
                    all_priors.append(priors_for_hp)
        return all_priors

    def _set_has_conditions(self) -> None:
        has_conditions = len(self.configspace.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param, sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X
