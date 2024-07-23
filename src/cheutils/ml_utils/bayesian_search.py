import numpy as np
from cheutils.debugger import Debugger
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numpy.random import PCG64
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from cheutils.ml_utils.model_options import get_regressor

DBUGGER = Debugger()
class BayesianSearch(object):
    def __init__(self, param_grid: dict, model_option:str=None, n_iters: int=100, random_state: int=100, **kwargs):
        self.param_grid = param_grid
        self.model_option = model_option
        self.n_iters = n_iters
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.scoring_ = kwargs.get('scoring', 'neg_mean_squared_error')
        self.cv_ = kwargs.get('cv', 5)
        self.trials_ = None

    def fit(self, X, y=None, **kwargs):
        DBUGGER.debug('BayesianSearch: Fitting dataset, shape', X.shape, y.shape if y is not None else None)
        # Define the hyperparameter space
        params_space = {}
        for key, value in self.param_grid.items():
            if isinstance(value[0], int):
                if len(value) == 1 | (value[0] == value[-1]):
                    params_space[key] = value
                else:
                    params_space[key] = hp.choice(key, np.arange(value[0], value[-1], dtype=int))
            elif isinstance(value[0], float):
                if len(value) == 1 | (value[0] == value[-1]):
                    params_space[key] = value
                else:
                    params_space[key] = hp.uniform(key, value[0], value[-1])
            elif isinstance(value[0], bool):
                params_space[key] = hp.choice(key, value)
            else:
                params_space[key] = hp.choice(key, value)

        # Perform the optimization
        trials = Trials()
        self.best_params_ = fmin(self.best_mse_by_cv, params_space, algo=tpe.suggest, max_evals=100,
                                 trials=trials, rstate=np.random.Generator(PCG64(self.random_state)))
        DBUGGER.debug("Best bayesian hyperparameters: ", self.best_params_)
        # fit the model and predict
        self.best_estimator_ = get_regressor(model_option=self.model_option, **self.best_params_, random_state=self.random_state)
        best_result = self.best_mse_by_cv(self.best_params_)
        self.best_score_ = best_result.get('loss')
        self.cv_results_ = best_result.get('cv_results')
        self.trials_ = trials
        # fit estimator so it can be immediately used for predicting
        self.best_estimator_.fit(X, y)
        return self

    def best_mse_by_cv(self, params, X=None, y=None):
        """
        The objective function - minimizes loss
        :param params:
        :type params:
        :param X:
        :type X:
        :param y:
        :type y:
        :return:
        :rtype:
        """
        model = get_regressor(model_option=self.model_option, **params, random_state=self.random_state)
        cv_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
        return {'loss': cv_scores.mean(), 'cv_results': cv_scores, 'status': STATUS_OK}
