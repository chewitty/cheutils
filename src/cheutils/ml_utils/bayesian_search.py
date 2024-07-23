import numpy as np
from cheutils.debugger import Debugger
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from hpsklearn import HyperoptEstimator
from cheutils.ml_utils.model_options import get_hyperopt_regressor

DBUGGER = Debugger()
class BayesianSearch(object):
    def __init__(self, param_grid: dict, model_option:str=None, n_iters: int=100,
                 preprocessing: list=None, random_state: int=100, **kwargs):
        self.param_grid = param_grid
        self.model_option = model_option
        self.n_iters = n_iters
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.random_state = random_state
        self.base_estimator_ = None
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
                    params_space[key] = hp.choice(key, value)
                else:
                    params_space[key] = hp.choice(key, np.arange(value[0], value[-1], dtype=int))
            elif isinstance(value[0], float):
                if len(value) == 1 | (value[0] == value[-1]):
                    params_space[key] = hp.choice(key, value)
                else:
                    params_space[key] = hp.uniform(key, value[0], value[-1])
            elif isinstance(value[0], bool):
                params_space[key] = hp.choice(key, value)
            else:
                params_space[key] = hp.choice(key, value)

        # Perform the optimization
        self.best_estimator_ = HyperoptEstimator(regressor=get_hyperopt_regressor(self.model_option, **params_space),
                                                 preprocessing=self.preprocessing, loss_fn=mean_squared_error,
                                                 algo=tpe.suggest, max_evals=self.n_iters,
                                                 trial_timeout=60, refit=True, n_jobs=-1, seed=self.random_state,)
        self.best_estimator_.fit(X, y)
        self.base_estimator_ = self.best_estimator_.best_model().get('learner')
        self.best_score_ = self.best_estimator_.score(X, y)
        self.best_params_ = self.best_estimator_.get_params()
        self.trials_ = self.best_estimator_.trials
        DBUGGER.debug("Best bayesian hyperparameters: ", self.best_params_)
        return self

    def predict(self, X):
        assert X is not None, "A valid X expected"
        return self.base_estimator_.predict(X)


