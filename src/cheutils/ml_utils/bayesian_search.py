import numpy as np
from cheutils.debugger import Debugger
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_squared_error
from hpsklearn import HyperoptEstimator
from cheutils.ml_utils.model_options import get_hyperopt_regressor

DBUGGER = Debugger()
class BayesianSearch(object):
    def __init__(self, param_grid: dict, params_bounds: dict,
                 model_option:str=None, n_iters: int=100, num_params: int=5,
                 preprocessing: list=None, random_state: int=100, **kwargs):
        self.param_grid = param_grid
        self.params_bounds = params_bounds
        self.model_option = model_option
        self.n_iters = n_iters
        self.num_params = num_params
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
        self.params_space_ = {}

    def fit(self, X, y=None, **kwargs):
        DBUGGER.debug('BayesianSearch: Fitting dataset, shape', X.shape, y.shape if y is not None else None)
        # Define the hyperparameter space
        fudge_factor = 0.20 # in cases where the hyperparameter is a single value instead of a list of at least 2
        space_def = {}
        for key, value in self.param_grid.items():
            bounds = self.params_bounds.get(key.split('__')[-1])
            if bounds is not None:
                lbound, ubound = bounds
                if isinstance(value[0], int):
                    if len(value) == 1 | (value[0] == value[-1]):
                        min_val = max(int(value[0]*(1 - fudge_factor)), lbound)
                        max_val = min(int(value[0]*(1 + fudge_factor)), ubound)
                        cur_val = np.linspace(min_val, max_val, self.num_params, dtype=int)
                        cur_val = np.sort(np.where(cur_val < 0, 0, cur_val))
                        self.params_space_[key] = hp.choice(key, cur_val.tolist())
                        space_def[key] = cur_val.tolist()
                    else:
                        min_val = max(int(value[0]), lbound)
                        max_val = min(int(value[-1]), ubound)
                        cur_val = np.linspace(min_val, max_val, self.num_params, dtype=int)
                        cur_val = np.sort(np.where(cur_val < 0, 0, cur_val))
                        self.params_space_[key] = hp.choice(key, cur_val.tolist())
                        space_def[key] = cur_val.tolist()
                elif isinstance(value[0], float):
                    if len(value) == 1 | (value[0] == value[-1]):
                        min_val = max(value[0] * (1 + fudge_factor), lbound)
                        max_val = min(value[0] * (1 - fudge_factor), ubound)
                        self.params_space_[key] = hp.uniform(key, min_val, max_val)
                        space_def[key] = np.linspace(min_val, max_val, self.num_params, dtype=float)
                    else:
                        min_val = max(value[0], lbound)
                        max_val = min(value[-1], ubound)
                        self.params_space_[key] = hp.uniform(key, min_val, max_val)
                        space_def[key] = np.linspace(min_val, max_val, self.num_params, dtype=float)
                else:
                    pass
            else:
                self.params_space_[key] = hp.choice(key, value)
                space_def[key] = value
        DBUGGER.debug('BayesianSearch: Parameter space', space_def)
        # Perform the optimization
        self.best_estimator_ = HyperoptEstimator(regressor=get_hyperopt_regressor(self.model_option, **self.params_space_),
                                                 preprocessing=self.preprocessing, loss_fn=mean_squared_error,
                                                 algo=tpe.suggest, max_evals=self.n_iters,
                                                 trial_timeout=60, refit=True, n_jobs=-1, seed=self.random_state,)
        self.best_estimator_.fit(X, y)
        self.base_estimator_ = self.best_estimator_.best_model().get('learner')
        self.best_score_ = min(self.best_estimator_.trials.losses())
        self.best_params_ = self.best_estimator_.get_params()
        self.trials_ = self.best_estimator_.trials
        DBUGGER.debug("Best bayesian hyperparameters: ", self.best_params_)
        return self

    def predict(self, X):
        assert X is not None, "A valid X expected"
        return self.base_estimator_.predict(X)


