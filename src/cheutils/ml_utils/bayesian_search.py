from functools import partial

import numpy as np
from hyperopt import fmin, tpe, hp, mix, anneal, rand, space_eval
from hyperopt.pyll import scope
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from hpsklearn import HyperoptEstimator
from cheutils.common_base import CheutilsBase
from cheutils.ml_utils.model_options import get_hyperopt_regressor, get_regressor
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

HYPEROPT_ESTIMATORS = {'xgb_boost': 'xgboost_regression', 'lightgbm': 'lightgbm_regression',
                       'decision_tree': 'decision_tree_regressor', 'random_forest': 'random_forest_regressor',
                       'lasso': 'lasso', 'linear_regression': 'linear_regression', 'ridge': 'ridge',
                       'gradient_boosting': 'gradient_boosting_regressor'}

class HyperoptSearch(CheutilsBase):
    def __init__(self, model_option:str=None, max_evals: int=100, params_space: dict= None, loss_fn=mean_squared_error,
                 preprocessing: list=None, n_jobs: int=-1, algo=None, cv=None,
                 random_state: int=100, trial_timeout: int=None, **kwargs):
        super().__init__()
        self.model_option = model_option
        self.max_evals = max_evals
        self.cv = cv
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.random_state = random_state
        self.trial_timeout = trial_timeout
        self.n_jobs = n_jobs
        self.base_estimator_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.trials_ = None
        self.params_space = params_space if params_space is not None else {}
        self.loss_fn = mean_squared_error if loss_fn is None else loss_fn
        self.algo = algo

    def fit(self, X, y=None, **kwargs):
        LOGGER.debug('HyperoptSearch: Fitting dataset, shape {}, {}', X.shape, y.shape if y is not None else None)
        # Perform the optimization
        model_option = HYPEROPT_ESTIMATORS.get(self.model_option)
        self.best_estimator_ = HyperoptEstimator(regressor=get_hyperopt_regressor(model_option, **self.params_space),
                                                 preprocessing=self.preprocessing, loss_fn=self.loss_fn,
                                                 algo=self.algo, max_evals=self.max_evals,
                                                 trial_timeout=self.trial_timeout, refit=True, n_jobs=self.n_jobs,
                                                 seed=self.random_state, )
        self.best_estimator_.fit(X, y, n_folds=self.cv, cv_shuffle=True if self.cv is not None else False)
        self.base_estimator_ = self.best_estimator_.best_model().get('learner')
        self.best_score_ = min(self.best_estimator_.trials.losses())
        self.best_params_ = self.base_estimator_.get_params()
        self.trials_ = self.best_estimator_.trials
        self.cv_results_ = self.best_estimator_.trials
        LOGGER.debug('HyperoptSearch: Best hyperparameters  = \n{}', self.best_params_)
        return self

    def predict(self, X):
        assert X is not None, 'A valid X expected'
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        assert X is not None, 'A valid X expected'
        return self.base_estimator_.predict_proba(X)

class HyperoptSearchCV(CheutilsBase, BaseEstimator):
    def __init__(self, model_option:str=None, max_evals: int=100, algo=None,
                 cv=None, n_jobs: int=-1, params_space: dict= None, trial_timeout: int=60,
                 random_state: int=100, **kwargs):
        super().__init__()
        self.model_option = model_option
        self.max_evals = max_evals
        self.cv = cv
        self.n_jobs = n_jobs
        self.trial_timeout = trial_timeout
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = float('inf')
        self.cv_results_ = None
        self.scoring_ = kwargs.get('scoring', 'neg_mean_squared_error')
        self.params_space = params_space if params_space is not None else {}
        self.algo = algo
        self.X = None
        self.y = None

    def fit(self, X, y=None, **kwargs):
        LOGGER.debug('HyperoptSearchCV: Fitting dataset, shape {}, {}', X.shape, y.shape if y is not None else None)
        self.X = X
        self.y = y
        # Perform the optimization
        best_params = fmin(fn=self.__objective, space=self.params_space, algo=self.algo,
                           max_evals=self.max_evals, timeout=self.trial_timeout,
                           rstate=np.random.default_rng(self.random_state))
        self.best_params_ = space_eval(self.params_space, best_params)
        self.best_estimator_ = get_regressor(**self.__get_model_params(self.best_params_))
        self.best_estimator_.fit(X, y)
        LOGGER.debug('HyperoptSearchCV: Best hyperparameters  = \n{}', self.best_params_)
        return self

    def predict(self, X):
        assert X is not None, 'A valid X expected'
        assert self.best_estimator_ is not None, 'Model must be fitted before predict'
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        assert X is not None, 'A valid X expected'
        assert self.best_estimator_ is not None, 'Model must be fitted before predict_proba'
        return self.best_estimator_.predict_proba(X)

    def __objective(self, params):
        underlying_model = get_regressor(**self.__get_model_params(params))
        score = None
        if self.cv is not None:
            cv_score = cross_val_score(underlying_model, self.X, self.y, scoring=self.scoring_,
                                    cv=self.cv, n_jobs=self.n_jobs)
            min_score = abs(cv_score.mean())
            if min_score < self.best_score_:
                self.best_score_ = min_score
                self.cv_results_ = cv_score
        else:
            #no cross-validation
            underlying_model.fit(self.X, self.y)
            y_pred = underlying_model.predict(self.X)
            min_score = mean_squared_error(self.y, y_pred)
            if min_score < self.best_score_:
                self.best_score_ = min_score
        return min_score

    def __get_model_params(self, params):
        model_params = params.copy()
        model_params['model_option'] = self.model_option
        return model_params



