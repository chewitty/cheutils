from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from cheutils.ml_utils.model_options import get_regressor

class BayesianSearch(object):
    def __init__(self, param_grid: dict, model_option:str=None, **kwargs):
        self.param_grid = param_grid
        self.model_option = model_option
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.scoring = kwargs.get('scoring', 'neg_mean_squared_error')
        self.cv = kwargs.get('cv', 5)

    def fit(self, X, y=None, **kwargs):
        print('BayesianSearch: Fitting dataset, shape', X.shape, y.shape if y is not None else None)
        # Define the hyperparameter space
        params_space = {}
        for key, value in self.param_grid.items():
            if isinstance(value[0], int):
                params_space[key] = hp.quniform(key, value[0], value[-1], 1)
            elif isinstance(value[0], float):
                params_space[key] = hp.uniform(key, value[0], value[-1])
            elif isinstance(value[0], bool):
                params_space[key] = hp.choice(key, value)
            else:
                params_space[key] = hp.choice(key, value)

        # Define the objective function to minimize
        def objective(params):
            model = get_regressor(model_option=self.model_option)
            model.fit(X, y)
            y_pred = model.predict(X)
            score = mean_squared_error(y, y_pred)
            return {'loss': -score, 'status': STATUS_OK}
        # Perform the optimization
        trials = Trials()
        self.best_params_ = fmin(objective, params_space, algo=tpe.suggest, max_evals=100, trials=trials)
        print("Best set of hyperparameters: ", self.best_params_)
        # fit the model and predict
        self.best_estimator_ = get_regressor(model_option=self.model_option, **self.best_params_)
        self.cv_results_ = cross_val_score(self.best_estimator_, X, y, scoring=self.scoring, cv=self.cv,)
        self.best_score_ = self.cv_results_.mean()
        # fit estimator so it can be immediately used for predicting
        self.best_estimator_.fit(X, y)
        return self
