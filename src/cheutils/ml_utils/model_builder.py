import numpy as np
from cheutils.debugger import Debugger
from cheutils.decorator_debug import debug_func
from cheutils.decorator_timer import track_duration
from cheutils.ml_utils.pipeline_details import show_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

DBUGGER = Debugger()
@track_duration(name='fit')
@debug_func(enable_debug=True, prefix='fit')
def fit(pipeline: Pipeline, X, y, **kwargs):
    """
    Fit the model based on the pipeline
    :param pipeline:
    :param X:
    :param y:
    :return:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    assert y is not None, "A valid y expected"
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    if name is not None:
        show_pipeline(pipeline, name=name, save_to_file=True)
    else:
        show_pipeline(pipeline)
    pipeline.fit(X, y, **kwargs)

@track_duration(name='predict')
@debug_func(enable_debug=True, prefix='predict')
def predict(pipeline: Pipeline, X):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    return np.round(pipeline.predict(X),0).astype(int)

def predict_proba(pipeline: Pipeline, X):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    return pipeline.predict_proba(X)


def direct_predict(estimator, X):
    assert estimator is not None, "A valid estimator instance expected"
    assert X is not None, "A valid X expected"
    return estimator.predict(X)

def exclude_nulls(X, y):
    """
    Return dataset ready for predictions, scoring, and reporting - i.e., the prediction step does not need null values.
    :param X:
    :param y:
    :return:
    """
    assert X is not None, "A valid X expected"
    assert y is not None, "A valid y expected"
    # The prediction step does not need null values
    X_pred = X.copy(deep=True)
    y_pred = y.copy(deep=True)
    X_pred.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)
    null_rows = X_pred.isna().any(axis=1)
    X_pred.dropna(inplace=True)
    y_pred = y_pred[~null_rows]
    print("Shape of dataset available for predictions", X_pred.shape, y_pred.shape)
    return X_pred, y_pred

@track_duration(name='score')
@debug_func(enable_debug=True, prefix='score')
def score(y_true, y_pred, kind:str="mse"):
    assert y_true is not None, "A valid y_true expected"
    assert y_pred is not None, "A valid y_pred expected"
    if kind == "mse":
        mse_score = mean_squared_error(y_true, y_pred)
        return mse_score
    elif kind == "r2":
        r2score = r2_score(y_true, y_pred)
        return r2score
    else:
        raise ValueError("Score not yet implemented")
@track_duration(name='tune_model')
@debug_func(enable_debug=True, prefix='tune_model')
def tune_model(pipeline: Pipeline, X, y, params_grid: dict, cv, grid_search: bool = False,
               scoring: str = "neg_mean_squared_error", random_state=100, debug: bool = False, scan_params: bool=False, **kwargs):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert params_grid is not None, "A valid parameter grid (a dict) expected"
    assert (cv is not None), "A valid cv, either the number of folds or an instance of something like StratifiedKFold"
    n_iters = 1000
    n_jobs = -1
    DBUGGER.debug('Hyperparameters =', params_grid)
    search_cv = None
    if grid_search:
        if debug:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=2, error_score="raise",)
        else:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs,)
    else:
        if debug:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state, verbose=2, error_score="raise",)
        else:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state,)
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    if not scan_params:
        return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_
    else:
        return search_cv.cv_results_

@track_duration(name='coarse_fine_tune')
@debug_func(enable_debug=True, prefix='coarse_fine_tune')
def coarse_fine_tune(pipeline: Pipeline, X, y, params_grid: dict, cv, scaling_factor: float = 0.10,
                     scoring: str = "neg_mean_squared_error", random_state=100,
                     full_results: bool=False, param_bounds=None, max_params: int=5, **kwargs):
    """
    Perform a coarse-to-fine hyperparameter tuning consisting of two phases: a coarse search using RandomizedCV
    to identify a promising in the hyperparameter space where the optimal values are likely to be found; then,
    a fine search using another RandomizedSearchCV for a more detailed search within the narrower hyperparameter space
    to fine the best possible hyperparameter combination
    :param pipeline:
    :type pipeline:
    :param X:
    :type X:
    :param y:
    :type y:
    :param params_grid:
    :type params_grid:
    :param cv:
    :type cv:
    :param scaling_factor: the scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param scoring:
    :type scoring:
    :param random_state:
    :type random_state:
    :param full_results:
    :type full_results:
    :param param_bounds
    :param max_params: maximum number of parameter options for each hyperparameter in the second phase
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    assert params_grid is not None, "A valid parameter grid (a dict) expected"
    assert (cv is not None), "A valid cv, either the number of folds or an instance of something like StratifiedKFold"
    n_iters = 1000
    n_jobs = -1
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    # phase 1: Coarse search
    DBUGGER.debug('Hyperparameters =', params_grid)
    search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                    scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                    random_state=random_state, verbose=2, error_score="raise", )
    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    DBUGGER.debug('Preliminary best estimator =', (search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_))

    # phase 2: finer search
    narrow_param_grid = get_narrow_param_grid(search_cv.best_params_, scaling_factor=scaling_factor, param_bounds=param_bounds)
    DBUGGER.debug('Narrower hyperparameters =', narrow_param_grid)
    search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=narrow_param_grid,
                              scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs, verbose=2, error_score="raise", )
    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    # return the results accordingly
    if not full_results:
        return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_
    else:
        return search_cv.cv_results_

def get_narrow_param_grid(best_params: dict, scaling_factor: float=0.10, param_bounds=None, max_params: int=5):
    """
    Returns a narrower hyperparameter space based on the best parameters from the coarse search phase and a scaling factor
    :param best_params: the best combination of hyperparameters obtained from the coarse search phase
    :type best_params:
    :param scaling_factor: scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param param_bounds:
    :param max_params: maximum number of parameter options for each hyperparameter in the second phase
    :return:
    :rtype:
    """
    num_steps = max_params
    if param_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in best_params.items():
        param_bound = param_bounds.get(param.split('_')[-1])
        if isinstance(value, int):
            min_val = int(param_bound.get('lower')) if param_bound is not None else value
            max_val = int(param_bound.get('upper')) if param_bound is not None else value
            viable_span = int(((max_val - min_val + 1)/ 2)*scaling_factor)
            param_grid[param] = list(set([int(x) for x in np.linspace(max(value - viable_span, min_val), min(value + viable_span, max_val), num_steps)]))
        elif isinstance(value, float):
            min_val = float(param_bound.get('lower')) if param_bound is not None else value
            max_val = float(param_bound.get('upper')) if param_bound is not None else value
            viable_span = ((max_val - min_val + 1) / 2) * scaling_factor
            param_grid[param] = list(set([np.round(x, 3) for x in np.linspace(max(value - viable_span, min_val), min(value + viable_span, max_val), num_steps)]))
        else:
            param_grid[param] = [value]
    return param_grid

@track_duration(name='cv')
@debug_func(enable_debug=True, prefix='cv')
def cross_val_model(pipeline: Pipeline, X, y, scoring, cv=5, **fit_params):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert (cv is not None), "A valid cv, either the number of folds or an instance of something like StratifiedKFold"
    cv_scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, **fit_params)
    return cv_scores



