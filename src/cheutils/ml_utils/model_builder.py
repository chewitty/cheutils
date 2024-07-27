import numpy as np
import pandas as pd
from cheutils.debugger import Debugger
from cheutils.decorator_debug import debug_func
from cheutils.decorator_timer import track_duration
from cheutils.ml_utils.bayesian_search import BayesianSearch
from cheutils.ml_utils.model_options import get_params_grid, get_params_pounds, get_params
from cheutils.ml_utils.model_options import get_regressor
from cheutils.ml_utils.pipeline_details import show_pipeline
from cheutils.ml_utils.visualize import plot_hyperparameter
from cheutils.properties_util import AppProperties
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

n_jobs = -1
APP_PROPS = AppProperties()
model_option = APP_PROPS.get('model.active.model_option')
# number of iterations or parameters to sample
n_iters = int(APP_PROPS.get('model.n_iters.to_sample'))
# how fine or max number of parameters to create for narrower param_grip
num_params = int(APP_PROPS.get('model.num_params.to_sample'))
scoring = APP_PROPS.get('model.cross_val.scoring')
cv = int(APP_PROPS.get('model.cross_val.num_folds'))
random_state = int(APP_PROPS.get('model.random_state'))
grid_search = APP_PROPS.get_bol('model.tuning.grid_search.on')
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
    """
    Do prediction based on the pipeline and the X.
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas.DataFrame or numpy.ndarray
    :type X:
    :return: pandas.Series or numpy.ndarray
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    return np.round(pipeline.predict(X), 0).astype(int)


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
def score(y_true, y_pred, kind: str = "mse"):
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


def eval_metric_by_params(model_option, X, y, prefix: str = None, metric: str = 'mean_test_score',
                          svg_file: bool = False):
    """
    Evaluate the performance metric vs configured hyperparameters for the current model option
    using RandomizedSearchCV to identify optimal tuning ranges. This is a kind of coarse-to-fine search to
    identify a narrower hyperparameter space for a possible subsequent GridSearch.
    :param model_option:
    :type model_option:
    :param X:
    :type X:
    :param y:
    :type y:
    :param prefix:
    :type prefix:
    :param metric:
    :type metric:
    :param svg_file:
    :type svg_file:
    :return:
    :rtype:
    """
    model = get_regressor(model_option=model_option)
    model_params = get_params(model_option=model_option, prefix=prefix)
    for param in model_params:
        param_name = param.split(prefix)[-1]
        cv_results = tune_model(model, X, y, model_option)
        cur_results = pd.DataFrame(cv_results[3])[list(cv_results[3].keys())]
        param_scores = cur_results[['param_' + param, metric]]
        param_scores.rename(columns={'param_' + param: param_name, }, inplace=True)
        param_scores[metric] = np.abs(param_scores[metric])
        save_file = model_option + '_' + param_name + '_range.svg'
        plot_hyperparameter(param_scores, metric_label='mean_test_score', param_label=param_name,
                            save_to_file=save_file if svg_file else None)


@track_duration(name='tune_model')
@debug_func(enable_debug=True, prefix='tune_model')
def tune_model(pipeline: Pipeline, X, y, model_option: str, prefix: str = None, debug: bool = False, **kwargs):
    assert pipeline is not None, "A valid pipeline instance expected"
    params_grid = get_params_grid(model_option, prefix=prefix)
    DBUGGER.debug('Hyperparameters =', params_grid)
    search_cv = None
    if grid_search:
        if debug:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=2, error_score="raise", )
        else:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs, )
    else:
        if debug:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state, verbose=2, error_score="raise", )
        else:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state, )
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_, search_cv.cv_results_


@track_duration(name='coarse_fine_tune')
@debug_func(enable_debug=True, prefix='coarse_fine_tune')
def coarse_fine_tune(pipeline: Pipeline, X, y, skip_phase_1: bool = False, fine_search: str = 'random',
                     scaling_factor: float = 1.0, prefix: str = None,
                     **kwargs):
    """
    Perform a coarse-to-fine hyperparameter tuning consisting of two phases: a coarse search using RandomizedCV
    to identify a promising in the hyperparameter space where the optimal values are likely to be found; then,
    a fine search using another RandomizedSearchCV for a more detailed search within the narrower hyperparameter space
    to fine the best possible hyperparameter combination
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas DataFrame or numpy array
    :type X:
    :param y: pandas Series or numpy array
    :type y:
    :param skip_phase_1: elect to skip phase 1 and directly proceed to phase 2
    :param fine_search: the default is "random" but other options include "grid" and "bayesian", for the second phase
    :param scaling_factor: the scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param prefix: default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param kwargs:
    :type kwargs:
    :return: tuple -e.g., (best_estimator_, best_score_, best_params_, cv_results_)
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    # phase 1: Coarse search
    params_grid = get_params_grid(model_option, prefix=prefix)
    DBUGGER.debug('Hyperparameters =', params_grid)
    search_cv = None
    if not skip_phase_1:
        search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                       scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                       random_state=random_state, verbose=2, error_score="raise", )
        if name is not None:
            show_pipeline(search_cv, name=name, save_to_file=True)
        else:
            show_pipeline(search_cv)
        search_cv.fit(X, y)
        DBUGGER.debug('Preliminary best estimator =',
                      (search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_))

    # phase 2: finer search
    params_bounds = get_params_pounds(model_option, prefix=prefix)
    narrow_param_grid = params_grid if skip_phase_1 else None
    if not skip_phase_1:
        narrow_param_grid = get_narrow_param_grid(search_cv.best_params_, scaling_factor=scaling_factor,
                                                  params_bounds=params_bounds)
    DBUGGER.debug('Narrower hyperparameters =', narrow_param_grid)
    if 'random' == fine_search:
        search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=narrow_param_grid,
                                       scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs, verbose=2,
                                       error_score="raise", )
    elif "grid" == fine_search:
        search_cv = GridSearchCV(estimator=pipeline, param_grid=narrow_param_grid,
                                 scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=2, error_score="raise", )
    elif "bayesian" == fine_search:
        search_cv = BayesianSearch(param_grid=narrow_param_grid, params_bounds=params_bounds,
                                   scaling_factor=scaling_factor, model_option=model_option, n_iters=n_iters,
                                   num_params=num_params, random_state=random_state)
    else:
        DBUGGER.debug('Failure encountered: Unspecified or unsupported finer search type')
        raise KeyError('Unspecified or unsupported finer search type')

    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    # return the results accordingly
    return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_, search_cv.cv_results_


def get_narrow_param_grid(best_params: dict, scaling_factor: float = 1.0, params_bounds=None):
    """
    Returns a narrower hyperparameter space based on the best parameters from the coarse search phase and a scaling factor
    :param best_params: the best combination of hyperparameters obtained from the coarse search phase
    :type best_params:
    :param scaling_factor: scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param params_bounds:
    :return:
    :rtype:
    """
    num_steps = num_params
    if params_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in best_params.items():
        bounds = params_bounds.get(param.split('__')[-1])
        if bounds is not None:
            min_val, max_val = bounds
            if isinstance(value, int):
                min_val = int(min_val) if min_val is not None else value
                max_val = int(max_val) if max_val is not None else value
                viable_span = int(((max_val - min_val) / 2) * scaling_factor)
                cur_val = np.array([int(x) for x in
                                    np.linspace(max(value + viable_span, min_val), min(value - viable_span, max_val),
                                                num_steps)])
                cur_val = np.where(cur_val < 1, 1, cur_val)
                cur_val = np.sort(np.where(cur_val > max_val, max_val, cur_val))
                param_grid[param] = list(set(cur_val.tolist()))
            elif isinstance(value, float):
                min_val = float(min_val) if min_val is not None else value
                max_val = float(max_val) if max_val is not None else value
                viable_span = ((max_val - min_val) / 2) * scaling_factor
                cur_val = np.array([np.round(x, 3) for x in
                                    np.linspace(max(value + viable_span, min_val), min(value - viable_span, max_val),
                                                num_steps)])
                cur_val = np.where(cur_val < 0, 0, cur_val)
                cur_val = np.sort(np.where(cur_val > max_val, max_val, cur_val))
                param_grid[param] = list(set(cur_val.tolist()))
        else:
            param_grid[param] = [value]
    return param_grid


def get_seed_params(default_grid: dict, param_bounds=None):
    """
    Returns a narrower hyperparameter space based on the best parameters from the coarse search phase and a scaling factor
    :param default_grid: the default parameters grid
    :type default_grid:
    :param param_bounds:
    :return:
    :rtype:
    """
    if param_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in default_grid.items():
        param_bound = param_bounds.get(param.split('_')[-1])
        if isinstance(value, list):
            if isinstance(value[0], int):
                param_grid[param] = int(np.mean(value))
            elif isinstance(value[0], float):
                param_grid[param] = np.mean(value)
            else:
                param_grid[param] = value[0]
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
