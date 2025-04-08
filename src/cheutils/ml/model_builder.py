import numpy as np
import pandas as pd
import requests
import mlflow
from functools import partial

from sklearn.base import is_classifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from hyperopt import tpe, hp, mix, anneal, rand
from hyperopt.pyll import scope
from cheutils.common_utils import safe_copy
from cheutils.project_tree import save_excel
from cheutils.decorator_timer import track_duration
from cheutils.ml.bayesian_search import HyperoptSearch, HyperoptSearchCV
from cheutils.ml.model_support import get_params_grid, get_params_pounds
from cheutils.ml.pipeline_details import show_pipeline
from cheutils.loggers import LoguruWrapper
from cheutils.sqlite_util import (save_param_grid_to_sqlite_db, get_param_grid_from_sqlite_db,
                                  save_narrow_grid_to_sqlite_db, get_narrow_grid_from_sqlite_db,
                                  save_promising_interactions_to_sqlite_db, get_promising_interactions_from_sqlite_db)
from cheutils.properties_util import AppProperties
from cheutils.ml.model_properties import ModelProperties
from typing import cast

LOGGER = LoguruWrapper().get_logger()

# cache optimal num_params, keyed by model option
OPTIMAL_GRID_RES = {}
# Supported hyperopt algorithms
SUPPORTED_ALGOS = {'rand.suggest': rand.suggest, 'tpe.suggest': tpe.suggest, 'anneal.suggest': anneal.suggest}

def recreate_labels(pred_probas: pd.Series, desired_thres: float, class_labels: list=None):
    """
    Recreate class labels for the specified decision threshold.
    :param pred_probas:
    :param desired_thres:
    :param class_labels:
    :return:
    """
    if class_labels is None:
        class_labels = ['True', 'False']
    new_labels = (pred_probas >= desired_thres)
    new_labels = new_labels.astype(int)
    new_labels = new_labels.apply(lambda x: class_labels[0] if x==1 else class_labels[1])
    return new_labels

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
    X_pred = safe_copy(X)
    y_pred = safe_copy(y)
    X_pred.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)
    null_rows = X_pred.isna().any(axis=1)
    X_pred.dropna(inplace=True)
    y_pred = y_pred[~null_rows]
    LOGGER.debug('Shape of dataset available for predictions {}, {}', X_pred.shape, y_pred.shape)
    return X_pred, y_pred

@track_duration(name='promising_params_grid')
def promising_params_grid(pipeline: Pipeline, X, y, grid_resolution: int=None, prefix: str = None, 
                       random_state: int=None, **kwargs):
    """
    Perform phase 1 of the coarse-to-fine hyperparameter search consisting of a coarse search using RandomizedCV
    to identify a set of promising hyperparameters in the search space, where the optimal values are likely to be found
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas DataFrame or numpy array
    :type X:
    :param y: pandas Series or numpy array
    :type y:
    :param grid_resolution: the grid resolution or maximum number of values per parameter
    :param prefix: any model prefix - the default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param random_state: random seed for reproducibility
    :param kwargs:
    :type kwargs:
    :return: dictionary of promising hyperparameters
    :rtype: dict
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    assert pipeline is not None, "A valid pipeline instance expected"
    if random_state is None:
        random_state = __model_handler.get_random_seed()
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    # phase 1: Coarse search
    params_grid = get_params_grid(__model_handler.get_model_option(), prefix=prefix)
    LOGGER.debug('Configured hyperparameters = \n{}', params_grid)
    num_params = __model_handler.get_grid_resolution() if (grid_resolution is None) else grid_resolution
    # attempt to fetch promising grid from SQLite DB
    best_params = get_param_grid_from_sqlite_db(grid_resolution=num_params, grid_size=len(params_grid),
                                                model_option=__model_handler.get_model_option(), model_prefix=prefix,
                                                tb_name=__model_handler.get_model_option())
    if best_params is None:
        search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                       scoring=__model_handler.get_cross_val_scoring(), cv=__model_handler.get_cross_val_num_folds(), n_iter=__model_handler.get_n_iters(), n_jobs=__model_handler.get_n_jobs(),
                                       random_state=random_state, verbose=2, error_score="raise", )
        if name is not None:
            show_pipeline(search_cv, name=name, save_to_file=True)
        else:
            show_pipeline(search_cv)
        search_cv.fit(X, y)
        LOGGER.debug('Promising params_grid search results = \n{}',
                      (search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_))
        best_params = search_cv.best_params_
        # cache the promising grid to SQLite
        save_param_grid_to_sqlite_db(param_grid=best_params, model_prefix=prefix, grid_resolution=num_params,
                                     grid_size=len(params_grid), tb_name=__model_handler.get_model_option(), )
    return best_params

@track_duration(name='params_optimization')
def params_optimization(pipeline: Pipeline, X, y, promising_params_grid: dict,
                        with_narrower_grid: bool = False,
                        fine_search: str = 'hyperoptcv', scaling_factor: float = 0.20, grid_resolution: int=None,
                        prefix: str = None, cv: int=None, random_state: int=None, mlflow_exp: dict= None, **kwargs):
    """
    Perform a fine hyperparameter optimization or search - a fine search using bayesian optimization
    for a more detailed search within the narrower, promising hyperparameter space to identify the optimal
    hyperparameter combination.
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas DataFrame or numpy array
    :type X:
    :param y: pandas Series or numpy array
    :type y:
    :param promising_params_grid: a previously generated promising parameter grid or configured default grid
    :param with_narrower_grid: run the step 1 random search if True and not otherwise
    :param fine_search: the default is "hyperopt" but other options include "random", "grid" and "skoptimize", for the second phase
    :param scaling_factor: the scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param grid_resolution: the grid resolution or maximum number of values per parameter
    :param prefix: model prefix - the default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param cv: cross-validation strategy or number of folds; if unspecified attempts to used any configured strategy or number of folds
    :param random_state: random seed for reproducibility
    :param mlflow_exp: dict such as {'log': False, 'uri': None} indicating if this is part of a Mlflow experiment in which logging should be enabled - BUT only valid for "hyperoptcv"
    :param kwargs:
    :type kwargs:
    :return: tuple -e.g., (best_estimator_, best_score_, best_params_, cv_results_)
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    cv_strategy = cv if cv is not None else __model_handler.get_cross_val_num_folds()
    if mlflow_exp is not None:
        if mlflow_exp.get('log') is True:
            LOGGER.warning('Parameter optimization as part of Mlflow experiment run: \n')
            LOGGER.warning('Make sure that you have set the MLFLOW_TRACKING_URI environment variable')
            LOGGER.warning('Alternatively, make sure that you have called  appropriately')
            mlflow_uri = mlflow_exp.get('uri')
            if mlflow_uri is None:
                mlflow_uri = 'http://localhost:8080'
            mlflow.set_tracking_uri(uri=mlflow_uri)
            # check the version endpoint of mlflow server is running
            response = requests.get(mlflow_uri + '/version')
            LOGGER.info('Remote tracking server version: {}', mlflow.__version__)
            LOGGER.info('Client-side version of MLflow: {}', response.text)
            assert response.status_code == 200, 'The remote MLflow tracking server must be running ...'
            if not (response.text == mlflow.__version__):
                LOGGER.warning('The client-side version of MLflow is not aligned with the remote tracking server')
    if random_state is None:
        random_state = __model_handler.get_random_seed()
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    LOGGER.debug('Promising hyperparameters = \n{}', str(promising_params_grid))
    # get the parameter boundaries from the range specified in properties file
    params_bounds = get_params_pounds(__model_handler.get_model_option(), prefix=prefix)
    # fetch promising params grid from cache if possible
    num_params = __model_handler.get_grid_resolution() if (grid_resolution is None) else grid_resolution
    best_params = promising_params_grid
    if best_params is None:
        best_params = get_param_grid_from_sqlite_db(grid_resolution=num_params, grid_size=len(params_bounds),
                                                    model_option=__model_handler.get_model_option(),
                                                    model_prefix=prefix, tb_name=__model_handler.get_model_option())
    # fetch narrow params grid from cache if possible
    narrow_param_grid = get_narrow_param_grid(best_params, num_params, scaling_factor=scaling_factor,
                                              params_bounds=params_bounds, model_prefix=prefix)
    narrow_param_grid = narrow_param_grid if with_narrower_grid else get_params_grid(__model_handler.get_model_option(), prefix=prefix)
    # phase 2: perform finer search
    search_cv = None
    if __model_handler.get_find_grid_resolution():
        num_params = get_optimal_grid_resolution(pipeline, X, y, search_space=narrow_param_grid, params_bounds=params_bounds,
                                                 fine_search=fine_search, random_state=random_state)
    if 'hyperoptsk' == fine_search:
        search_cv = HyperoptSearch(params_space=__parse_params(narrow_param_grid,
                                                               num_params=num_params,
                                                               params_bounds=params_bounds,
                                                               fine_search=fine_search,
                                                               random_state=random_state),
                                   model_option=__model_handler.get_model_option(), max_evals=__model_handler.get_n_trials(),
                                   algo=__get_hyperopt_algos(), cv=cv_strategy,
                                   trial_timeout=__model_handler.get_trial_timeout(), random_state=random_state)
    elif "hyperoptcv" == fine_search:
        search_cv = HyperoptSearchCV(estimator=pipeline, params_space=__parse_params(narrow_param_grid,
                                                                                     num_params=num_params,
                                                                                     params_bounds=params_bounds,
                                                                                     fine_search=fine_search,
                                                                                     random_state=random_state),
                                     cv=cv_strategy, scoring=__model_handler.get_cross_val_scoring(), algo=__get_hyperopt_algos(),
                                     max_evals=__model_handler.get_n_trials(), n_jobs=__model_handler.get_n_jobs(), mlflow_exp=mlflow_exp,
                                     trial_timeout=__model_handler.get_trial_timeout(), model_prefix=prefix, random_state=random_state)
    elif 'skoptimize' == fine_search:
        search_cv = BayesSearchCV(estimator=pipeline, search_spaces=__parse_params(narrow_param_grid,
                                                                                   num_params=num_params,
                                                                                   params_bounds=params_bounds,
                                                                                   fine_search=fine_search,
                                                                                   random_state=random_state),
                                  scoring=__model_handler.get_cross_val_scoring(), cv=cv_strategy, n_iter=__model_handler.get_n_iters(),
                                  n_jobs=__model_handler.get_n_jobs(),
                                  random_state=random_state, verbose=10, )
    else:
        LOGGER.error('Failure encountered: Unspecified or unsupported finer search type')
        raise KeyError('Unspecified or unsupported finer search type')

    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    # return the results accordingly
    return search_cv.best_estimator_, abs(search_cv.best_score_), search_cv.best_params_, search_cv.cv_results_

def get_optimal_grid_resolution(pipeline: Pipeline, X, y, search_space: dict, params_bounds=None, cache_value: bool = True,
                                fine_search: str = 'hyperoptcv', random_state: int=100, **kwargs):
    """
    Find the optimal grid resolution or maximum number of parameters needed to specify the given hyperparameter search space.
    :param pipeline:
    :param X:
    :type X:
    :param y:
    :type y:
    :param search_space: prevailing hyperparameter search space
    :type search_space:
    :param params_bounds: usually the configured parameters grid, which provides the widest bounds of the hyperparameter search space
    :type params_bounds:
    :param cache_value: cache the value so it may be reused subsequently
    :param fine_search: can either be 'skoptimize' or 'hyperoptcv'
    :param random_state:
    :type random_state:
    :return:
    :rtype:
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    if random_state is None:
        random_state = __model_handler.get_random_seed()
    num_params = OPTIMAL_GRID_RES.get(__model_handler.get_model_option())
    with_cv = __model_handler.get_cross_val_num_folds() if __model_handler.get_grid_resolution_wth_cv() else None
    if num_params is None:
        scores = []
        start = __model_handler.get_grid_resolution().get('start')
        end = __model_handler.get_grid_resolution().get('end')
        step = __model_handler.get_grid_resolution().get('step')
        param_ids = range(start, end, step)
        for n_params in param_ids:
            finder = None
            if 'hyperoptsk' == fine_search:

                finder = HyperoptSearch(params_space=__parse_params(search_space,
                                                                    num_params=n_params,
                                                                    params_bounds=params_bounds,
                                                                    fine_search=fine_search,
                                                                    random_state=random_state),
                                        model_option=__model_handler.get_model_option(), max_evals=10, algo=__get_hyperopt_algos(), cv=with_cv,
                                        trial_timeout=__model_handler.get_trial_timeout(), random_state=random_state)
            elif 'hyperoptcv' == fine_search:
                finder = HyperoptSearchCV(estimator=pipeline, params_space=__parse_params(search_space,
                                                                                          num_params=n_params,
                                                                                          params_bounds=params_bounds,
                                                                                          fine_search=fine_search,
                                                                                          random_state=random_state),
                                          cv=with_cv, scoring=__model_handler.get_cross_val_scoring(), algo=__get_hyperopt_algos(),
                                          max_evals=10, n_jobs=__model_handler.get_n_jobs(),
                                          trial_timeout=__model_handler.get_trial_timeout(), random_state=random_state)
            elif 'skoptimize' == fine_search:
                finder = BayesSearchCV(estimator=pipeline, search_spaces=__parse_params(search_space,
                                                                                        num_params=n_params,
                                                                                        params_bounds=params_bounds,
                                                                                        fine_search=fine_search,
                                                                                        random_state=random_state),
                                       scoring=__model_handler.get_cross_val_scoring(), cv=with_cv, n_iter=5, n_jobs=__model_handler.get_n_jobs(),
                                       random_state=random_state, verbose=10, )
            else:
                LOGGER.error('Failure encountered: Unspecified or unsupported finer search type')
                raise KeyError('Unspecified or unsupported finer search type')
            show_pipeline(finder)
            finder.fit(X, y)
            scores.append(finder.best_score_)
        num_params = param_ids[np.argmin(scores)]
        opt_params_df = pd.DataFrame({'num_params': param_ids, 'score': scores})
        filename = 'optimal_num_params.xlsx'
        save_excel(opt_params_df, file_name=filename)
        if cache_value:
            OPTIMAL_GRID_RES[__model_handler.get_model_option()] = num_params
    LOGGER.debug('Optimal grid resolution = {}', num_params)
    return num_params

def get_narrow_param_grid(promising_params: dict, num_params:int, scaling_factor: float = 1.0, params_bounds: dict= None, model_prefix: str=None):
    """
    Returns a generated hyperparameter grid based on a specified (usually, promising hyperparameter grid) from a coarse or random search and a scaling factor that defines the size of the space centered on the set promising hyperparameters.
    :param promising_params: the promising set of hyperparameters or seed hyperparameters obtained from a coarse or random search
    :type promising_params: dict
    :param num_params: the grid resolution or number that defines the granularity of the narrower hyperparameter space
    :param scaling_factor: scaling factor used to control how much the size of the hyperparameter search space around the set promising hyperparameters
    :type scaling_factor:
    :param params_bounds: usually the bounding values of the relevant configured parameters grid - defines the widest bounds of the hyperparameter search space
    :param model_prefix: prevailing model prefix
    :return:
    :rtype:
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    params_bounds = {} if params_bounds is None else params_bounds
    params_cache_key = str(num_params) + '_' + str(np.round(scaling_factor, 2)).replace('.', '_')
    model_option = __model_handler.get_model_option()
    narrower_grid = get_narrow_grid_from_sqlite_db(tb_name=model_option, cache_key=params_cache_key,
                                                   model_option=model_option, model_prefix=model_prefix)
    if narrower_grid is not None:
        LOGGER.debug('Reusing previously generated narrower hyperparameter grid ...')
        LOGGER.debug('Narrower hyperparameters = \n{}', narrower_grid)
        return narrower_grid
    num_steps = num_params
    if params_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in promising_params.items():
        bounds = params_bounds.get(param.split('__')[-1])
        if bounds is not None:
            min_val, max_val = bounds.get('start'), bounds.get('end')
            if isinstance(value, (int, np.integer)):
                min_val = int(min_val) if min_val is not None else value
                max_val = int(max_val) if max_val is not None else value
                std_dev = np.std([min_val, max_val])
                viable_span = int(std_dev * scaling_factor)
                cur_val = np.array([int(x) for x in
                                    np.linspace(max(value + viable_span, min_val), min(value - viable_span, max_val),
                                                num_steps)])
                cur_val = np.where(cur_val < 1, 1, cur_val)
                cur_val = list(set(np.where(cur_val > max_val, max_val, cur_val)))
                cur_val.sort()
                param_grid[param] = np.array(cur_val, dtype=int)
            elif isinstance(value, float):
                min_val = float(min_val) if min_val is not None else value
                max_val = float(max_val) if max_val is not None else value
                std_dev = np.std([min_val, max_val])
                viable_span = std_dev * scaling_factor
                cur_val = np.array([np.round(x, 3) for x in
                                    np.linspace(max(value + viable_span, min_val), min(value - viable_span, max_val),
                                                num_steps)])
                cur_val = np.where(cur_val < 0, 0, cur_val)
                cur_val = list(set(np.where(cur_val > max_val, max_val, cur_val)))
                cur_val.sort()
                param_grid[param] = np.array(cur_val, dtype=float)
        else:
            param_grid[param] = [value]
    #NARROW_PARAM_GRIDS[params_cache_key] = param_grid
    save_narrow_grid_to_sqlite_db(param_grid, tb_name=__model_handler.get_model_option(), cache_key=params_cache_key,
                                  model_prefix=model_prefix)
    LOGGER.debug('Narrower hyperparameters = \n{}', param_grid)
    return param_grid

@track_duration(name='promising_interactions')
def promising_interactions(pipeline: Pipeline, X, y, baseline_score: float, candidate_feats: list,
                           grid_resolution: int=None, prefix: str = None, tb_name: str=None,
                           random_state: int=None, **kwargs):
    """
    Perform a cross-validation search for promising feature interactions - identify which feature interactions improve
    the simple model performance (i.e., the simple model being the model with the obvious features derived from the training dataset).
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas DataFrame or numpy array
    :type X:
    :param y: pandas Series or numpy array
    :type y:
    :param baseline_score: the simple model current performance score
    :param candidate_feats: list of candidate feature columns provisionally selected for the feature interactions investigation.
    :param grid_resolution: the grid resolution or maximum number of values per parameter
    :param prefix: any model prefix - the default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param random_state: random seed for reproducibility
    :param tb_name: the name of the underlying sqlite table for saving any promising interactions
    :param kwargs:
    :type kwargs:
    :return: dictionary of promising hyperparameters
    :rtype: dict
    """
    assert pipeline is not None, 'Valid pipeline with an estimator required'
    assert baseline_score is not None, 'A baseline performance score is required'
    assert candidate_feats is not None and len(candidate_feats) > 1, 'A valid list of at least two features required'
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    assert pipeline is not None, "A valid pipeline instance expected"
    if random_state is None:
        random_state = __model_handler.get_random_seed()
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    promising_feats = get_promising_interactions_from_sqlite_db(tb_name=tb_name) if tb_name is not None else get_promising_interactions_from_sqlite_db()
    if promising_feats is not None and not(not promising_feats):
        return promising_feats
    params_grid = get_params_grid(__model_handler.get_model_option(), prefix=prefix)
    LOGGER.debug('Configured hyperparameters = \n{}', params_grid)
    num_params = __model_handler.get_grid_resolution() if (grid_resolution is None) else grid_resolution
    # attempt to fetch promising grid from SQLite DB
    best_params = get_param_grid_from_sqlite_db(grid_resolution=num_params, grid_size=len(params_grid),
                                                model_option=__model_handler.get_model_option(), model_prefix=prefix,
                                                tb_name=__model_handler.get_model_option())
    # use the configured grid
    best_params = params_grid if best_params is None else best_params
    train = safe_copy(X)
    promising_feats = []
    for i, c1 in enumerate(candidate_feats):
        for j, c2 in enumerate(candidate_feats[i + 1:]):
            n = f"{c1}_with_{c2}"
            train[n] = train[c1] * train[c2]
            cv_score = cross_val_score(pipeline, train, y, scoring=__model_handler.get_cross_val_scoring(),
                                       cv=__model_handler.get_cross_val_num_folds(),
                                       n_jobs=__model_handler.get_n_jobs(), error_score='raise')
            min_score = -np.nanmean(cv_score) if is_classifier(pipeline) else abs(np.nanmean(cv_score))
            LOGGER.debug('Mean score = {}\n', min_score)
            if min_score < baseline_score:
                promising_feats.append(n)
            train.drop(columns=[n], inplace=True)
    del train
    # cache the promising interaction features to SQLite
    if len(promising_feats) > 0:
        if tb_name is not None:
            save_promising_interactions_to_sqlite_db(promising_interactions=promising_feats, tb_name=tb_name)
        else:
            save_promising_interactions_to_sqlite_db(promising_interactions=promising_feats, )
    promising_feats = get_promising_interactions_from_sqlite_db(tb_name=tb_name) if tb_name is not None else get_promising_interactions_from_sqlite_db()
    return promising_feats

def __parse_params(default_grid: dict, params_bounds: dict=None, num_params: int=3, fine_search: str = 'hyperoptcv', random_state: int=100) -> dict:
    """
    Prepares the set of hyperparameters for the underlying estimator algorithm based on the specified hyperparameter search space.
    :param default_grid: source hyperparameter grid
    :type default_grid:
    :param params_bounds: bounding hyperparameter values
    :type params_bounds:
    :param num_params: specified grid resolution or maximum number of parameters in specifying each of the hyperparameter values in the search space
    :type num_params:
    :param fine_search: the Bayesian optimization algorithm choice - can be 'skoptimize', 'hyperoptcv', or 'hyperoptsk' - the default is 'hyperoptcv'
    :type fine_search:
    :param random_state:
    :type random_state:
    :return: dictionary of hyperparameters appropriate for the chosen estimator algorithm
    :rtype:
    """
    params_bounds = {} if params_bounds is None else params_bounds
    param_grid = {}
    if 'skoptimize' == fine_search:
        for param, value in default_grid.items():
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value[0], (int, np.integer)):
                    min_val, max_val = int(max(1, min(value))), int(max(value))
                    if min_val == max_val:
                        param_grid[param] = Categorical([max_val], transform='identity')
                    else:
                        param_grid[param] = Integer(min_val, max_val, prior='log-uniform')
                elif isinstance(value[0], float):
                    min_val, max_val = max(0.0001, min(value)), max(value)
                    param_grid[param] = Real(min_val, max_val, prior='log-uniform')
                else:
                    param_grid[param] = Categorical(value, transform='identity')
            else:
                param_grid[param] = [value]
        #LOGGER.debug('Scikit-optimize parameter space = \n{}', param_grid)
    elif ('hyperoptsk' == fine_search) | ('hyperoptcv' == fine_search):
        # Define the hyperparameter space
        fudge_factor = 0.40  # in cases where the hyperparameter is a single value instead of a list of at least 2
        for key, value in default_grid.items():
            bounds = params_bounds.get(key.split('__')[-1])
            if bounds is not None:
                lbound, ubound = bounds.get('start'), bounds.get('end')
                if isinstance(value[0], (int, np.integer)):
                    if len(value) == 1 | (value[0] == value[-1]):
                        min_val = max(int(value[0] * (1 - fudge_factor)), lbound)
                        max_val = min(int(value[0] * (1 + fudge_factor)), ubound)
                        cur_val = np.linspace(min_val, max_val, max(num_params, 2), dtype=int)
                        cur_val = np.sort(np.where(cur_val < 0, 0, cur_val))
                        cur_range = cur_val.tolist()
                        cur_range.sort()
                        param_grid[key] = scope.int(hp.quniform(key, min(cur_range), max(cur_range), num_params))
                    else:
                        min_val = max(int(value[0]), lbound)
                        max_val = min(int(value[-1]), ubound)
                        cur_val = np.linspace(min_val, max_val, max(num_params, 2), dtype=int)
                        cur_val = np.sort(np.where(cur_val < 0, 0, cur_val))
                        cur_range = cur_val.tolist()
                        cur_range.sort()
                        param_grid[key] = scope.int(hp.choice(key, cur_range))
                elif isinstance(value[0], float):
                    if len(value) == 1 | (value[0] == value[-1]):
                        min_val = np.exp(max(value[0] * (1 + fudge_factor), lbound))
                        max_val = np.exp(min(value[0] * (1 - fudge_factor), ubound))
                        param_grid[key] = hp.uniform(key, np.log(min_val), np.log(max_val))
                    else:
                        min_val = np.exp(max(value[0], lbound))
                        max_val = np.exp(min(value[-1], ubound))
                        param_grid[key] = hp.uniform(key, np.log(min_val), np.log(max_val))
                else:
                    pass
            else:
                if isinstance(value[0], (int, np.integer)):
                    cur_range = value
                    cur_range.sort()
                    param_grid[key] = hp.choice(key, cur_range)
                else:
                    param_grid[key] = hp.choice(key, value)
        #LOGGER.debug('Sample in hyperopt parameter space = \n{}', sample(param_grid))
    else:
        LOGGER.error('Parsed search space = \n{}', param_grid)
        raise ValueError(f'Missing implementation for search type = {fine_search}')
    return param_grid

def __get_seed_params(default_grid: dict, param_bounds=None):
    """
    Returns a generated hyperparameter grid with single values of each of the configured estimator hyperparameters.
    :param default_grid: the default parameters grid
    :type default_grid:
    :param param_bounds: bounding hyperparameter values
    :return: dictionary with single values for the hyperparameters in the default parameters grid
    :rtype: dict
    """
    if param_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in default_grid.items():
        param_bound = param_bounds.get(param.split('_')[-1])
        if isinstance(value, list):
            if isinstance(value[0], (int, np.integer)):
                param_grid[param] = int(np.mean(value))
            elif isinstance(value[0], float):
                param_grid[param] = np.mean(value)
            else:
                param_grid[param] = value[0]
        else:
            param_grid[param] = [value]
    return param_grid

def __get_hyperopt_algos():
    """
    Get the hyperopt algorithms to use for hyperparameter optimization.
    :return: an appropriate hyperopt algorithm(s) - based on what is configured (i.e., the 'model.hyperopt.algos' property)
    :rtype:
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    p_suggest = []
    if __model_handler.get_hyperopt_algos() is not None:
        for key, value in __model_handler.get_hyperopt_algos().items():
            algo = SUPPORTED_ALGOS.get(key)
            if algo is not None:
                p_suggest.append((value, SUPPORTED_ALGOS.get(key)))
    config_algos = partial(mix.suggest, p_suggest=p_suggest)
    return config_algos
