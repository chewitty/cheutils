import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from hyperopt import tpe, hp, mix, anneal, rand
from hyperopt.pyll import scope
from cheutils.project_tree import save_excel
from cheutils.decorator_timer import track_duration
from cheutils.ml_utils.bayesian_search import HyperoptSearch, HyperoptSearchCV
from cheutils.ml_utils.model_options import get_params_grid, get_params_pounds
from cheutils.ml_utils.pipeline_details import show_pipeline
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties

LOGGER = LoguruWrapper().get_logger()
APP_PROPS = AppProperties()
prop_key = 'project.models.supported'
MODELS_SUPPORTED = APP_PROPS.get_dict_properties(prop_key)
assert (MODELS_SUPPORTED is not None), 'Models supported must be specified'
LOGGER.info('Models supported = {}', MODELS_SUPPORTED)
N_JOBS = int(APP_PROPS.get('model.active.n_jobs'))
# the model option selected as default
MODEL_OPTION = APP_PROPS.get('model.active.model_option')
# number of iterations or parameters to sample
N_ITERS = int(APP_PROPS.get('model.n_iters.to_sample'))
# number of hyperopt trials to iterate over
N_TRIALS = int(APP_PROPS.get('model.n_trials.to_sample'))
# max number of parameters to create for narrower param_grip - which defines how finely discretized the grid is
CONFIGURED_NUM_PARAMS = int(APP_PROPS.get('model.optimal.grid_resolution'))
# determine optimal number of parameters automatically, using the range specified or configured grid as the boundary
USE_OPTIMAL_GRID_RESOL = APP_PROPS.get_bol('model.grid_resolution.find_optimal')
NUM_PARAMS_RANGE = [int(APP_PROPS.get_dict_properties('model.num_params.sample_range').get('start')),
                       int(APP_PROPS.get_dict_properties('model.num_params.sample_range').get('end'))]
OPTIMAL_PARAMS_CV = APP_PROPS.get_bol('model.num_params.find_optimal.cv')
OPTIMAL_PARAMS_CV = False if OPTIMAL_PARAMS_CV is None else OPTIMAL_PARAMS_CV
# the cross_validation scoring metric
SCORING = APP_PROPS.get('model.cross_val.scoring')
CV = int(APP_PROPS.get('model.cross_val.num_folds'))
RANDOM_SEED = int(APP_PROPS.get('model.random_seed'))
# the hyperopt trial timeout
TRIAL_TIMEOUT = int(APP_PROPS.get('model.trial_timeout'))
# whether grid search is on
GRID_SEARCH_ON = APP_PROPS.get_bol('model.tuning.grid_search.on')
# cache narrower parameter grid, keyed by num_params and scaling_factor
NARROW_PARAM_GRIDS = {}
# cache best preliminary params, keyed by num_params
BEST_PARAM_GRIDS = {}
# cache optimal num_params, keyed by model option
OPTIMAL_NUM_PARAMS = {}
# Hyperopt algorithms
SUPPORTED_ALGOS = {'rand.suggest': rand.suggest, 'tpe.suggest': tpe.suggest, 'anneal.suggest': anneal.suggest}
CONFIG_ALGOS = APP_PROPS.get_dict_properties('model.hyperopt.algos')
p_suggest = []
if CONFIG_ALGOS is not None:
    for key, value in CONFIG_ALGOS.items():
        algo = SUPPORTED_ALGOS.get(key)
        if algo is not None:
            p_suggest.append((value, SUPPORTED_ALGOS.get(key)))
HYPEROPT_ALGOS = partial(mix.suggest, p_suggest=p_suggest)

@track_duration(name='fit')
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
    LOGGER.debug('Shape of dataset available for predictions {}, {}', X_pred.shape, y_pred.shape)
    return X_pred, y_pred

@track_duration(name='promising_params_grid')
def promising_params_grid(pipeline: Pipeline, X, y, grid_resolution: int=None, prefix: str = None, 
                       random_state: int=None, **kwargs):
    """
    Perform phase 1 of the coarse-to-fine hyperparameter tuning consisting of a coarse search using RandomizedCV
    to identify a promising in the hyperparameter space where the optimal values are likely to be found
    :param pipeline: estimator or pipeline instance with estimator
    :type pipeline:
    :param X: pandas DataFrame or numpy array
    :type X:
    :param y: pandas Series or numpy array
    :type y:
    :param grid_resolution: the grid resolution or maximum number of values per parameter
    :param prefix: default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param random_state: random seed for reproducibility
    :param kwargs:
    :type kwargs:
    :return: tuple -e.g., (best_estimator_, best_score_, best_params_, cv_results_)
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    if random_state is None:
        random_state = RANDOM_SEED
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    # phase 1: Coarse search
    params_grid = get_params_grid(MODEL_OPTION, prefix=prefix)
    LOGGER.debug('Configured hyperparameters = \n{}', params_grid)
    num_params = CONFIGURED_NUM_PARAMS if (grid_resolution is None) else grid_resolution
    best_params = BEST_PARAM_GRIDS.get(num_params)
    if best_params is None:
        search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                       scoring=SCORING, cv=CV, n_iter=N_ITERS, n_jobs=N_JOBS,
                                       random_state=random_state, verbose=2, error_score="raise", )
        if name is not None:
            show_pipeline(search_cv, name=name, save_to_file=True)
        else:
            show_pipeline(search_cv)
        search_cv.fit(X, y)
        LOGGER.debug('Preliminary best estimator = \n{}',
                      (search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_))
        best_params = search_cv.best_params_
        BEST_PARAM_GRIDS[num_params] = best_params
    return best_params

@track_duration(name='params_optimization')
def params_optimization(pipeline: Pipeline, X, y, promising_params_grid: dict, with_narrower_grid: bool = False,
                fine_search: str = 'hyperoptcv', scaling_factor: float = 0.20, grid_resolution: int=None, prefix: str = None,
                random_state: int=None, mlflow_log: bool=False, **kwargs):
    """
    Perform a fine hyperparameter optimization or tuning consisting of a fine search using bayesian optimization
    for a more detailed search within the narrower hyperparameter space to fine the best possible
    hyperparameter combination
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
    :param prefix: default is None; but could be estimator name in pipeline or pipeline instance - e.g., "main_model"
    :param random_state: random seed for reproducibility
    :param mlflow_log: if mlflow logging should be enabled - only valid for "hyperoptcv"
    :param kwargs:
    :type kwargs:
    :return: tuple -e.g., (best_estimator_, best_score_, best_params_, cv_results_)
    :rtype:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    if random_state is None:
        random_state = RANDOM_SEED
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    LOGGER.debug('Promising hyperparameters = \n{}', str(promising_params_grid))
    # get the parameter boundaries from the range specified in properties file
    params_bounds = get_params_pounds(MODEL_OPTION, prefix=prefix)
    # fetch promising params grdi from cache if possible
    num_params = CONFIGURED_NUM_PARAMS if (grid_resolution is None) else grid_resolution
    best_params = BEST_PARAM_GRIDS.get(num_params) if promising_params_grid is None else promising_params_grid
    # fetch narrow params grid from cache if possible
    params_cache_key = str(num_params) + '_' + str(np.round(scaling_factor, 2)).replace('.', '_')
    narrow_param_grid = NARROW_PARAM_GRIDS.get(params_cache_key) if with_narrower_grid else get_params_grid(MODEL_OPTION, prefix=prefix)
    # phase 2: perform finer search
    # generate narrow grid as required
    if with_narrower_grid & (narrow_param_grid is None):
        narrow_param_grid = get_narrow_param_grid(best_params, num_params, scaling_factor=scaling_factor,
                                                  params_bounds=params_bounds)
    LOGGER.debug('Narrower hyperparameters = \n{}', narrow_param_grid)
    search_cv = None
    if USE_OPTIMAL_GRID_RESOL:
        num_params = get_optimal_grid_resolution(pipeline, X, y, search_space=narrow_param_grid, params_bounds=params_bounds,
                                                 fine_search=fine_search, random_state=random_state)
    if 'hyperoptsk' == fine_search:
        search_cv = HyperoptSearch(params_space=__parse_params(narrow_param_grid,
                                                               num_params=num_params,
                                                               params_bounds=params_bounds,
                                                               fine_search=fine_search,
                                                               random_state=random_state),
                                   model_option=MODEL_OPTION, max_evals=N_TRIALS, algo=HYPEROPT_ALGOS, cv=CV,
                                   trial_timeout=TRIAL_TIMEOUT, random_state=random_state)
    elif "hyperoptcv" == fine_search:
        search_cv = HyperoptSearchCV(estimator=pipeline, params_space=__parse_params(narrow_param_grid,
                                                                                     num_params=num_params,
                                                                                     params_bounds=params_bounds,
                                                                                     fine_search=fine_search,
                                                                                     random_state=random_state),
                                     model_option=MODEL_OPTION, cv=CV, scoring=SCORING, algo=HYPEROPT_ALGOS,
                                     max_evals=N_TRIALS, n_jobs=N_JOBS, mlflow_log=mlflow_log,
                                     trial_timeout=TRIAL_TIMEOUT, random_state=random_state)
    elif 'skoptimize' == fine_search:
        search_cv = BayesSearchCV(estimator=pipeline, search_spaces=__parse_params(narrow_param_grid,
                                                                                   num_params=num_params,
                                                                                   params_bounds=params_bounds,
                                                                                   fine_search=fine_search,
                                                                                   random_state=random_state),
                                  scoring=SCORING, cv=CV, n_iter=5, n_jobs=N_JOBS,
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
    if random_state is None:
        random_state = RANDOM_SEED
    num_params = OPTIMAL_NUM_PARAMS.get(MODEL_OPTION)
    with_cv = CV if OPTIMAL_PARAMS_CV else None
    if num_params is None:
        scores = []
        param_ids = range(NUM_PARAMS_RANGE[0], NUM_PARAMS_RANGE[1] + 1)
        for n_params in param_ids:
            finder = None
            if 'hyperoptsk' == fine_search:

                finder = HyperoptSearch(params_space=__parse_params(search_space,
                                                                    num_params=n_params,
                                                                    params_bounds=params_bounds,
                                                                    fine_search=fine_search,
                                                                    random_state=random_state),
                                       model_option=MODEL_OPTION, max_evals=10, algo=HYPEROPT_ALGOS, cv=with_cv,
                                       trial_timeout=TRIAL_TIMEOUT, random_state=random_state)
            elif 'hyperoptcv' == fine_search:
                finder = HyperoptSearchCV(estimator=pipeline, params_space=__parse_params(search_space,
                                                                                          num_params=n_params,
                                                                                          params_bounds=params_bounds,
                                                                                          fine_search=fine_search,
                                                                                          random_state=random_state),
                                          model_option=MODEL_OPTION, cv=with_cv, scoring=SCORING, algo=HYPEROPT_ALGOS,
                                          max_evals=10, n_jobs=N_JOBS,
                                          trial_timeout=TRIAL_TIMEOUT, random_state=random_state)
            elif 'skoptimize' == fine_search:
                finder = BayesSearchCV(estimator=pipeline, search_spaces=__parse_params(search_space,
                                                                                        num_params=n_params,
                                                                                        params_bounds=params_bounds,
                                                                                        fine_search=fine_search,
                                                                                        random_state=random_state),
                                       scoring=SCORING, cv=with_cv, n_iter=5, n_jobs=N_JOBS,
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
            OPTIMAL_NUM_PARAMS[MODEL_OPTION] = num_params
    LOGGER.debug('Optimal grid resolution = {}', num_params)
    return num_params

def get_narrow_param_grid(promising_params: dict, num_params:int, scaling_factor: float = 1.0, params_bounds: dict= None):
    """
    Returns a narrower hyperparameter space based on the best parameters from the coarse or random search and a scaling factor
    :param promising_params: the best or promising combination of hyperparameters obtained from a coarse or random search
    :type promising_params: dict
    :param num_params: the number that defines the granularity of the narrower hyperparameter space
    :param scaling_factor: scaling factor used to control how much the hyperparameter search space from the coarse search is narrowed
    :type scaling_factor:
    :param params_bounds: usually the configured parameters grid, which provides the widest bounds of the hyperparameter search space
    :return:
    :rtype:
    """
    params_bounds = {} if params_bounds is None else params_bounds
    params_cache_key = str(num_params) + '_' + str(np.round(scaling_factor, 2)).replace('.', '_')
    narrower_grid = NARROW_PARAM_GRIDS.get(params_cache_key)
    if narrower_grid is not None:
        LOGGER.debug('Reusing previously generated narrower hyperparameter grid ...')
        return narrower_grid
    num_steps = num_params
    if params_bounds is None:
        param_bounds = {}
    param_grid = {}
    for param, value in promising_params.items():
        bounds = params_bounds.get(param.split('__')[-1])
        if bounds is not None:
            min_val, max_val = bounds
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
    NARROW_PARAM_GRIDS[params_cache_key] = param_grid
    return param_grid

def __parse_params(default_grid: dict, params_bounds: dict=None, num_params: int=3, fine_search: str = 'hyperoptcv', random_state: int=100) -> dict:
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
        fudge_factor = 0.20  # in cases where the hyperparameter is a single value instead of a list of at least 2
        for key, value in default_grid.items():
            bounds = params_bounds.get(key.split('__')[-1])
            if bounds is not None:
                lbound, ubound = bounds
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
                        param_grid[key] = scope.int(hp.quniform(key, min(cur_range), max(cur_range), num_params))
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
            if isinstance(value[0], (int, np.integer)):
                param_grid[param] = int(np.mean(value))
            elif isinstance(value[0], float):
                param_grid[param] = np.mean(value)
            else:
                param_grid[param] = value[0]
        else:
            param_grid[param] = [value]
    return param_grid
