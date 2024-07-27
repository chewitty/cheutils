import numpy as np
from cheutils.debugger import Debugger
from cheutils.properties_util import AppProperties
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from hpsklearn import lasso, linear_regression, ridge, gradient_boosting_regressor
from hpsklearn import xgboost_regression, lightgbm_regression, decision_tree_regressor, random_forest_regressor

APP_PROPS = AppProperties()
DEBUGGER = Debugger()
def get_regressor(**model_params):
    """
    Gets a specified regressor configured with key 'model_option'.
    """
    cur_model_params = model_params.copy()
    model_option = None
    if 'model_option' in cur_model_params:
        model_option = cur_model_params.get('model_option')
        del cur_model_params['model_option']
    if 'params_grid_key' in cur_model_params:
        params_grid_key = cur_model_params.get('params_grid_key')
        del cur_model_params['params_grid_key']
    if 'lasso' == model_option:
        model = Lasso(**cur_model_params)
    elif 'linear' == model_option:
        model = LinearRegression(**cur_model_params)
    elif 'ridge' == model_option:
        model = Ridge(**cur_model_params)
    elif 'gradient_boosting' == model_option:
        model = GradientBoostingRegressor(**cur_model_params)
    elif 'xgb_boost' == model_option:
        model = XGBRegressor(**cur_model_params)
    elif 'light_gbm' == model_option:
        model = LGBMRegressor(**cur_model_params)
    elif 'decision_tree' == model_option:
        model = DecisionTreeRegressor(**cur_model_params)
    elif 'random_forest' == model_option:
        model = RandomForestRegressor(**cur_model_params)
    else:
        DEBUGGER.debug('Failure encountered: Unspecified or unsupported regressor')
        raise KeyError('Unspecified or unsupported regressor')
    return model

def get_hyperopt_regressor(model_option, **model_params):
    if 'lasso' == model_option:
        model = lasso(model_option, **model_params)
    elif 'linear' == model_option:
        model = linear_regression(model_option, **model_params)
    elif 'ridge' == model_option:
        model = ridge(model_option, **model_params)
    elif 'gradient_boosting' == model_option:
        model = gradient_boosting_regressor(model_option, **model_params)
    elif 'xgb_boost' == model_option:
        model = xgboost_regression(model_option, **model_params)
    elif 'light_gbm' == model_option:
        model = lightgbm_regression(model_option, **model_params)
    elif 'decision_tree' == model_option:
        model = decision_tree_regressor(model_option, **model_params)
    elif 'random_forest' == model_option:
        model = random_forest_regressor(model_option, **model_params)
    else:
        DEBUGGER.debug('Failure encountered: Unspecified or unsupported hyperopt wrapper')
        raise KeyError('Unspecified or unsupported hyperopt wrapper')
    return model

def get_params_grid(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    return __get_regressor_params(model_option, params_key_stem=params_key_stem, prefix=prefix)

def get_params_pounds(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    return APP_PROPS.get_ranges(prop_key=params_key_stem + model_option)

def __get_regressor_params(model_option, params_key_stem: str='model.param_grids.', prefix: str=None):
    params_grid = {}
    params_grid_dict = APP_PROPS.get_dict_properties(prop_key=params_key_stem + model_option)
    param_keys = params_grid_dict.keys()
    for param_key in param_keys:
        param = params_grid_dict.get(param_key)
        if param is not None:
            numsteps = int(param.get('num'))
            param_type = param.get('type')
            if param_type == int:
                start = int(param.get('start'))
                end = int(param.get('end'))
                if prefix is None:
                    params_grid[param_key] = np.linspace(start, end, numsteps).astype(int).tolist()
                else:
                    params_grid[prefix + '__' + param_key] = np.linspace(start, end, numsteps).astype(int).tolist()
            elif param_type == float:
                start = float(param.get('start'))
                end = float(param.get('end'))
                if prefix is None:
                    params_grid[param_key] = np.linspace(start, end, numsteps).tolist()
                else:
                    params_grid[prefix + '__' + param_key] = np.linspace(start, end, numsteps).tolist()
            elif param_type == bool:
                if prefix is None:
                    params_grid[param_key] = [bool(x) for x in param.get('values') if (param.get('values') is not None)]
                else:
                    params_grid[prefix + '__' + param_key] = [bool(x) for x in param.get('values') if (param.get('values') is not None)]
            else:
                if prefix is None:
                    params_grid[param_key] = [x for x in param.get('values') if (param.get('values') is not None)]
                else:
                    params_grid[prefix + '__' + param_key] = [x for x in param.get('values') if (param.get('values') is not None)]
    if params_grid is None:
        params_grid = {}
    #DEBUGGER.debug('Hyperparameter grid: {}'.format(params_grid))
    return params_grid

def get_default_grid(param_key: str, model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    param_grid = get_params_grid(model_option=model_option, params_key_stem=params_key_stem, prefix=prefix)
    param_keys = param_grid.keys()
    rel_param_grid = {}
    for key in param_keys:
        if param_key == key:
            rel_param_grid = {key: param_grid.get(key)}
    DEBUGGER.debug('Default hyperparameter grid params: {}'.format(rel_param_grid))
    return rel_param_grid

def get_params(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    param_grid = get_params_grid(model_option=model_option, params_key_stem=params_key_stem, prefix=prefix)
    return param_grid.keys()
