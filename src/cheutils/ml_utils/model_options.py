import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def get_regressor(**model_params):
    """
    Gets a specified regressor configured with key 'model_option'.
    """
    cur_model_params = model_params.copy()
    model_option = None
    if 'model_option' in cur_model_params:
        model_option = cur_model_params.get('model_option')
        del cur_model_params['model_option']
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
        raise KeyError('Unspecified regressor')
    return model

def get_params_grid(model_option: str, prefix: str):
    return __get_regressor_params(model_option, prefix)

def __get_regressor_params(model_option, prefix: str):
    params_grid = {}
    if prefix is None:
        if 'xgb_boost' == model_option:
            params_grid = {'n_estimators':     np.linspace(5, 201, 10).astype(int).tolist(),
                           'max_depth':        np.linspace(5, 11, 6).astype(int).tolist(),
                           'max_leaves':       [73],
                           'learning_rate':    np.linspace(0.30, 0.51, 10).tolist(),
                           'colsample_bytree': [0.83],
                           'reg_alpha':        [0.175],
                           }
            """params_grid = {'n_estimators'    : np.linspace(5, 401, 5).astype(int).tolist(),
                           'max_depth'       : np.linspace(1, 31, 5).astype(int).tolist(),
                           'max_leaves'      : np.linspace(2, 125, 5).astype(int).tolist(),
                           'learning_rate'   : np.linspace(0.0, 1.0, 5).tolist(),
                           'colsample_bytree': np.linspace(0.0, 1.0, 5).tolist(),
                           'subsample'       : np.linspace(0.0, 1.0, 5).tolist(),
                           'gamma'           : np.linspace(0.0, 1.0, 5).tolist(),
                           'reg_alpha'       : np.linspace(0.0, 0.51, 5).tolist(),
                           }"""
        elif 'random_forest' == model_option:
            params_grid = {'n_estimators':     np.linspace(370, 481, 20).astype(int).tolist()[0::2],
                           'max_depth':        np.linspace(11, 15, 1).astype(int).tolist()[0::2],
                           'min_samples_split': np.linspace(7, 21, 3).astype(int).tolist()[0::2],
                           'min_samples_leaf': np.linspace(2, 15, 3).astype(int).tolist()[0::2],
                           'max_leaf_nodes':   np.linspace(300, 401, 10).astype(int).tolist()[0::2],
                           }
        else:
            params_grid = {}
    else:
        if 'lasso' == model_option:
            params_grid = {prefix + '_alpha': np.arange(0.0001, 5.2, 0.2).tolist()[0::2], }
        elif 'ridge' == model_option:
            params_grid = {prefix + '_alpha': np.arange(0.0001, 5.2, 0.2).tolist()[0::2], }
        elif 'gradient_boosting' == model_option:
            params_grid = {prefix + '_max_depth'    : np.arange(3, 17).tolist()[0::2],
                           prefix + '_learning_rate': np.arange(0.05, 2.1, 0.2).tolist()[0::2],
                           prefix + '_subsample'    : np.arange(0.1, 1.1, 0.2).tolist()[0::2],
                           }
        elif 'xgb_boost' == model_option:
            params_grid = {prefix + '_n_estimators'    : np.arange(5, 15, 2).tolist()[0::2],
                           prefix + '_max_depth'       : np.arange(2, 5).tolist()[0::2],
                           prefix + '_max_leaves'      : np.arange(2, 7).tolist()[0::2],
                           prefix + '_learning_rate'   : np.arange(0.01, 0.1, 0.02).tolist()[0::2],
                           prefix + '_subsample'       : np.arange(0.8, 1.1, 0.1).tolist()[0::2],
                           prefix + '_scale_pos_weight': np.arange(0.8, 1.1, 0.1).tolist()[0::2],
                           prefix + '_gamma'           : np.arange(0.4, 1.1, 0.1).tolist()[0::2],
                           prefix + '_reg_alpha'       : np.arange(0.025, 0.071, 0.01).tolist()[0::2],
                           }
        elif 'light_gbm' == model_option:
            params_grid = {prefix + '_max_depth'    : np.arange(3, 17).tolist()[0::2],
                           prefix + '_learning_rate': np.arange(0.0005, 1.1, 0.2).tolist()[0::2],
                           prefix + '_reg_alpha'    : np.arange(0.1, 5.1, 0.2).tolist()[0::2],
                           }
        elif 'decision_tree' == model_option:
            params_grid = {prefix + '_max_depth'       : np.arange(3, 17).tolist()[0::2],
                           prefix + '_min_samples_leaf': np.arange(3, 13, 2).tolist()[0::2],
                           prefix + '_max_leaf_nodes'  : np.arange(2, 100, 5).tolist()[0::2],
                           }
        elif 'random_forest' == model_option:
            params_grid = {prefix + '_n_estimators'     : np.arange(370, 481, 20).tolist()[0::2],
                           prefix + '_max_depth'        : np.arange(11, 15, 1).tolist()[0::2],
                           prefix + '_min_samples_split': np.arange(7, 21, 3).tolist()[0::2],
                           prefix + '_min_samples_leaf' : np.arange(2, 15, 3).tolist()[0::2],
                           prefix + '_max_leaf_nodes'   : np.arange(300, 401, 10).tolist()[0::2],
                           }
        else:
            params_grid = {}
    return params_grid

def get_default_grid(param_key: str, model_option: str, prefix: str):
    param_grid = get_params_grid(model_option=model_option, prefix=prefix)
    param_keys = param_grid.keys()
    rel_param_grid = {}
    for key in param_keys:
        if param_key == key:
            rel_param_grid = {key: param_grid.get(key)}
    return rel_param_grid

def get_params(model_option: str, prefix: str):
    param_grid = get_params_grid(model_option=model_option, prefix=prefix)
    return param_grid.keys()
