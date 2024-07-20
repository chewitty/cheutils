import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_regressor(**model_params):
    """
    Gets a specified regressor configured with key 'model_option'.
    """
    cur_model_params = model_params.copy()
    model_option = cur_model_params.get('model_option')
    if 'model_option' in cur_model_params:
        del cur_model_params['model_option']
    if 'xgb_boost' == model_option:
        model = XGBRegressor(**cur_model_params)
    elif 'random_forest' == model_option:
        model = RandomForestRegressor(**cur_model_params)
    else:
        raise KeyError('Unspecified regressor')
    return model

def get_params_grid(model_option: str, prefix: str):
    return __get_regressor_params(model_option, prefix)

def __get_regressor_params(model_option, prefix: str):
    params_grid = {}
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