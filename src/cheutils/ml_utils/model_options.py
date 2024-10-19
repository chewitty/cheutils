import numpy as np
import importlib
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties

LOGGER = LoguruWrapper().get_logger()
APP_PROPS = AppProperties()
prop_key = 'project.models.supported'
MODELS_SUPPORTED = APP_PROPS.get_dict_properties(prop_key)
assert MODELS_SUPPORTED is not None, 'Models supported must be specified'

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
    model_info = MODELS_SUPPORTED.get(model_option)
    assert model_info is not None, 'Model info must be specified'
    model_class = getattr(importlib.import_module(model_info.get('module_package')), model_info.get('module_name'))
    try:
        model = model_class(**cur_model_params)
    except TypeError as err:
        LOGGER.debug('Failure encountered: Unspecified or unsupported regressor')
        raise KeyError('Unspecified or unsupported regressor')
    return model

def get_hyperopt_regressor(model_option, **model_params):
    model_info = MODELS_SUPPORTED.get(model_option)
    assert model_info is not None, 'Model info must be specified'
    model_class = getattr(importlib.import_module(model_info.get('module_package')), model_info.get('module_name'))
    try:
        model = model_class(model_option, **model_params)
    except TypeError as err:
        LOGGER.debug('Failure encountered: Unspecified or unsupported regressor')
        raise KeyError('Unspecified or unsupported regressor')
    return model

def get_params_grid(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    return __get_regressor_params(model_option, params_key_stem=params_key_stem, prefix=prefix)

def get_params_pounds(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    return APP_PROPS.get_ranges(prop_key=params_key_stem + model_option)

def parse_grid_types(from_grid: dict, params_key_stem: str='model.param_grids.', model_option: str=None, prefix: str=None):
    assert from_grid is not None, 'A valid parameter grid must be provided'
    params_grid = {}
    params_grid_dict = APP_PROPS.get_dict_properties(prop_key=params_key_stem + model_option)
    param_keys = from_grid.keys()
    for param_key in param_keys:
        param = params_grid_dict.get(param_key)
        if param is not None:
            param_type = param.get('type')
            if param_type == int:
                if prefix is None:
                    params_grid[param_key] = int(from_grid.get(param_key))
                else:
                    params_grid[prefix + '__' + param_key] = int(from_grid.get(param_key))
            elif param_type == float:
                if prefix is None:
                    params_grid[param_key] = float(from_grid.get(param_key))
                else:
                    params_grid[prefix + '__' + param_key] = float(from_grid.get(param_key))
            elif param_type == bool:
                if prefix is None:
                    params_grid[param_key] = bool(from_grid.get(param_key))
                else:
                    params_grid[prefix + '__' + param_key] = bool(from_grid.get(param_key))
            else:
                if prefix is None:
                    params_grid[param_key] = from_grid.get(param_key)
                else:
                    params_grid[prefix + '__' + param_key] = from_grid.get(param_key)
    if params_grid is None:
        params_grid = {}
    return params_grid

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
                    params_grid[param_key] = np.linspace(start, end, numsteps, dtype=int).tolist()
                else:
                    params_grid[prefix + '__' + param_key] = np.linspace(start, end, numsteps, dtype=int).tolist()
            elif param_type == float:
                start = float(param.get('start'))
                end = float(param.get('end'))
                if prefix is None:
                    params_grid[param_key] = np.round(np.linspace(start, end, numsteps), 4).tolist()
                else:
                    params_grid[prefix + '__' + param_key] = np.round(np.linspace(start, end, numsteps), 4).tolist()
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
    #LOGGER.debug('Hyperparameter grid: {}'.format(params_grid))
    return params_grid

def get_default_grid(param_key: str, model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    param_grid = get_params_grid(model_option=model_option, params_key_stem=params_key_stem, prefix=prefix)
    param_keys = param_grid.keys()
    rel_param_grid = {}
    for key in param_keys:
        if param_key == key:
            rel_param_grid = {key: param_grid.get(key)}
    LOGGER.debug('Default hyperparameter grid params: {}'.format(rel_param_grid))
    return rel_param_grid

def get_params(model_option: str, params_key_stem: str='model.param_grids.', prefix: str=None):
    param_grid = get_params_grid(model_option=model_option, params_key_stem=params_key_stem, prefix=prefix)
    return param_grid.keys()
