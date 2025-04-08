import numpy as np
import importlib
import re
from cheutils.properties_util import AppProperties
from cheutils.loggers import LoguruWrapper
from cheutils.ml.model_properties import ModelProperties
from typing import cast

LOGGER = LoguruWrapper().get_logger()

def get_estimator(**model_params):
    """
    Gets a specified estimator configured with key 'model_option'.
    :param model_params: model parameters
    :type model_params:
    :return: estimator instance
    :rtype:
    """
    cur_model_params = model_params.copy()
    model_option = None
    if 'model_option' in cur_model_params:
        model_option = cur_model_params.get('model_option')
        del cur_model_params['model_option']
    if 'params_grid_key' in cur_model_params:
        params_grid_key = cur_model_params.get('params_grid_key')
        del cur_model_params['params_grid_key']
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    model_info = __model_handler.get_models_supported().get(model_option)
    assert model_info is not None, 'Model info must be specified'
    model_class = getattr(importlib.import_module(model_info.get('package')), model_info.get('name'))
    try:
        # clean incoming grid of any prefixes
        clean_model_params = {}
        for param_key, param_val in cur_model_params.items():
            if '__' in param_key:
                conf_param_key = param_key.split('__')[1]
                clean_model_params[conf_param_key] = param_val
            else:
                clean_model_params[param_key] = param_val
        # default parameters are those that are not necessarily included in the configured list for optimization
        default_params = model_info.get('default_params')
        if default_params is not None:
            for key, value in default_params.items():
                if key not in clean_model_params:
                    clean_model_params[key] = value
        model = model_class(**clean_model_params)
        return model
    except TypeError as err:
        LOGGER.debug('Failure encountered: Unspecified or unsupported estimator')
        raise KeyError('Unspecified or unsupported estimator')

def get_hyperopt_estimator(model_option, **model_params):
    """
    Get a specified estimator configured with key 'model_option' - specifically relevant to hyperoptsklearn estimators.
    :param model_option: model option string
    :type model_option:
    :param model_params: model parameters
    :type model_params:
    :return: estimator instance
    :rtype:
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    model_info = __model_handler.get_models_supported().get(model_option)
    assert model_info is not None, 'Model info must be specified'
    model_class = getattr(importlib.import_module(model_info.get('package')), model_info.get('name'))
    try:
        model = model_class(model_option, **model_params)
    except TypeError as err:
        LOGGER.debug('Failure encountered: Unspecified or unsupported estimator')
        raise KeyError('Unspecified or unsupported estimator')
    return model

def get_params_grid(model_option: str, prefix: str=None):
    """
    Gets the hyperparameters grid as configured in the application properties file.
    :param model_option: model option, e.g. 'random_forest'
    :type model_option:
    :param prefix: model prefix, e.g. 'main_model'
    :type prefix:
    :return: dictionary of model hyperparameters keys and values
    :rtype: dict
    """
    return __get_estimator_params(model_option, prefix=prefix)

def get_params_pounds(model_option: str, prefix: str=None):
    """
    Gets the hyperparameters bounding values as configured in the application properties file.
    :param model_option: model option, e.g. 'random_forest'
    :type model_option:
    :param prefix: model prefix, e.g. 'main_model'
    :type prefix:
    :return: dictionary of hyperparameters bounding values
    :rtype: dict
    """
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    return __model_handler.get_params_grid(model_option=model_option, is_range=True)

def parse_grid_types(from_grid: dict, model_option: str=None, prefix: str=None):
    """
    Parses the specified hyperparameters grid and returns a copy with hyperparameters of the relevant or configured type.
    :param from_grid: source hyperparameters grid, which is possibly generated
    :type from_grid:
    :param model_option: the relevant model option, if any, e.g. 'random_forest'
    :type model_option:
    :param prefix: model prefix, if any, e.g. 'main_model'
    :type prefix:
    :return: dictionary of hyperparameters grid
    :rtype: dict
    """
    assert from_grid is not None, 'A valid parameter grid must be provided'
    params_grid = {}
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    params_grid_dict = __model_handler.get_params_grid(model_option=model_option)
    param_keys = from_grid.keys()
    for param_key in param_keys:
        conf_param_key = param_key.split('__')[1] if '__' in param_key else param_key
        param = params_grid_dict.get(conf_param_key)
        param_val = from_grid.get(param_key)
        param_val = eval(re.sub(r'\s+', ' ', param_val)) if isinstance(param_val, str) else param_val
        if param is not None:
            param_type = param.get('type')
            if param_type == int:
                if prefix is None:
                    params_grid[param_key] = param_val if isinstance(param_val, list) else int(param_val)
                else:
                    params_grid[prefix + '__' + conf_param_key] = param_val if isinstance(param_val, list) else int(param_val)
            elif param_type == float:
                if prefix is None:
                    params_grid[param_key] = param_val if isinstance(param_val, list) else float(param_val)
                else:
                    params_grid[prefix + '__' + conf_param_key] = param_val if isinstance(param_val, list) else float(param_val)
            elif param_type == bool:
                if prefix is None:
                    params_grid[param_key] = param_val if isinstance(param_val, list) else bool(param_val)
                else:
                    params_grid[prefix + '__' + conf_param_key] = param_val if isinstance(param_val, list) else bool(param_val)
            else:
                if prefix is None:
                    params_grid[param_key] = param_val
                else:
                    params_grid[prefix + '__' + conf_param_key] = param_val
    if params_grid is None:
        params_grid = {}
    return params_grid

def get_param_defaults(param_key: str, model_option: str, prefix: str=None):
    """
    Get the configured default values for the specified parameter key.
    :param param_key: specified key
    :type param_key:
    :param model_option: any specified supported model option
    :type model_option:
    :param prefix: elevant model prefix, if any, e.g. 'model_model'
    :type prefix:
    :return: dictionary item for the specified key
    :rtype: dict
    """
    param_grid = get_params_grid(model_option=model_option, prefix=prefix)
    param_keys = param_grid.keys()
    rel_param_grid = {}
    for key in param_keys:
        if param_key == key:
            rel_param_grid = {key: param_grid.get(key)}
    LOGGER.debug('Default hyperparameter value: {}'.format(rel_param_grid))
    return rel_param_grid

def __get_estimator_params(model_option, prefix: str=None):
    params_grid = {}
    __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
    params_grid_dict = __model_handler.get_params_grid(model_option=model_option)
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
