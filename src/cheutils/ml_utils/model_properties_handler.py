from cheutils.properties_util import AppProperties, AppPropertiesHandler, PropertiesException
from cheutils.decorator_singleton import singleton
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

@singleton
class ModelProperties(AppPropertiesHandler):
    __app_props: AppProperties

    def __init__(self):
        super().__init__()
        self.__model_properties = {}
        self.__app_props = AppProperties()
        self.__app_props.subscribe(self)
        self._load(prop_key='models_supported')

    # overriding abstract method
    def reload(self):
        for key in self.__model_properties.keys():
            self.__model_properties[key] = self._load(prop_key=key)

    def _load(self, prop_key: str=None):
        LOGGER.debug('Attempting to load model property: {}', prop_key)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')()

    def __getattr__(self, item):
        msg = f'Attempting to load unspecified model property: {item}'
        LOGGER.error(msg)
        raise PropertiesException(msg)

    def _load_params(self, prop_key: str=None, params: dict=None):
        LOGGER.debug('Attempting to load model property: {}, {}', prop_key, params)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')(**params)

    def _load_range(self, prop_key: str=None, params: dict=None):
        LOGGER.debug('Attempting to load model property: {}, {}', prop_key, params)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')(**params)

    def _load_unspecified(self):
        raise PropertiesException('Attempting to load unspecified model property')

    def _load_models_supported(self):
        key = 'project.models.supported'
        self.__model_properties['models_supported'] = self.__app_props.get_dict_properties(key)

    def _load_params_grid(self, params: dict=None):
        key = 'model.params_grid.' + str(params.get('model_option'))
        self.__model_properties['model_params_grid_' + str(params.get('model_option'))] = self.__app_props.get_dict_properties(key)

    def _load_params_range(self, params: dict=None):
        key = 'model.params_grid.' + str(params.get('model_option'))
        self.__model_properties['model_params_grid_' + str(params.get('model_option'))] = self.__app_props.get_ranges(key)

    def get_models_supported(self):
        return self.__model_properties.get('models_supported')

    def get_params_grid(self, model_option: str=None, is_range: bool=False):
        key = 'model_params_grid_' + model_option
        value = self.__model_properties.get(key)
        if value is None:
            if is_range:
                self._load_params(prop_key='params_grid', params=model_option)
            else:
                self._load_range(prop_key='params_range', params=model_option)
            return self.__model_properties.get(key)
        return value

