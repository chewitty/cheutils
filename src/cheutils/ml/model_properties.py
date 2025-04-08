import importlib
from sklearn.metrics import make_scorer
from cheutils.properties_util import AppProperties, AppPropertiesHandler
from cheutils.decorator_singleton import singleton
from cheutils.loggers import LoguruWrapper
from cheutils.exceptions import PropertiesException

LOGGER = LoguruWrapper().get_logger()

@singleton
class ModelProperties(AppPropertiesHandler):
    __app_props: AppProperties

    def __init__(self, name: str=None,):
        super().__init__(name=name)
        self.__model_properties = {}
        self.__app_props = AppProperties()

    # overriding abstract method
    def reload(self):
        """
        Reloads the underlying application properties.
        :return:
        :rtype:
        """
        self.__model_properties = {}
        """for key in self.__model_properties.keys():
            try:
                self.__model_properties[key] = self._load(prop_key=key)
            except Exception as err:
                LOGGER.warning('Problem reloading property: {}, {}', key, err)
                pass"""

    def _load(self, prop_key: str=None):
        LOGGER.debug('Attempting to load model property: {}', prop_key)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')()

    def __getattr__(self, item):
        msg = f'Attempting to load unspecified model property: {item}'
        LOGGER.warning(msg)

    def _load_params(self, prop_key: str=None, params: dict=None):
        LOGGER.debug('Attempting to load model property: {}, {}', prop_key, params)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')(params)

    def _load_range(self, prop_key: str=None, params: dict=None):
        LOGGER.debug('Attempting to load model property: {}, {}', prop_key, params)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')(params)

    def _load_unspecified(self):
        raise PropertiesException('Attempting to load unspecified model property')

    def _load_models_supported(self):
        key = 'project.models.supported'
        self.__model_properties['models_supported'] = self.__app_props.get_dict_properties(key)

    def _load_n_jobs(self):
        key = 'model.active.n_jobs'
        self.__model_properties['n_jobs'] = int(self.__app_props.get(key))

    def _load_model_option(self):
        key = 'model.active.model_option'
        self.__model_properties['model_option'] = self.__app_props.get(key)

    def _load_n_iters(self):
        key = 'model.active.n_iters'
        self.__model_properties['n_iters'] = int(self.__app_props.get(key))

    def _load_n_trials(self):
        key = 'model.active.n_trials'
        self.__model_properties['n_trials'] = int(self.__app_props.get(key))

    def _load_grid_resolution(self):
        key = 'model.active.grid_resolution'
        self.__model_properties['grid_resolution'] = int(self.__app_props.get(key))

    def _load_find_grid_resolution(self):
        key = 'model.find_optimal.grid_resolution'
        self.__model_properties['find_grid_resolution'] = self.__app_props.get_bol(key)

    def _load_params_grid(self, params: dict=None):
        key = 'model.params_grid.' + str(params.get('model_option'))
        self.__model_properties['model_params_grid_' + str(params.get('model_option'))] = self.__app_props.get_dict_properties(key)

    def _load_params_range(self, params: dict=None):
        key = 'model.params_grid.' + str(params.get('model_option'))
        self.__model_properties['model_params_grid_' + str(params.get('model_option'))] = self.__app_props.get_ranges(key)

    def _load_grid_resolutions_sample(self):
        key = 'model.grid_resolutions.sample'
        self.__model_properties['grid_resolutions_sample'] = self.__app_props.get_dict_properties(key)

    def _load_grid_resolution_with_cv(self):
        key = 'model.find_optimal.grid_resolution.with_cv'
        self.__model_properties['grid_resolution_with_cv'] = self.__app_props.get_bol(key)

    def _load_hyperopt_algos(self):
        key = 'model.hyperopt.algos'
        self.__model_properties['hyperopt_algos'] = self.__app_props.get_dict_properties(key)

    def _load_cross_val_scoring(self):
        key = 'model.cross_val.scoring_fn'
        scorer_dict = self.__app_props.get_dict_properties(key)
        if scorer_dict is not None:
            scorer_package = scorer_dict.get('package')
            scorer_fn = scorer_dict.get('func')
            scorer_params = scorer_dict.get('params')
            self.__model_properties['cross_val_scoring'] = make_scorer(getattr(importlib.import_module(scorer_package), scorer_fn), **scorer_params)
        else:
            key = 'model.cross_val.scoring'
            self.__model_properties['cross_val_scoring'] = self.__app_props.get(key)

    def _load_cross_val_num_folds(self):
        key = 'model.cross_val.num_folds'
        key_st = 'model.cross_val.strategy'
        cv_strategy = self.__app_props.get_dict_properties(key_st)
        if cv_strategy is not None:
            cv_package = cv_strategy.get('package')
            cv_fn = cv_strategy.get('func')
            cv_params = cv_strategy.get('params')
            splitter_params = {} if cv_params is None or (not cv_params) else cv_params
            splitter_class = getattr(importlib.import_module(cv_package), cv_fn)
            splitter_obj = None
            try:
                splitting_obj = splitter_class(**splitter_params)
                self.__model_properties['cross_val_num_folds'] = splitting_obj
            except TypeError as err:
                LOGGER.error('Problem encountered instantiating cv strategy: {}, {}', cv_fn, err)
                # default to the usual
                self.__model_properties['cross_val_num_folds'] = int(self.__app_props.get(key))
        else:
            self.__model_properties['cross_val_num_folds'] = int(self.__app_props.get(key))

    def _load_random_seed(self):
        key = 'model.active.random_seed'
        self.__model_properties['random_seed'] = int(self.__app_props.get(key))

    def _load_trial_timeout(self):
        key = 'model.active.trial_timeout'
        self.__model_properties['trial_timeout'] = int(self.__app_props.get(key))

    def get_models_supported(self):
        value = self.__model_properties.get('models_supported')
        if value is None:
            self._load_models_supported()
        return self.__model_properties.get('models_supported')

    def get_params_grid(self, model_option: str=None, is_range: bool=False):
        key = 'model_params_grid_' + model_option
        value = self.__model_properties.get(key)
        if value is None:
            if is_range:
                self._load_range(prop_key='params_range', params={'model_option': model_option})
            else:
                self._load_params(prop_key='params_grid', params={'model_option': model_option})
            return self.__model_properties.get(key)
        return value

    def get_n_jobs(self):
        value = self.__model_properties.get('n_jobs')
        if value is None:
            self._load_n_jobs()
        return self.__model_properties.get('n_jobs')

    def get_n_iters(self):
        value = self.__model_properties.get('n_iters')
        if value is None:
            self._load_n_iters()
        return self.__model_properties.get('n_iters')

    def get_n_trials(self):
        value = self.__model_properties.get('n_trials')
        if value is None:
            self._load_n_trials()
        return self.__model_properties.get('n_trials')

    def get_grid_resolution(self):
        value = self.__model_properties.get('grid_resolution')
        if value is None:
            self._load_grid_resolution()
        return self.__model_properties.get('grid_resolution')

    def get_find_grid_resolution(self):
        value = self.__model_properties.get('find_grid_resolution')
        if value is None:
            self._load_find_grid_resolution()
        return self.__model_properties.get('find_grid_resolution')

    def get_grid_resolutions_sample(self):
        value = self.__model_properties.get('grid_resolutions_sample')
        if value is None:
            self._load_grid_resolutions_sample()
        return self.__model_properties.get('grid_resolutions_sample')

    def get_model_option(self):
        value = self.__model_properties.get('model_option')
        if value is None:
            self._load_model_option()
        return self.__model_properties.get('model_option')

    def get_grid_resolution_with_cv(self):
        value = self.__model_properties.get('grid_resolution_with_cv')
        if value is None:
            self._load_grid_resolution_with_cv()
        return self.__model_properties.get('grid_resolution_with_cv')

    def get_hyperopt_algos(self):
        value = self.__model_properties.get('hyperopt_algos')
        if value is None:
            self._load_hyperopt_algos()
        return self.__model_properties.get('hyperopt_algos')

    def get_cross_val_scoring(self):
        value = self.__model_properties.get('cross_val_scoring')
        if value is None:
            self._load_cross_val_scoring()
        return self.__model_properties.get('cross_val_scoring')

    def get_cross_val_num_folds(self):
        value = self.__model_properties.get('cross_val_num_folds')
        if value is None:
            self._load_cross_val_num_folds()
        return self.__model_properties.get('cross_val_num_folds')

    def get_random_seed(self):
        value = self.__model_properties.get('random_seed')
        if value is None:
            self._load_random_seed()
        return self.__model_properties.get('random_seed')

    def get_trial_timeout(self):
        value = self.__model_properties.get('trial_timeout')
        if value is None:
            self._load_trial_timeout()
        return self.__model_properties.get('trial_timeout')



