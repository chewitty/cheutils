from cheutils.properties_util import AppProperties, AppPropertiesHandler
from cheutils.exceptions import PropertiesException
from cheutils.loggers import LoguruWrapper
from cheutils.decorator_singleton import singleton

LOGGER = LoguruWrapper().get_logger()

@singleton
class ProjectTreeProperties(AppPropertiesHandler):
    __app_props: AppProperties

    def __init__(self, name: str=None):
        super().__init__(name=name)
        self.__project__tree_properties = {}
        self.__app_props = AppProperties()

    # overriding abstract method
    def reload(self):
        self.__project__tree_properties = {}
        """for key in self.__project__tree_properties.keys():
            try:
                self.__project__tree_properties[key] = self._load(prop_key=key)
            except Exception as err:
                LOGGER.warning('Problem reloading property: {}, {}', key, err)
                pass"""

    def _load(self, prop_key: str=None):
        LOGGER.debug('Attempting to load project tree property: {}', prop_key)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')()

    def __getattr__(self, item):
        msg = f'Attempting to load unspecified project tree property: {item}'
        LOGGER.warning(msg)

    def _load_unspecified(self):
        raise PropertiesException('Attempting to load unspecified project tree property')

    def _load_project_root(self):
        key = 'project.root.dir'
        self.__project__tree_properties['project_root'] = self.__app_props.get(key) if self.__app_props.get(key) is not None else './'

    def _load_project_namespace(self):
        key = 'project.namespace'
        self.__project__tree_properties['project_namespace'] = self.__app_props.get(key) if self.__app_props.get(key) is not None else 'my_proj'

    def _load_project_data(self):
        key = 'project.data.dir'
        self.__project__tree_properties['project_data'] = self.__app_props.get(key) if self.__app_props.get(key) is not None else './data/'

    def _load_project_output(self):
        key = 'project.output.dir'
        self.__project__tree_properties['project_output'] = self.__app_props.get(key) if self.__app_props.get(key) is not None else './output/'

    def get_proj_namespace(self):
        value = self.__project__tree_properties.get('project_namespace')
        if value is None:
            self._load_project_namespace()
        return self.__project__tree_properties.get('project_namespace')

    def get_proj_root(self):
        value = self.__project__tree_properties.get('project_root')
        if value is None:
            self._load_project_root()
        return self.__project__tree_properties.get('project_root')

    def get_proj_data(self):
        value = self.__project__tree_properties.get('project_data')
        if value is None:
            self._load_project_data()
        return self.__project__tree_properties.get('project_data')

    def get_proj_output(self):
        value = self.__project__tree_properties.get('project_output')
        if value is None:
            self._load_project_output()
        return self.__project__tree_properties.get('project_output')