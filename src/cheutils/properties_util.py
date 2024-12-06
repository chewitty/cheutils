import datetime
import os
import importlib
from jproperties import Properties
from ast import literal_eval
from abc import ABC, abstractmethod
from cheutils.decorator_debug import debug_func
from cheutils.decorator_singleton import singleton
from cheutils.exceptions import PropertiesException
from cheutils.loggers import LoguruWrapper
from cheutils.common_utils import properties_to_frame

# Define project constants.
APP_CONFIG = 'app-config.properties'
LOGGER = LoguruWrapper().get_logger()

"""
Abstract class for properties handlers - i.e., implementations can subscribe or unsubscribe to the prevailing singleton instance of `AppProperties` 
to be notified of changes in the properties file that trigger a reloading of the properties file, if prompted via `AppProperties` reload() method.
"""
class AppPropertiesHandler(ABC):
    def __init__(self, name: str=None):
        assert name is not None and not (not name), 'A valid handler name is required'
        super().__init__()
        self.__name = name

    def get_name(self):
        return self.__name

    @abstractmethod
    def reload(self):
        """
        Reloads the underlying appliation properties file (``app-config.properties``)
        :return: None
        :rtype:
        """
        pass

"""
Utilities for reading project properties or configuration files. When instantiated, it loads the first 
app-config.properties found anywhere in the project root folder or subfolders. Usually, it is 
recommended that the app-config.properties is stored in the data subfolder of the project root. It also supports
a reload method, which allows a reload of the properties file anytime subsequently as desired.
"""
@singleton
class AppProperties(object):
    instance__ = None
    app_props__ = None
    handlers__ = {}
    """
    A static method responsible for creating and returning a new instance (called before __init__)
    """
    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created, 
        or else returns the previous singleton object
        """
        if AppProperties.instance__ is None:
            AppProperties.instance__ = super().__new__(cls)
        return AppProperties.instance__

    """
    An instance method, the class constructor, responsible for initializing the attributes of the newly created
    """
    @debug_func(enable_debug=True, prefix='app_config')
    def __init__(self, *args, **kwargs):
        """
        Initializes the properties utility and loads the first `app-config.properties` found anywhere in
        the project root folder or subfolders. Usually, it is recommended that the app-config.properties is stored
        in the data subfolder of the project root.
        """
        # Load the properties file
        self.__load()

    def reload(self) -> None:
        """
        Reload the properties configuration file. All subscribed handlers' reload() method is also called, triggering a refresh of any cached properties.
        :return: None
        :rtype:
        """
        cur_props = self.app_props__
        try:
            self.__load()
            self.__notify_handlers()
            LOGGER.success('Successfully reloaded = {}', APP_CONFIG)
        except Exception as ex:
            # revert to previous version
            self.app_props__ = cur_props
            LOGGER.warning('Could not reload = {}', APP_CONFIG)
            raise ex

    def __str__(self):

        path_to_app_config = os.path.join(self.get_subscriber('proj_handler').get_proj_data(), APP_CONFIG)
        info = 'AppProperties based on properties file = ' + path_to_app_config
        LOGGER.info(info)
        return info

    def get(self, prop_key=None):
        """
        Get the value associated with the specified key.
        Parameters:
            prop_key(str): the property name for which a value is required.
        Returns:
            (str): the value associated with the specified key or None if there is no value; None if the key specified is None.
        """
        if prop_key is None:
            return None
        avail_prop = self.app_props__.get(prop_key)
        prop_value = None if (avail_prop is None) else avail_prop.data
        if prop_value is None:
            return None
        return prop_value.strip()

    def get_bol(self, prop_key=None):
        """
        Get the value associated with the specified key.
        Parameters:
            prop_key(str): the property name for which a value is required.
        Returns:
            (bool): the value associated with the specified key as bool or None if there is no value; None if the key specified is None.
        """
        if prop_key is None:
            return None
        prop_item = self.app_props__.get(prop_key)
        if prop_item is None:
            return None
        prop_value = prop_item.data
        if prop_value is None:
            return None
        return bool(eval(prop_value.strip()))

    def get_keys(self):
        """
        Get the list of keys held in the properties file.
        :return:
            list(str): a list of keys
        """
        items_view = self.app_props__.items()
        list_keys = []
        for item in items_view:
            list_keys.append(item[0])
        return list_keys

    def get_list(self, prop_key=None):
        """
        Get the value associated with the specified key as a list of strings.
        Parameters:
            prop_key(str): the property name for which a value is required.
        Returns:
            list(str): the value associated with the specified key as a list of strings or None if there is no value; None if the key specified is None.
        """
        if prop_key is None:
            return None
        prop_item = self.app_props__.get(prop_key)
        if prop_item is None:
            return None
        prop_value = prop_item.data
        if prop_value is None:
            return None
        tmp_list = prop_value.replace('\'', '').replace('\"', '').strip('][').split(',')
        result_list = list([x.strip() for x in tmp_list])
        return result_list

    def get_list_properties(self, prop_key=None):
        """
        Get the value associated with the specified key as a list.
        Parameters:
            prop_key(str): the property name for which a value is required.
        Returns:
            list(Union[list, dict]): the value associated with the specified key as a list of either Union[str, dict] or None if there is no value; None if the key specified is None.
        """
        if prop_key is None:
            return None
        prop_item = self.app_props__.get(prop_key)
        if prop_item is None:
            return None
        prop_value = prop_item.data
        if prop_value is None:
            return None
        internal_item = prop_value[1:-1].rstrip('\,')
        internal_item = eval(internal_item)
        result_list = list(internal_item)
        return result_list

    def is_set(self, prop_key=None):
        """
        Get the value associated with the specified key as a bool, based on flags set on/off in the properties file.
        Parameters:
            prop_key(str): the full property name - property name in the properties file as stem plus the key-part of a
            key/value pair specified as part of a list of (key=value) pairs in the properties file, appended to the
            property name separated with a dot(.).
        Returns:
            (bool): the on (1) or off (0) flag based on the specified key-part; the default is off (0), if no matching
            entry if found in the property value matching the prop_key stem in the properties file
        """
        if prop_key is None:
            return False
        # extract the appropriate property name for look up
        prop_vals = prop_key.split('.')
        key_part = prop_vals[len(prop_vals)-1]
        prop_stem = prop_key[0:len(prop_key) - len(key_part)-1]
        prop_item = self.app_props__.get(prop_stem)
        if prop_item is None:
            return False
        prop_value = prop_item.data
        if prop_value is None:
            return False
        # otherwise, identify the specific key-value pair
        key_val_pairs = self.get_flags(prop_stem)
        LOGGER.info('Flags = {}', key_val_pairs)
        key_set = key_val_pairs.get(key_part)
        if key_set is None:
            return False
        else:
            return key_set

    def get_flags(self, prop_key=None):
        """
        Get the flag values associated with the specified key as a dict of bools, based on flags set on/off in the properties file.
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(bool): a dict of on (1) or off (0) flags based on the specified key; the default is off (0).
        """
        if prop_key is None:
            return None
        prop_value = self.app_props__.get(prop_key).data
        LOGGER.info('Key-value property stem = {}', prop_key)
        if prop_value is None:
            return None
        tmp_list = prop_value.replace('\'', '').replace('\"', '').strip('][').split(',')
        flags = {}
        for item in tmp_list:
            val_pair = item.split('=')
            val_pair = [x.replace('\'', '').replace('\"', '').strip('') for x in val_pair]
            flags[val_pair[0].strip()] = bool(eval(val_pair[1].strip()))
        return flags

    def get_properties(self, prop_key=None):
        """
        Get the key-value pairs as value associated with the specified key as a dict set as key=value in the properties file.
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(str): a dict of string key-value pairs based on the specified key; the default is all
            configured properties as a dataframe.
        """
        if prop_key is None:
            return properties_to_frame(self.app_props__.properties)
        prop_value = self.app_props__.get(prop_key)
        if prop_value is None:
            return None
        prop_value = prop_value.data
        tmp_list = prop_value.replace('\'', '').replace('\"', '').strip('][').split(',')
        properties = {}
        for item in tmp_list:
            val_pair = item.split('=')
            val_pair = [x.replace('\'', '').replace('\"', '').strip('') for x in val_pair]
            properties[val_pair[0].strip()] = val_pair[1].strip()
        return properties

    def get_dict_properties(self, prop_key=None):
        """
        Gets any properties configured as a dictionary in the properties file
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(str): a dict of key-value pairs based on the specified key; the default is None.
        """
        if prop_key is None:
            return None
        prop_value = self.app_props__.get(prop_key)
        if prop_value is None:
            return None
        prop_value = prop_value.data
        properties = eval(prop_value)
        return properties

    def get_ranges(self, prop_key=None):
        """
        Gets the ranges associated with the specified key as a dict of tuples, based on ranges set in the properties file.
        :param prop_key:
        :type prop_key:
        :return: tuple of min and max values for the specified key; the default is None.
        :rtype:
        """
        prop_dict = self.get_dict_properties(prop_key)
        prop_ranges = {}
        for param, value in prop_dict.items():
            min_val = int(value.get('start')) if value is not None else value
            max_val = int(value.get('end')) if value is not None else value
            prop_ranges[param] = (min_val, max_val)
        return prop_ranges

    def get_property(self, prop_key=None):
        """
        Get the property value associated with the specified key, in a list of key-value pairs in the properties file.
        Parameters:
            prop_key(str): the full property name - property name in the properties file as stem plus the key-part of a
            key/value pair specified as part of a list of (key=value) pairs in the properties file, appended to the
            property name separated with a dot(.).
        Returns:
            (str): the value based on the specified key-part; the default is None, if no matching
            entry if found in the property value matching the prop_key stem in the properties file
        """
        if prop_key is None:
            return None
        # extract the appropriate property name for look up
        prop_vals = prop_key.split('.')
        key_part = prop_vals[len(prop_vals) - 1]
        prop_stem = prop_key[0:len(prop_key) - len(key_part) - 1]
        prop_item = self.app_props__.get(prop_stem)
        if prop_item is None:
            return None
        prop_value = prop_item.data
        if prop_value is None:
            return None
        # otherwise, identify the specific key-value pair
        key_val_pairs = self.get_properties(prop_stem)
        val_part = key_val_pairs.get(key_part)
        if val_part is None:
            return None
        else:
            return val_part

    def get_types(self, prop_key=None):
        """
        Get the key-value pairs as value associated with the specified key as a dict of types set as key=value in the properties file.
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(str): a dict of key-value pairs based on the specified key, with the
            value being the object type associated with the key; the default is None.
        """
        if prop_key is None:
            return None
        prop_value = self.app_props__.get(prop_key)
        if prop_value is None:
            return None
        prop_value = prop_value.data
        tmp_list = prop_value.replace('\'', '').replace('\"', '').strip('][').split(',')
        properties = {}
        for item in tmp_list:
            val_pair = item.split('=')
            val_pair = [x.replace('\'', '').replace('\"', '').strip('') for x in val_pair]
            type_val = val_pair[1].strip()
            if 'str'.casefold() == type_val.casefold():
                properties[val_pair[0].strip()] = str
            elif 'float'.casefold() == type_val.casefold():
                properties[val_pair[0].strip()] = float
            elif 'int'.casefold() == type_val.casefold():
                properties[val_pair[0].strip()] = int
            elif 'date'.casefold() == type_val.casefold():
                properties[val_pair[0].strip()] = datetime.date
            elif 'datetime'.casefold() == type_val.casefold():
                properties[val_pair[0].strip()] = datetime
            else:
                properties[val_pair[0].strip()] = str
        return properties

    def dump(self, props: dict):
        """
        Dump the properties in the specified dict as a dataframe of key, value columns
        :param props:
        :type props:
        :return:
        :rtype:
        """
        assert props is not None, 'A valid properties dictionary is required'
        return properties_to_frame(props)

    def load_custom_properties(self, prop_file_name: str)->Properties:
        """
        Loads any custom properties file, other than the default app-config.properties
        that is loaded automatically, from the project folder or any project subfolder.
        :param prop_file_name: the custom properties file name, not including any folder paths (e.g., ds-config.properties)
        :type prop_file_name: str
        :return: properties object that can be used as needed
        :rtype: Properties
        """
        LOGGER.info('Searching for custom config = {}', prop_file_name)
        # walk through the directory tree and try to locate correct resource suggest
        path_to_app_config = None
        found_resource = False
        for dirpath, dirnames, files in os.walk('.', topdown=False):
            if prop_file_name in files:
                path_to_app_config = os.path.join(dirpath, prop_file_name)
                found_resource = True
                LOGGER.info('Using custom config = {}', path_to_app_config)
                break
        if not found_resource:
            path_to_app_config = os.path.join(self.get_subscriber('proj_handler').get_proj_root(), prop_file_name)
            LOGGER.warning('Will attempt to load custom config = {}', path_to_app_config)
        try:
            custom_props = Properties()
            LOGGER.info('Loading {}', path_to_app_config)
            with open(path_to_app_config, 'rb') as prop_file:
                custom_props.load(prop_file)
        except Exception as ex:
            LOGGER.exception(ex)
            raise PropertiesException(ex)
        # log message on completion
        LOGGER.info('Custom properties loaded = {}', path_to_app_config)
        return custom_props

    def get_subscriber(self, handler: str)->AppPropertiesHandler:
        assert handler is not None and not (not handler), 'A valid properties handler name is required'
        prop_handler = self.handlers__.get(handler)
        if prop_handler is not None:
            return prop_handler
        handler_prop = self.get_dict_properties('project.properties.' + handler)
        if handler_prop is None:
            LOGGER.error('No properties found for handler = {}', handler)
            raise PropertiesException(f'No properties found for handler = {handler}')
        handler_module = handler_prop.get('module_name')
        handler_package = handler_prop.get('module_package')
        handler_class = getattr(importlib.import_module(handler_package), handler_module)
        try:
            prop_handler = handler_class(handler)
            self.subscribe(prop_handler)
            return prop_handler
        except TypeError as err:
            LOGGER.error('Problem encountered instantiating properties handler: {}, {}', handler_module, err)
            raise PropertiesException('Problem encountered instantiating properties handler: {}', handler_module)

    def __load(self)->None:
        """
        Load the underlying properties file from the project data folder or any project subfolder.
        :return:
        :rtype:
        """
        # prepare to load app-config.properties
        LOGGER.info('Searching for application config = {}', APP_CONFIG)
        # walk through the directory tree and try to locate correct resource suggest
        found_resource = False
        path_to_app_config = None
        for dirpath, dirnames, files in os.walk('.', topdown=False):
            if APP_CONFIG in files:
                path_to_app_config = os.path.join(dirpath, APP_CONFIG)
                found_resource = True
                LOGGER.info('Using project-specific application config = {}', path_to_app_config)
                break
        if not found_resource:
            path_to_app_config = os.path.join(self.get_subscriber('proj_handler').get_proj_root(), APP_CONFIG)
            LOGGER.warning('Attempting to use global application config = {}', path_to_app_config)
        try:
            self.app_props__ = Properties()
            LOGGER.info('Loading {}', path_to_app_config)
            with open(path_to_app_config, 'rb') as prop_file:
                self.app_props__.load(prop_file)
        except Exception as ex:
            LOGGER.exception(ex)
            raise PropertiesException(ex)
        # log message on completion
        LOGGER.info('Application properties loaded = {}', path_to_app_config)

    def subscribe(self, handler: AppPropertiesHandler):
        """
        Adds the specified handler to the subscribers list, indicating that the handlers properties will be refreshed when a global refresh is triggered.
        :param handler: specified handler interested in receiving notifications when properties are refreshed.
        :type handler:
        :return:
        :rtype:
        """
        if handler is not None:
            self.handlers__[handler.get_name()] = handler

    def unsubscribe(self, handler: AppPropertiesHandler):
        """
        Removes the specified handler from the subscribers' list, indicating that the handlers properties will no longer be refreshed when a global refresh is triggered.
        :param handler:
        :type handler:
        :return:
        :rtype:
        """
        if handler is not None:
            popped_handler = self.handlers__.pop(handler.get_name(), None)
            LOGGER.debug('Removed handler = {}', popped_handler)

    def __notify_handlers(self):
        """
        Triggers a reload on each of its subscribers.
        :return:
        :rtype:
        """
        for handler in self.handlers__.values():
            handler.reload()
        # reset project namespace accordingly
        LoguruWrapper().set_prefix(prefix=self.get_subscriber('proj_handler').get_proj_namespace())