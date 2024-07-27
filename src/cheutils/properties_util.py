import datetime
import os

from cheutils.debugger import Debugger
from cheutils.decorator_debug import debug_func
from cheutils.decorator_singleton import singleton
from cheutils.project_tree import get_data_dir, get_root_dir
from jproperties import Properties

APP_CONFIG_FILENAME = 'app-config.properties'
DBUGGER = Debugger()
class PropertiesException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
        super().__init__(self.message)

    def __str__(self):
        if self.message:
            return 'PropertiesException, {0} '.format(self.message)
        else:
            return 'PropertiesException raised'

"""
Utilities for reading properties files
"""
@singleton
class AppProperties(object):
    instance__ = None
    app_props__ = None
    '''
    A static method responsible for creating and returning a new instance (called before __init__)
    '''
    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created, 
        or else returns the previous singleton object
        """
        if AppProperties.instance__ is None:
            AppProperties.instance__ = super().__new__(cls)
        return AppProperties.instance__

    '''
    An instance method, the class constructor, responsible for initializing the attributes of the newly created
    '''

    @debug_func(enable_debug=True, prefix='app_config')
    def __init__(self, *args, **kwargs):
        """
        Initializes the properties utility.
        """
        # prepare to load app-config.properties
        path_to_app_config = os.path.join(get_data_dir(), APP_CONFIG_FILENAME)
        DBUGGER.debug('Desired application config', path_to_app_config)
        # walk through the directory tree and try to locate correct resource suggest
        found_resource = False
        for dirpath, dirnames, files in os.walk('.', topdown=False):
            if dirpath.endswith('data'):
                if APP_CONFIG_FILENAME in files:
                    path_to_app_config = os.path.join(dirpath, APP_CONFIG_FILENAME)
                    found_resource = True
                    DBUGGER.debug('Using project-specific application config', path_to_app_config)
                    break
        if not found_resource:
            DBUGGER.debug('Using global application config', path_to_app_config)
            path_to_app_config = os.path.join(get_root_dir(), APP_CONFIG_FILENAME)
        try:
            self.app_props__ = Properties()
            DBUGGER.debug('Loading', path_to_app_config)
            with open(path_to_app_config, 'rb') as prop_file:
                self.app_props__.load(prop_file)                
        except Exception as ex:
            raise PropertiesException(ex)
        # log message on completion
        DBUGGER.debug('Application properties loaded', path_to_app_config)

    def __str__(self):
        path_to_app_config = os.path.join(get_data_dir(), APP_CONFIG_FILENAME)
        info = 'AppProperties created, using properties file' + path_to_app_config
        DBUGGER.debug(info)
        return info
    
    '''
    Get the value associated with the specified key.
    '''
    def get(self, prop_key=None):
        """
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

    '''
        Get the value associated with the specified key.
        '''

    def get_bol(self, prop_key=None):
        """
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

    '''
    Get the list of keys held in the properties file.
    '''
    def get_keys(self):
        """
        Returns the list of keys as a list of strings.
        :return:
            list(str): a list of keys
        """
        items_view = self.app_props__.items()
        list_keys = []
        for item in items_view:
            list_keys.append(item[0])
        return list_keys

    '''
        Get the value associated with the specified key as a list of strings.
    '''
    def get_list(self, prop_key=None):
        """
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

    '''
        Get the value associated with the specified key as a bool, based on flags set on/off in the properties file.
    '''
    def is_set(self, prop_key=None):
        """
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
        DBUGGER.debug('Flags', key_val_pairs)
        key_set = key_val_pairs.get(key_part)
        if key_set is None:
            return False
        else:
            return key_set

    '''
        Get the flag values associated with the specified key as a dict of bools, based on flags set on/off in the properties file.
    '''
    def get_flags(self, prop_key=None):
        """
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(bool): a dict of on (1) or off (0) flags based on the specified key; the default is off (0).
        """
        if prop_key is None:
            return None
        prop_value = self.app_props__.get(prop_key).data
        DBUGGER.debug('Key-value property stem', prop_key)
        if prop_value is None:
            return None
        tmp_list = prop_value.replace('\'', '').replace('\"', '').strip('][').split(',')
        flags = {}
        for item in tmp_list:
            val_pair = item.split('=')
            val_pair = [x.replace('\'', '').replace('\"', '').strip('') for x in val_pair]
            flags[val_pair[0].strip()] = bool(eval(val_pair[1].strip()))
        return flags

    '''
        Get the key-value pairs as value associated with the specified key as a dict of strings set as key=value in the properties file.
    '''
    def get_properties(self, prop_key=None):
        """
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(str): a dict of string key-value pairs based on the specified key; the default is None.
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
            properties[val_pair[0].strip()] = val_pair[1].strip()
        return properties

    def get_dict_properties(self, prop_key=None):
        """
        Parameters:
            prop_key(str): the full property name, as in the properties file, for which a value is required
        Returns:
            dict(str): a dict of string key-value pairs based on the specified key; the default is None.
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
        prop_dict = self.get_dict_properties(prop_key)
        prop_ranges = {}
        for param, value in prop_dict.items():
            min_val = int(value.get('start')) if value is not None else value
            max_val = int(value.get('end')) if value is not None else value
            prop_ranges[param] = (min_val, max_val)
        return prop_ranges

    '''
        Get the property value associated with the specified key, in a list of key-value pairs in the properties file.
    '''
    def get_property(self, prop_key=None):
        """
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
        #DBUGGER.debug('Properties', key_val_pairs)
        val_part = key_val_pairs.get(key_part)
        if val_part is None:
            return None
        else:
            return val_part
            
    '''
        Get the key-value pairs as value associated with the specified key as a dict of types set as key=value in the properties file.
    '''
    def get_types(self, prop_key=None):
        """
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