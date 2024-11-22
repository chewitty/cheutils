import importlib
from IPython.display import display
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties

LOGGER = LoguruWrapper().get_logger()
APP_PROPS = AppProperties()
CONFIG_TRANSFORMERS = APP_PROPS.get_dict_properties('model.selectivescaler.transformers')

from cheutils.project_tree import save_to_html

def show_pipeline(pipeline, name: str='pipeline.png', save_to_file: bool=False):
    # Review the pipeline
    set_config(display='diagram')
    # with display='diagram', simply use display() to see the diagram
    display(pipeline)
    # if desired, set display back to the default
    set_config(display='text')
    # save it to file
    if save_to_file:
        save_to_html(pipeline, file_name=name)
        LOGGER.debug('Pipeline diagram saved to file: {}', name)

def get_column_transformers(remainder: str='passthrough', force_int_remainder_cols: bool=False):
    """
    Prepare and return a ColumnTransformer object with the specified transformers.
    :param remainder: By default, only the specified columns in transformers are transformed and combined in the
    output, and the non-specified columns are dropped. (default of 'drop'). By specifying remainder='passthrough',
    all remaining columns that were not specified in transformers, but present in the data passed to fit will be
    automatically passed through
    :type remainder:
    :param force_int_remainder_cols: Force the columns of the last entry of transformers_, which corresponds to the
    “remainder” transformer, to always be stored as indices (int) rather than column names (str). See description of
    the transformers_ attribute for details.
    :type force_int_remainder_cols:
    :return: ColumnTransformer object with specified transformers, or None if no transformers are configured.
    :rtype:
    """
    if (CONFIG_TRANSFORMERS is not None) or not (not CONFIG_TRANSFORMERS):
        LOGGER.debug('Configured column transformers: {}', CONFIG_TRANSFORMERS)
        transformers = []
        for item in CONFIG_TRANSFORMERS.values():
            name = item.get('name')
            tf_params = item.get('transformer_params')
            cols = list(item.get('columns'))
            tf_class = getattr(importlib.import_module(item.get('transformer_package')),
                               item.get('transformer_name'))
            try:
                tf = tf_class(**tf_params)
            except TypeError as err:
                LOGGER.debug('Failure encountered instantiating transformer: {}', name)
                raise KeyError('Unspecified or unsupported transformer')
            transformers.append((name, tf, cols))
        return ColumnTransformer(transformers=transformers, remainder=remainder,
                                 force_int_remainder_cols=force_int_remainder_cols, verbose_feature_names_out=False)
    return None
