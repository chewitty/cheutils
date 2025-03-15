import os
import importlib
import pandas as pd
import numpy as np
import geolib.geohash as gh
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from cheutils.decorator_singleton import singleton
from cheutils.common_utils import apply_clipping, parse_special_features, safe_copy, get_outlier_cat_thresholds, get_quantiles
from cheutils.project_tree import save_excel
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties, AppPropertiesHandler
from cheutils.exceptions import PropertiesException, FeatureGenException
from cheutils.data_prep_support import apply_replace_patterns, apply_calc_feature, apply_type, force_joblib_cleanup
import tsfresh.defaults
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import restrict_input_to_index
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_categorical_dtype
from scipy.stats import iqr

LOGGER = LoguruWrapper().get_logger()

@singleton
class DataPrepProperties(AppPropertiesHandler):
    __app_props: AppProperties

    def __init__(self, name: str=None):
        super().__init__(name=name)
        self.__data_prep_properties = {}
        self.__app_props = AppProperties()

    # overriding abstract method
    def reload(self):
        self.__data_prep_properties = {}
        """for key in self.__data_prep_properties.keys():
            try:
                self.__data_prep_properties[key] = self._load(prop_key=key)
            except Exception as err:
                LOGGER.warning('Problem reloading property: {}, {}', key, err)
                pass"""

    def _load(self, prop_key: str=None):
        LOGGER.debug('Attempting to load data property: {}', prop_key)
        return getattr(self, '_load_' + prop_key, lambda: 'unspecified')()

    def __getattr__(self, item):
        msg = f'Attempting to load unspecified data property: {item}'
        LOGGER.warning(msg)

    def _load_unspecified(self):
        raise PropertiesException('Attempting to load unspecified data property')

    def _load_selective_column_transformers(self):
        key = 'model.selective_column.transformers'
        conf_pipelines = self.__app_props.get_list_properties(key)
        if (conf_pipelines is not None) and not (not conf_pipelines):
            LOGGER.debug('Preparing configured column transformer pipelines: \n{}', conf_pipelines)
            col_transformers = []
            for pipeline in conf_pipelines:
                if pipeline is None:
                    break
                pipe_name = pipeline.get('pipeline_name')
                pipeline_tfs = pipeline.get('transformers') # list of transformers
                pipeline_cols = pipeline.get('columns') # columns mapped to the pipeline
                if pipeline_cols is None or (not pipeline_cols):
                    continue
                pipeline_steps = []
                for item in pipeline_tfs:
                    tf_name = item.get('name')
                    tf_module = item.get('module')
                    tf_package = item.get('package')
                    tf_params = item.get('params')
                    tf_params = {} if tf_params is None or (not tf_params) else tf_params
                    tf_class = getattr(importlib.import_module(tf_package), tf_module)
                    try:
                        tf = tf_class(**tf_params)
                        pipeline_steps.append((tf_name, tf))
                    except TypeError as err:
                        LOGGER.error('Problem encountered instantiating transformer: {}, {}', tf_name, err)
                col_pipeline: Pipeline = Pipeline(steps=pipeline_steps)
                col_transformers.append((pipe_name, col_pipeline, pipeline_cols))
            self.__data_prep_properties['selective_column_transformers'] = col_transformers

    def _load_binarizer_column_transformers(self):
        key = 'model.binarizer_column.transformers'
        conf_pipelines = self.__app_props.get_list_properties(key)
        if (conf_pipelines is not None) and not (not conf_pipelines):
            LOGGER.debug('Preparing configured binarizer transformer pipelines: \n{}', conf_pipelines)
            col_transformers = []
            for pipeline in conf_pipelines:
                if pipeline is None:
                    break
                pipe_name = pipeline.get('pipeline_name')
                pipeline_tfs = pipeline.get('transformers') # list of transformers
                pipeline_cols = pipeline.get('columns') # columns mapped to the pipeline
                if pipeline_cols is None or (not pipeline_cols):
                    continue
                pipeline_steps = []
                for item in pipeline_tfs:
                    tf_name = item.get('name')
                    tf_module = item.get('module')
                    tf_package = item.get('package')
                    tf_params = item.get('params')
                    tf_params = {} if tf_params is None or (not tf_params) else tf_params
                    tf_class = getattr(importlib.import_module(tf_package), tf_module)
                    try:
                        tf = tf_class(**tf_params)
                        pipeline_steps.append((tf_name, tf))
                    except TypeError as err:
                        LOGGER.error('Problem encountered instantiating transformer: {}, {}', tf_name, err)
                col_pipeline: Pipeline = Pipeline(steps=pipeline_steps)
                col_transformers.append((pipe_name, col_pipeline, pipeline_cols))
            self.__data_prep_properties['binarizer_column_transformers'] = col_transformers

    def _load_feat_sel_transformers(self, estimator):
        assert estimator is not None, 'A valid feature section estimator must be provided'
        key = 'model.feat_selection.transformers'
        conf_transformers = self.__app_props.get_dict_properties(key)
        if (conf_transformers is not None) and not (not conf_transformers):
            LOGGER.debug('Preparing configured feature selection transformer options: \n{}', conf_transformers)
            col_transformers = {}
            for selector, transformer_conf in conf_transformers.items():
                if transformer_conf is None or (not transformer_conf):
                    continue
                tf_module = transformer_conf.get('module')
                tf_package = transformer_conf.get('package')
                tf_params = transformer_conf.get('params')
                tf_params = {} if tf_params is None or (not tf_params) else tf_params
                tf_class = getattr(importlib.import_module(tf_package), tf_module)
                col_transformers[selector] = (tf_class, tf_params)
            self.__data_prep_properties['feat_sel_transformers'] = col_transformers

    def _load_target_encoder(self):
        key = 'model.target.encoder'
        conf_pipeline = self.__app_props.get_dict_properties(key)
        if (conf_pipeline is not None) and not (not conf_pipeline):
            LOGGER.debug('Preparing configured target encoder pipeline: \n{}', conf_pipeline)
            target_encs = [] # tg_encoders
            pipe_name = conf_pipeline.get('pipeline_name')
            pipeline_tg_enc = conf_pipeline.get('target_encoder') # a single target encoder
            pipeline_cols = conf_pipeline.get('columns') # columns mapped to the pipeline
            if pipeline_cols is None or (not pipeline_cols):
                pipeline_cols = []
            pipeline_steps = []
            if pipeline_tg_enc is not None:
                tf_name = pipeline_tg_enc.get('name')
                tf_module = pipeline_tg_enc.get('module')
                tf_package = pipeline_tg_enc.get('package')
                tf_params = pipeline_tg_enc.get('params')
                tf_params = {} if tf_params is None or (not tf_params) else tf_params
                tf_class = getattr(importlib.import_module(tf_package), tf_module)
                tf_obj = None
                try:
                    tf_obj = tf_class(**tf_params)
                    #tf_obj.set_output(transform='pandas')
                    pipeline_steps.append((tf_name, tf_obj))
                except TypeError as err:
                    LOGGER.error('Problem encountered instantiating target encoder: {}, {}', tf_name, err)
                tg_enc_pipeline: Pipeline = Pipeline(steps=pipeline_steps)
                target_encs.append((pipe_name, tg_enc_pipeline, pipeline_cols))
            self.__data_prep_properties['target_encoder'] = target_encs

    def _load_sqlite3_db(self):
        key = 'project.sqlite3.db'
        self.__data_prep_properties['sqlite3_db'] = self.__app_props.get(key)

    def _load_winsorize_limits(self):
        key = 'func.winsorize.limits'
        limits = self.__app_props.get_list(key)
        if limits is not None and not (not limits):
            self.__data_prep_properties['winsorize_limits'] = [float(item) for item in limits if limits is not None]

    def _load_ds_props(self, ds_config_file_name: str=None):
        LOGGER.debug('Attempting to load datasource properties: {}', ds_config_file_name)
        return getattr(self.__app_props, 'load_custom_properties', lambda: 'unspecified')(ds_config_file_name)

    def _load_replace_table(self, ds_namespace: str, tb_name: str):
        key = 'db.to_tables.replace.' + ds_namespace + '.' + tb_name
        prop_key = 'replace_table_' + ds_namespace + '_' + tb_name
        self.__data_prep_properties[prop_key] = self.__app_props.get_bol(key)

    def _load_delete_by(self, ds_namespace: str, tb_name: str):
        key = 'db.to_tables.replace.' + ds_namespace + '.' + tb_name
        prop_key = 'delete_table_' + ds_namespace + '_' + tb_name
        self.__data_prep_properties[prop_key] = self.__app_props.get_properties(key)

    def get_selective_column_transformers(self):
        value = self.__data_prep_properties.get('selective_column_transformers')
        if value is None:
            self._load_selective_column_transformers()
        return self.__data_prep_properties.get('selective_column_transformers')

    def get_binarizer_column_transformers(self):
        value = self.__data_prep_properties.get('binarizer_column_transformers')
        if value is None:
            self._load_binarizer_column_transformers()
        return self.__data_prep_properties.get('binarizer_column_transformers')

    def get_feat_sel_transformers(self, estimator):
        value = self.__data_prep_properties.get('feat_sel_transformers')
        if value is None:
            self._load_feat_sel_transformers(estimator)
        return self.__data_prep_properties.get('feat_sel_transformers')

    def get_target_encoder(self):
        value = self.__data_prep_properties.get('target_encoder')
        if value is None:
            self._load_target_encoder()
        return self.__data_prep_properties.get('target_encoder')

    def get_sqlite3_db(self):
        value = self.__data_prep_properties.get('sqlite3_db')
        if value is None:
            self._load_sqlite3_db()
        return self.__data_prep_properties.get('sqlite3_db')

    def get_winsorize_limits(self):
        value = self.__data_prep_properties.get('winsorize_limits')
        if value is None:
            self._load_winsorize_limits()
        return self.__data_prep_properties.get('winsorize_limits')

    def get_ds_config(self, ds_key: str, ds_config_file_name: str):
        assert ds_key is not None and not (not ds_key), 'A valid datasource key or name required'
        assert ds_config_file_name is not None and not (not ds_config_file_name), 'A valid datasource file name required'
        value = self.__data_prep_properties.get(ds_key)
        if value is None:
            self.__data_prep_properties[ds_key] = self._load_ds_props(ds_config_file_name=ds_config_file_name)
        return self.__data_prep_properties.get(ds_key)

    def get_replace_tb(self, ds_namespace: str, tb_name: str):
        assert ds_namespace is not None and not (not ds_namespace), 'A valid namespace is required'
        assert tb_name is not None and not (not tb_name), 'A valid table name required'
        prop_key = 'replace_table_' + ds_namespace + '_' + tb_name
        value = self.__data_prep_properties.get(prop_key)
        if value is None:
            self._load_replace_table(ds_namespace=ds_namespace, tb_name=tb_name)
        return self.__data_prep_properties.get(prop_key)

    def get_delete_by(self, ds_namespace: str, tb_name: str):
        assert ds_namespace is not None and not (not ds_namespace), 'A valid namespace is required'
        assert tb_name is not None and not (not tb_name), 'A valid table name required'
        prop_key = 'delete_table_' + ds_namespace + '_' + tb_name
        value = self.__data_prep_properties.get(prop_key)
        if value is None:
            self._load_delete_by(ds_namespace=ds_namespace, tb_name=tb_name)
        return self.__data_prep_properties.get(prop_key)

class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms datetimes, generating additional prefixed 'dow', 'wk', 'month', 'qtr', 'wkend' features for all relevant columns
    (specified) in the dataframe; drops the datetime column by default but can be retained as desired.
    """
    def __init__(self, rel_cols: list, prefixes: list, drop_rel_cols: list=None, **kwargs):
        """
        Transforms datetimes, generating additional prefixed 'dow', 'wk', 'month', 'qtr', 'wkend' features for all relevant
        columns (specified) in the dataframe; drops the datetime column by default but can be retained as desired.
        :param rel_cols: the column labels for desired datetime columns in the dataframe
        :type rel_cols: list
        :param prefixes: the corresponding prefixes for the specified datetime columns, e.g., 'date_'
        :type prefixes: list
        :param drop_rel_cols: the coresponding list of index matching flags indicating whether to drop the original
        datetime column or not; if not specified, defaults to True for all specified columns
        :type drop_rel_cols: list
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)
        self.target = None
        self.rel_cols = rel_cols
        self.prefixes = prefixes
        self.drop_rel_cols = drop_rel_cols

    def fit(self, X, y=None):
        LOGGER.debug('DateFeaturesTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y  # possibly passed in chain
        return self

    def transform(self, X, y=None):
        LOGGER.debug('DateFeaturesTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y,)
        LOGGER.debug('DateFeaturesTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('DateFeaturesTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('DateFeaturesTransformer: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X = safe_copy(X)
        new_X.reset_index(drop=True, inplace=True)
        # otherwise also generate the following features
        for rel_col, prefix in zip(self.rel_cols, self.prefixes):
            if not is_datetime64_any_dtype(new_X[rel_col]):
                new_X[rel_col] = pd.to_datetime(new_X[rel_col], errors='coerce', utc=True)  # to be absolutely sure it is datetime
            new_X.loc[:, prefix + 'dow'] = new_X[rel_col].dt.dayofweek
            null_dayofweek = new_X[prefix + 'dow'].isna()
            nulldofwk = new_X[null_dayofweek]
            new_X[prefix + 'dow'] = new_X[prefix + 'dow'].astype(int)
            new_X.loc[:, prefix + 'wk'] = new_X[rel_col].apply(lambda x: pd.Timestamp(x).week)
            new_X[prefix + 'wk'] = new_X[prefix + 'wk'].astype(int)
            new_X.loc[:, prefix + 'month'] = new_X[rel_col].dt.month
            new_X[prefix + 'month'] = new_X[prefix + 'month'].astype(int)
            new_X.loc[:, prefix + 'qtr'] = new_X[rel_col].dt.quarter
            new_X[prefix + 'qtr'] = new_X[prefix + 'qtr'].astype(int)
            new_X.loc[:, prefix + 'wkend'] = np.where(new_X[rel_col].dt.dayofweek.isin([5, 6]), 1, 0)
            new_X[prefix + 'wkend'] = new_X[prefix + 'wkend'].astype(int)
            new_X.loc[:, prefix + 'd15'] = np.where(new_X[rel_col].dt.dayofweek == 15, 1, 0)
            new_X[prefix + 'd15'] = new_X[prefix + 'd15'].astype(int)
            new_X.loc[:, prefix + 'eom'] = new_X[rel_col].dt.is_month_end
            new_X[prefix + 'eom'] = new_X[prefix + 'eom'].astype(int)
        if len(self.rel_cols) > 0:
            if self.drop_rel_cols is None or not self.drop_rel_cols:
                new_X.drop(columns=self.rel_cols, inplace=True)
            else:
                to_drop_cols = []
                for index, to_drop_col in enumerate(self.rel_cols):
                    if self.drop_rel_cols[index]:
                        to_drop_cols.append(to_drop_col)
                new_X.drop(columns=to_drop_cols, inplace=True)
        return new_X

    def get_date_cols(self):
        """
        Returns the transformed date columns, if any
        :return:
        """
        return self.rel_cols

    def get_target(self):
        return self.target
def feature_selection_transformer(selector: str, estimator, passthrough: bool=False, ):
    """
    Meta-transformer for selecting features based on recursive feature selection, select from model, or any other equivalent class.
    @see https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python?noredirect=1&lq=1
    """
    assert selector is not None, 'A valid configured feature selector option must be provided'
    __data_handler: DataPrepProperties = AppProperties().get_subscriber('data_handler')
    tf_base_class, tf_params = __data_handler.get_feat_sel_transformers(estimator).get(selector) # a tuple (trans_class, trans_params)
    class FeatureSelectionTransformer(tf_base_class):
        """
        Returns features based on ranking with recursive feature elimination.
        """
        def __init__(self, estimator=None, passthrough: bool=False, random_state: int=100, **kwargs):
            self.random_state = random_state
            self.estimator = estimator
            super().__init__(self.estimator, ** kwargs)
            self.target = None
            self.selected_cols = None
            self.passthrough = passthrough
            self.fitted = False
            self.selector = None

        def fit(self, X, y=None, **fit_params):
            if self.passthrough:
                self.selected_cols = list(X.columns)
            if self.fitted and not self.passthrough:
                return self
            LOGGER.debug('FeatureSelectionTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            self.target = y  # possibly passed in chain
            super().fit(X, y, **fit_params)
            self.fitted = True
            return self

        def transform(self, X, y=None, **fit_params):
            LOGGER.debug('FeatureSelectionTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            new_X = self.__do_transform(X, y=None)
            LOGGER.debug('FeatureSelectionTransformer: Transformed dataset, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
            return new_X

        def fit_transform(self, X, y=None, **fit_params):
            LOGGER.debug('FeatureSelectionTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            self.fit(X, y, **fit_params)
            self.target = y
            new_X = self.__do_transform(X, y, **fit_params)
            LOGGER.debug('FeatureSelectionTransformer: Fit-transformed dataset, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
            return new_X

        def __do_transform(self, X, y=None, **fit_params):
            if self.selected_cols is None and not self.passthrough:
                if y is None:
                    transformed_X = super().transform(X)
                else:
                    transformed_X = super().fit_transform(X, y, **fit_params)
                self.selected_cols = list(X.columns[self.get_support()])
                new_X = pd.DataFrame(transformed_X, columns=self.selected_cols)
                self.__save_importances()
            else:
                new_X = pd.DataFrame(X, columns=self.selected_cols)
            for col in self.selected_cols:
                try:
                    new_X[col] = pd.to_numeric(new_X[col], )
                except ValueError as ignore:
                    LOGGER.warning('Potential dtype issue: {}', ignore)
            return new_X

        def get_selected_features(self):
            """
            Return the selected features or column labels.
            :return:
            """
            return self.selected_cols

        def get_target(self):
            return self.target

        def __save_importances(self):
            try:
                importances = None
                try:
                    importances = self.estimator_.coef_
                except AttributeError:
                    importances = self.estimator_.feature_importances_
                if importances is not None:
                    importances = pd.Series(importances[:len(self.selected_cols)], index=self.selected_cols, name='importance')
                    save_excel(importances, file_name='feature_importances.xlsx', tag_label=selector, index=True)
            except Exception as ignore:
                LOGGER.warning('Could not save feature importances: {}', ignore)

    tf_instance = None
    try:
        tf_instance = FeatureSelectionTransformer(estimator=estimator, passthrough=passthrough, **tf_params, )
        tf_instance.selector = selector
    except TypeError as err:
        LOGGER.error('Problem encountered instantiating feature selection transformer: {}, {}', selector, err)
    return tf_instance

class DropSelectedColsTransformer(BaseEstimator, TransformerMixin):
    """
    Drops selected columns from the dataframe.
    """
    def __init__(self, rel_cols: list, **kwargs):
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.target = None
        self.fitted = False

    def fit(self, X, y=None):
        if self.fitted:
            return self
        LOGGER.debug('DropSelectedColsTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        self.fitted = True
        return self

    def transform(self, X, y=None):
        LOGGER.debug('DropSelectedColsTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        new_X = self.__do_transform(X, y)
        LOGGER.debug('DropSelectedColsTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        LOGGER.debug('DropSelectedColsTransformer: Columns dropped = {}', self.rel_cols)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        LOGGER.debug('DropSelectedColsTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, **fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X = X.drop(columns=self.rel_cols) if self.rel_cols is not None and not (not self.rel_cols) else X
        return new_X

    def get_target(self):
        """
        Returns the transformed target if any
        :return:
        """
        return self.target

class SelectiveColumnTransformer(ColumnTransformer):
    def __init__(self, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        # if configuring more than one column transformer make sure verbose_feature_names_out=True
        # to ensure the prefixes ensure uniqueness in the feature names
        __data_handler: DataPrepProperties = AppProperties().get_subscriber('data_handler')
        conf_transformers = __data_handler.get_selective_column_transformers()
        super().__init__(transformers=conf_transformers,
                         remainder=remainder, force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.num_transformers = len(conf_transformers)
        self.feature_names = None

    def fit(self, X, y=None, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        super().fit(X, y, **fit_params)
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('SelectiveColumnTransformer: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('SelectiveColumnTransformer: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if y is None:
            transformed_X = super().transform(X, **fit_params)
        else:
            transformed_X = super().fit_transform(X, y, **fit_params)
        feature_names_out = super().get_feature_names_out().tolist()
        if self.num_transformers > 1:
            feature_names_out.reverse()
            # sort out any potential duplicates - noting how column transformers concatenate transformed and
            # passthrough columns
            feature_names = [feature_name.split('__')[-1] for feature_name in feature_names_out]
            duplicate_feature_idxs = []
            desired_feature_names_s = set()
            desired_feature_names = []
            for idx, feature_name in enumerate(feature_names):
                if feature_name not in desired_feature_names_s:
                    desired_feature_names_s.add(feature_name)
                    desired_feature_names.append(feature_name)
                else:
                    duplicate_feature_idxs.append(idx)
            desired_feature_names.reverse()
            duplicate_feature_idxs = [len(feature_names) - 1 - idx for idx in duplicate_feature_idxs]
            if duplicate_feature_idxs is not None and not (not duplicate_feature_idxs):
                transformed_X = np.delete(transformed_X, duplicate_feature_idxs, axis=1)
        else:
            desired_feature_names = feature_names_out
        desired_feature_names = [feature_name.split('__')[-1] for feature_name in desired_feature_names]
        new_X = pd.DataFrame(transformed_X, columns=desired_feature_names)
        self.feature_names = desired_feature_names
        return new_X

class BinarizerColumnTransformer(ColumnTransformer):
    def __init__(self, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        # if configuring more than one column transformer make sure verbose_feature_names_out=True
        # to ensure the prefixes ensure uniqueness in the feature names
        __data_handler: DataPrepProperties = AppProperties().get_subscriber('data_handler')
        conf_transformers = __data_handler.get_binarizer_column_transformers()
        super().__init__(transformers=conf_transformers,
                         remainder=remainder, force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True if len(conf_transformers) > 1 else False,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.num_transformers = len(conf_transformers)
        self.feature_names = None

    def fit(self, X, y=None, **fit_params):
        LOGGER.debug('BinarizerColumnTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        super().fit(X, y, **fit_params)
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('BinarizerColumnTransformer: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('BinarizerColumnTransformer: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('BinarizerColumnTransformer: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('BinarizerColumnTransformer: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if y is None:
            transformed_X = super().transform(X, **fit_params)
        else:
            transformed_X = super().fit_transform(X, y, **fit_params)
        feature_names_out = super().get_feature_names_out().tolist()
        if self.num_transformers > 1:
            feature_names_out.reverse()
            # sort out any potential duplicates - noting how column transformers concatenate transformed and
            # passthrough columns
            feature_names = [feature_name.split('__')[-1] for feature_name in feature_names_out]
            duplicate_feature_idxs = []
            desired_feature_names_s = set()
            desired_feature_names = []
            for idx, feature_name in enumerate(feature_names):
                if feature_name not in desired_feature_names_s:
                    desired_feature_names_s.add(feature_name)
                    desired_feature_names.append(feature_name)
                else:
                    duplicate_feature_idxs.append(idx)
            desired_feature_names.reverse()
            duplicate_feature_idxs = [len(feature_names) - 1 - idx for idx in duplicate_feature_idxs]
            transformed_X = np.delete(transformed_X, duplicate_feature_idxs, axis=1)
        else:
            desired_feature_names = feature_names_out
        new_X = pd.DataFrame(transformed_X, columns=desired_feature_names)
        self.feature_names = desired_feature_names
        return new_X

class DataPrepTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols: list=None, int_cols: list=None, float_cols: list=None,
                 masked_cols: dict=None, special_features: dict=None, drop_feats_cols: bool=True,
                 calc_features: dict=None, synthetic_features: dict=None, lag_features: dict=None,
                 gen_target: dict=None, correlated_cols: list=None, replace_patterns: list=None,
                 gen_cat_col: dict=None, pot_leak_cols: list=None, clip_data: dict=None,
                 include_target: bool=False, **kwargs):
        """
        Preprocessing dataframe columns to ensure consistent data types and formatting, and optionally extracting any
        special features described by dictionaries of feature mappings - e.g.,
        special_features = {'col_label1': {'feat_mappings': {'Trailers': 'trailers', 'Deleted Scenes': 'deleted_scenes', 'Behind the Scenes': 'behind_scenes', 'Commentaries': 'commentaries'}, 'sep': ','}, }.
        :param date_cols: any date columns to be concerted to datetime
        :type date_cols:
        :param int_cols: any int columns to be converted to int
        :type int_cols:
        :param float_cols: any float columns to be converted to float
        :type float_cols:
        :param masked_cols: dictionary of columns and function generates a mask or a mask (bool Series) - e.g., {'col_label1': mask_func)
        :type masked_cols:
        :param special_features: dictionaries of feature mappings - e.g., special_features = {'col_label1': {'feat_mappings': {'Trailers': 'trailers', 'Deleted Scenes': 'deleted_scenes', 'Behind the Scenes': 'behind_scenes', 'Commentaries': 'commentaries'}, 'sep': ','}, }
        :type special_features:
        :param drop_feats_cols: drop special_features cols if True
        :param calc_features: dictionary of calculated column labels with their corresponding column generation functions - e.g., {'col_label1': {'func': col_gen_func1, 'is_numeric': True, 'inc_target': False, 'delay': False, 'kwargs': {}}, 'col_label2': {'func': col_gen_func2, 'is_numeric': True, 'inc_target': False, 'delay': False, 'kwargs': {}}
        :param synthetic_features: dictionary of calculated column labels with their corresponding column generation functions, for cases involving features not present in test data - e.g., {'new_col1': {'func': col_gen_func1, 'agg_col': 'col_label1', 'agg_func': 'median', 'id_by_col': 'id', 'sort_by_cols': 'date', 'inc_target': False, 'impute_agg_func': 'mean', 'kwargs': {}}, 'new_col2': {'func': col_gen_func2, 'agg_col': 'col_label2', 'agg_func': 'median', 'id_by_col': 'id', 'sort_by_cols': 'date', 'inc_target': False, 'impute_agg_func': 'mean', 'kwargs': {}}
        :param lag_features: dictionary of calculated column labels to hold lagging calculated values with their corresponding column lagging calculation functions - e.g., {'col_label1': {'filter_by': ['filter_col1', 'filter_col2'], period=0, 'drop_rel_cols': False, }, 'col_label2': {'filter_by': ['filter_col3', 'filter_col4'], period=0, 'drop_rel_cols': False, }}
        :param gen_target: dictionary of target column label and target generation function (e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'target_col': 'target_collabel', 'target_gen_func': target_gen_func, 'other_val': 0}
        :param correlated_cols: columns that are moderately to highly correlated and should be dropped
        :param gen_cat_col: dictionary specifying a categorical column label to be generated from a numeric column, with corresponding bins and labels - e.g., {'cat_col': 'num_col_label', 'bins': [1, 2, 3, 4, 5], 'labels': ['A', 'B', 'C', 'D', 'E']})
        :param pot_leak_cols: columns that could potentially introduce data leakage and should be dropped
        :param clip_data: clip outliers from the data based on categories defined by the filterby key and whether to enforce positive threshold defined by the pos_thres key - e.g., clip_data = {'rel_cols': ['col1', 'col2'], 'filterby': 'col_label1', 'pos_thres': False}
        :param include_target: include the target Series in the returned first item of the tuple if True (usually during exploratory analysis only); default is False (when as part of model pipeline)
        :param replace_patterns: list of dictionaries of pattern (e.g., regex strings with values as replacements) - e.g., [{'rel_col': 'col_with_strs', 'replace_dict': {}, 'regex': False, }]
        :param kwargs:
        :type kwargs:
        """
        self.date_cols = date_cols
        self.int_cols = int_cols
        self.float_cols = float_cols
        self.masked_cols = masked_cols
        self.special_features = special_features
        self.drop_feats_cols = drop_feats_cols
        self.gen_target = gen_target
        self.calc_features = calc_features
        self.synthetic_features = synthetic_features
        self.lag_features = lag_features
        self.correlated_cols = correlated_cols
        self.replace_patterns = replace_patterns
        self.gen_cat_col = gen_cat_col
        self.pot_leak_cols = pot_leak_cols
        self.clip_data = clip_data
        self.include_target = include_target
        self.target = None
        self.gen_calc_features = {} # to hold generated features from the training set - i.e., these features are generated during fit()
        self.gen_global_aggs = {}
        self.basic_calc_features = {}
        self.delayed_calc_features = {}
        self.transform_global_aggs = {}
        self.fitted = False

    def fit(self, X, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('DataPrepTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # do any necessary pre-processing
        new_X, new_y = self.__pre_process(X, y, date_cols=self.date_cols, int_cols=self.int_cols,
                                          float_cols=self.float_cols,
                                          masked_cols=self.masked_cols, special_features=self.special_features,
                                          drop_feats_cols=self.drop_feats_cols, gen_target=self.gen_target,
                                          gen_cat_col=self.gen_cat_col,
                                          clip_data=self.clip_data, include_target=self.include_target, )
        # sort of sequence of calculated features
        if self.calc_features is not None:
            for col, col_gen_func_dict in self.calc_features.items():
                delay_calc = col_gen_func_dict.get('delay')
                if delay_calc is not None and delay_calc:
                    self.delayed_calc_features[col] = col_gen_func_dict
                else:
                    self.basic_calc_features[col] = col_gen_func_dict
        # then, generate any features that may depend on synthetic features (i.e., features not present in test data)
        self.__gen_synthetic_features(new_X, new_y if new_y is not None else y)
        self.target = new_y if new_y is not None else y
        self.fitted = True
        return self

    def transform(self, X):
        LOGGER.debug('DataPrepTransformer: Transforming dataset, shape = {}', X.shape)
        # be sure to patch in any generated target column
        new_X, new_y = self.__do_transform(X)
        self.target = new_y if new_y is not None else self.target
        LOGGER.debug('DataPrepTransformer: Transformed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('DataPrepTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # be sure to patch in any generated target column
        self.fit(X, y, **fit_params)
        new_X, new_y = self.__do_transform(X, y)
        self.target = new_y
        LOGGER.debug('DataPrepTransformer: Fit-transformed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X

    def __generate_features(self, X: pd.DataFrame, y: pd.Series = None, gen_cols: dict = None, return_y: bool = False,
                          target_col: str = None, **kwargs) -> pd.DataFrame:
        """
        Generate the target variable from available data in X, and y.
        :param X: the raw input dataframe, may or may not contain the features that contribute to generating the target variable
        :type X:
        :param y: part or all of the raw target variable, may contribute to generating the actual target
        :type y:
        :param gen_cols: dictionary of new feature column labels and their corresponding value generation functions
            and default values - e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'feat_col': (val_gen_func, alter_val)}
        :type gen_cols: dict
        :param return_y: if True, add back a column with y or a modified version to the returned dataframe
        :param target_col: the column label of the target column - either as a hint or may be encountered as part of any generation function.
        :param kwargs:
        :type kwargs:
        :return: a dataframe with the generated features
        :rtype:
        """
        assert X is not None, 'A valid DataFrame expected as input'
        assert gen_cols is not None and not (
            not gen_cols), 'A valid dictionary of new feature column labels and their corresponding value generation functions and optional default values expected as input'
        new_X = safe_copy(X)
        # add back the target column, in case it is needed
        if y is not None:
            if isinstance(y, pd.Series):
                new_X[y.name] = safe_copy(y)
            else:
                if target_col is not None and not (not target_col):
                    new_X[target_col] = safe_copy(y)
        try:
            for col, val_gen_func in gen_cols.items():
                new_X[col] = new_X.apply(val_gen_func[0], axis=1)
                if val_gen_func[1] is not None:
                    new_X[col].fillna(val_gen_func[1], inplace=True)
            # drop the target column again
            if not return_y:
                if y is not None and isinstance(y, pd.Series):
                    new_X.drop(columns=[y.name], inplace=True)
                else:
                    if target_col is not None and not (not target_col):
                        if target_col in new_X.columns:
                            new_X.drop(columns=[target_col], inplace=True)
            return new_X
        except Exception as err:
            LOGGER.error('Something went wrong with feature generation, skipping: {}', err)
            raise FeatureGenException(f'Something went wrong with feature generation, skipping: {err}')

    def __generate_target(self, X: pd.DataFrame, y: pd.Series = None, gen_target: dict = None, include_target: bool = False,
                        **kwargs):
        """
        Generate the target variable from available data in X, and y.
        :param X: the raw input dataframe, may or may not contain the features that contribute to generating the target variable
        :type X:
        :param y: part or all of the raw target variable, may contribute to generating the actual target
        :type y:
        :param gen_target: dictionary of target column label and target generation function (e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'target_col': 'target_collabel', 'target_gen_func': target_gen_func}
        :type gen_target:
        :param include_target: include the target Series in the returned first item of the tuple if True; default is False
        :type include_target:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        assert X is not None, 'A valid DataFrame expected as input'
        new_X = X
        new_y = y
        try:
            if gen_target is not None:
                target_gen_col = {
                        gen_target.get('target_col'): (gen_target.get('target_gen_func'), gen_target.get('alter_val'))}
                new_X = self.__generate_features(new_X, new_y, gen_cols=target_gen_col, return_y=include_target,
                                          target_col=gen_target.get('target_col'), )
                new_y = new_X[gen_target.get('target_col')]
        except Exception as warnEx:
            LOGGER.warning('Something went wrong with target variable generation, skipping: {}', warnEx)
            pass
        return new_X, new_y

    def __pre_process(self, X, y=None, date_cols: list = None, int_cols: list = None, float_cols: list = None,
                      masked_cols: dict = None, special_features: dict = None, drop_feats_cols: bool = True,
                      gen_target: dict = None, replace_patterns: list = None, pot_leak_cols: list = None,
                      clip_data: dict = None, gen_cat_col: dict = None, include_target: bool = False, ):
        """
        Pre-process dataset by handling date conversions, type casting of columns, clipping data,
        generating special features, calculating new features, masking columns, dropping correlated
        and potential leakage columns, and generating target variables if needed.
        :param X: Input dataframe with data to be processed
        :param y: Optional target Series; default is None
        :param date_cols: any date columns to be concerted to datetime
        :type date_cols: list
        :param int_cols: Columns to be converted to integer type
        :type int_cols: list
        :param float_cols: Columns to be converted to float type
        :type float_cols: list
        :param masked_cols: dictionary of columns and function generates a mask or a mask (bool Series) - e.g., {'col_label1': mask_func)
        :type masked_cols: dict
        :param special_features: dictionaries of feature mappings - e.g., special_features = {'col_label1': {'feat_mappings': {'Trailers': 'trailers', 'Deleted Scenes': 'deleted_scenes', 'Behind the Scenes': 'behind_scenes', 'Commentaries': 'commentaries'}, 'sep': ','}, }
        :type special_features: dict
        :param drop_feats_cols: drop special_features cols if True
        :type drop_feats_cols: bool
        :param gen_target: dictionary of target column label and target generation function (e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'target_col': 'target_collabel', 'target_gen_func': target_gen_func}
        :type gen_target: dict
        :param clip_data: clip the data based on categories defined by the filterby key and whether to enforce positive threshold defined by the pos_thres key - e.g., clip_data = {'rel_cols': ['col1', 'col2'], 'filterby': 'col_label1', 'pos_thres': False}
        :type clip_data: dict
        :param gen_cat_col: dictionary specifying a categorical column label to be generated from a numeric column, with corresponding bins and labels - e.g., {'cat_col': 'num_col_label', 'bins': [1, 2, 3, 4, 5], 'labels': ['A', 'B', 'C', 'D', 'E']})
        :param include_target: include the target Series in the returned first item of the tuple if True; default is False
        :param replace_patterns: list of dictionaries of pattern (e.g., regex strings with values as replacements) - e.g., [{'rel_col': 'col_with_strs', 'replace_dict': {}, 'regex': False, }]
        :type replace_patterns: list
        :return: Processed dataframe and updated target Series
        :rtype: tuple(pd.DataFrame, pd.Series or None)
        """
        LOGGER.debug('DataPrepTransformer: Pre-processing dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X
        new_y = y
        # process columns with  strings to replace patterns
        if self.replace_patterns is not None:
            if len(self.replace_patterns) > 1:
                result_out = Parallel(n_jobs=-1, backend='threading')(delayed(apply_replace_patterns)(new_X, replace_dict) for replace_dict in self.replace_patterns)
                for result in result_out:
                    new_X.loc[:, result[0]] = result[1]
                # free up memory usage by joblib pool
                force_joblib_cleanup()
            else:
                if self.replace_patterns is not None and not (not self.replace_patterns):
                    new_X.loc[:, self.replace_patterns[0].get('rel_col')] = apply_replace_patterns(new_X, self.replace_patterns[0])[1]
        # parse date columns
        if date_cols is not None:
            for col in date_cols:
                if col in new_X.columns and not is_datetime64_any_dtype(new_X[col]):
                    new_X.loc[:, col] = pd.to_datetime(new_X[col], errors='coerce', utc=True)
        # parse int columns
        if int_cols is not None:
            for col in int_cols:
                if col in new_X.columns:
                    new_X.loc[:, col] = new_X[col].astype(int)
        # parse float columns
        if float_cols is not None:
            for col in float_cols:
                if col in new_X.columns:
                    new_X.loc[:, col] = new_X[col].astype(float)
        # generate any categorical column
        if gen_cat_col is not None:
            num_col = gen_cat_col.get('num_col')
            if num_col in new_X.columns:
                cat_col = gen_cat_col.get('cat_col')
                bins = gen_cat_col.get('bins')
                labels = gen_cat_col.get('labels')
                new_X.loc[:, cat_col] = pd.cut(new_X[num_col], bins=bins, labels=labels)
        # process any data clipping; could also use the generated categories above to apply clipping
        if clip_data:
            rel_cols = clip_data.get('rel_cols')
            filterby = clip_data.get('filterby')
            pos_thres = clip_data.get('pos_thres')
            new_X = apply_clipping(new_X, rel_cols=rel_cols, filterby=filterby, pos_thres=pos_thres)
        # process any special features
        def process_feature(col, feat_mappings, sep: str = ','):
            created_features = new_X[col].apply(lambda x: parse_special_features(x, feat_mappings, sep=sep))
            new_feat_values = {mapping: [] for mapping in feat_mappings.values()}
            for index, col in enumerate(feat_mappings.values()):
                for row in range(created_features.shape[0]):
                    new_feat_values.get(col).append(created_features.iloc[row][index])
                new_X.loc[:, col] = new_feat_values.get(col)

        if special_features is not None:
            rel_cols = special_features.keys()
            for col in rel_cols:
                # first apply any regex replacements to clean-up
                regex_pat = special_features.get(col).get('regex_pat')
                regex_repl = special_features.get(col).get('regex_repl')
                if regex_pat is not None:
                    new_X.loc[:, col] = new_X[col].str.replace(regex_pat, regex_repl, regex=True)
                # then process features mappings
                feat_mappings = special_features.get(col).get('feat_mappings')
                sep = special_features.get(col).get('sep')
                process_feature(col, feat_mappings, sep=sep if sep is not None else ',')
            if drop_feats_cols:
                to_drop = [col for col in rel_cols if col in new_X.columns]
                new_X.drop(columns=to_drop, inplace=True)
        # apply any masking logic
        if masked_cols is not None:
            for col, mask in masked_cols.items():
                if col not in new_X.columns:
                    continue
                new_X.loc[:, col] = np.where(new_X.agg(mask, axis=1), 1, 0)
        # generate any target variables as needed
        # do this safely so that if any missing features is encountered, as with real unseen data situation where
        # future variable is not available at the time of testing, then ignore the target generation as it ought
        # to be predicted
        new_X, new_y = self.__generate_target(new_X, new_y, gen_target=gen_target, include_target=include_target, )
        LOGGER.debug('DataPrepTransformer: Pre-processed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X, new_y

    def __post_process(self, X, correlated_cols: list = None, pot_leak_cols: list = None, ):
        """
        Post-processing that may be required.
        :param X: dataset
        :type X: pd.DataFrame
        :param correlated_cols: columns that are moderately to highly correlated and should be dropped
        :type correlated_cols: list
        :param pot_leak_cols: columns that could potentially introduce data leakage and should be dropped
        :type pot_leak_cols: list
        :return:
        :rtype:
        """
        LOGGER.debug('DataPrepTransformer: Post-processing dataset, out shape = {}', X.shape)
        new_X = X
        if correlated_cols is not None or not (not correlated_cols):
            to_drop = [col for col in correlated_cols if col in new_X.columns]
            new_X.drop(columns=to_drop, inplace=True)
        if pot_leak_cols is not None or not (not pot_leak_cols):
            to_drop = [col for col in pot_leak_cols if col in new_X.columns]
            new_X.drop(columns=to_drop, inplace=True)
        LOGGER.debug('DataPrepTransformer: Post-processed dataset, out shape = {}', new_X.shape)
        return new_X

    def __gen_lag_features(self, X, y=None):
        # generate any calculated lagging columns as needed
        trans_lag_features = None
        if self.lag_features is not None:
            indices = X.index
            lag_feats = {}
            for col, col_filter_by_dict in self.lag_features.items():
                rel_col = col_filter_by_dict.get('rel_col')
                filter_by_cols = col_filter_by_dict.get('filter_by')
                period = int(col_filter_by_dict.get('period'))
                freq = col_filter_by_dict.get('freq')
                drop_rel_cols = col_filter_by_dict.get('drop_rel_cols')
                if filter_by_cols is not None or not (not filter_by_cols):
                    lag_feat = X.sort_values(by=filter_by_cols).shift(period=period, freq=freq)[rel_col]
                else:
                    lag_feat = X.shift(period)[rel_col]
                if drop_rel_cols is not None or not (not drop_rel_cols):
                    if drop_rel_cols:
                        X.drop(columns=[rel_col], inplace=True)
                lag_feats[col] = lag_feat.values
            trans_lag_features = pd.DataFrame(lag_feats, index=indices)
        return trans_lag_features

    def __gen_synthetic_features(self, X, y=None, ):
        # generate any calculated columns as needed - the input features
        # include one or more synthetic features, not present in test data
        if self.synthetic_features is not None:
            new_X = X
            for col, col_gen_func_dict in self.synthetic_features.items():
                # each col_gen_func_dict specifies {'func': col_gen_func1, 'inc_target': False, 'kwargs': {}}
                # to include the target as a parameter to the col_gen_func, and any keyword arguments
                # generate feature function specification should include at least an id_by_col
                # but can also include sort_by_cols
                col_gen_func = col_gen_func_dict.get('func')
                func_kwargs: dict = col_gen_func_dict.get('kwargs')
                inc_target = col_gen_func_dict.get('inc_target')
                if col_gen_func is not None:
                    if inc_target is not None and inc_target:
                        if (func_kwargs is not None) or not (not func_kwargs):
                            new_X[:, col] = new_X.apply(col_gen_func, func_kwargs, target=self.target, axis=1, )
                        else:
                            new_X[:, col] = new_X.apply(col_gen_func, target=self.target, axis=1, )
                    else:
                        if (func_kwargs is not None) or not (not func_kwargs):
                            new_X[:, col] = new_X.apply(col_gen_func, func_kwargs, axis=1)
                        else:
                            new_X[:, col] = new_X.apply(col_gen_func, axis=1)

    def __transform_calc_features(self, X, y=None, calc_features: dict=None):
        # generate any calculated columns as needed - the input features
        # includes only features present in test data - i.e., non-synthetic features
        trans_calc_features = None
        if calc_features is not None:
            indices = X.index
            calc_feats = {}
            results_out = Parallel(n_jobs=-1, backend='threading')(delayed(apply_calc_feature)(X, col, col_gen_func_dict) for col, col_gen_func_dict in calc_features.items())
            for result in results_out:
                calc_feats[result[0]] = result[1]
                is_numeric = calc_features.get(result[0]).get('is_numeric')
                is_numeric = True if is_numeric is None else is_numeric
                impute_agg_func = calc_features.get(result[0]).get('impute_agg_func')
                if is_numeric:
                    self.transform_global_aggs[result[0]] = result[1].agg(impute_agg_func if impute_agg_func is not None else 'median')
                else:
                    self.transform_global_aggs[result[0]] = result[1].value_counts().index[0]
            trans_calc_features = pd.DataFrame(calc_feats, index=indices)
            # free up memory usage by joblib pool
            force_joblib_cleanup()
        return trans_calc_features

    def __merge_features(self, source: pd.DataFrame, features: pd.DataFrame, rel_col: str=None, left_on: list=None, right_on: list=None, synthetic: bool = False):
        assert source is not None, 'Source dataframe cannot be None'
        if features is not None:
            # check if existing columns need to be dropped from source
            cols_in_source = [col for col in features.columns if col in source.columns]
            if left_on is not None:
                for col in left_on:
                    cols_in_source.remove(col)
            if cols_in_source is not None and not (not cols_in_source):
                source.drop(columns=cols_in_source, inplace=True)
            # now merge and replace the new columns in source
            if (left_on is None) and (right_on is None):
                source = pd.merge(source, features, how='left', left_index=True, right_index=True)
            elif (left_on is not None) and (right_on is not None):
                source = pd.merge(source, features, how='left', left_on=left_on, right_on=right_on)
            elif left_on is not None:
                source = pd.merge(source, features, how='left', left_on=left_on, right_index=True)
            else:
                source = pd.merge(source, features, how='left', left_index=True, right_index=True)
            # impute as needed
            if synthetic:
                contains_nulls = source[rel_col].isnull().values.any()
                if contains_nulls:
                    if synthetic:
                        if rel_col is not None:
                            global_agg = self.gen_global_aggs[rel_col]
                            source[rel_col] = source[rel_col].fillna(global_agg)
                        else:
                            for col in cols_in_source:
                                global_agg = self.gen_global_aggs[col]
                                source[rel_col] = source[col].fillna(global_agg)
            else:
                for col in cols_in_source:
                    global_agg = self.transform_global_aggs[col]
                    source[col] = source[rel_col].fillna(global_agg)
        return source

    def __do_transform(self, X, y=None, **fit_params):
        # do any required pre-processing
        new_X, new_y = self.__pre_process(X, y, date_cols=self.date_cols, int_cols=self.int_cols, float_cols=self.float_cols,
                                   masked_cols=self.masked_cols, special_features=self.special_features,
                                   drop_feats_cols=self.drop_feats_cols, gen_target=self.gen_target,
                                   gen_cat_col=self.gen_cat_col,
                                   clip_data=self.clip_data, include_target=self.include_target,)
        # apply any basic calculated features
        calc_feats = self.__transform_calc_features(X, y=y, calc_features=self.basic_calc_features)
        new_X = self.__merge_features(new_X, calc_feats, )
        # then apply any delayed calculated features
        calc_feats = self.__transform_calc_features(new_X, y=y, calc_features=self.delayed_calc_features)
        new_X = self.__merge_features(new_X, calc_feats, )
        # apply any generated features
        for key, gen_features in self.gen_calc_features.items():
            gen_spec = self.synthetic_features.get(key)
            sort_by_cols = gen_spec.get('sort_by_cols')
            grp_by_candidates = [gen_spec.get('id_by_col')]
            if sort_by_cols is not None and not (not sort_by_cols):
                grp_by_candidates.extend(sort_by_cols)
            keys = [col for col in grp_by_candidates if col is not None]
            new_X = self.__merge_features(new_X, gen_features, key, left_on=keys, right_on=keys, synthetic=True)
        # then apply any post-processing
        new_X = self.__post_process(new_X, correlated_cols=self.correlated_cols, pot_leak_cols=self.pot_leak_cols,)
        return new_X, new_y

    def get_params(self, deep=True):
        return {
            'date_cols': self.date_cols,
            'int_cols': self.int_cols,
            'float_cols': self.float_cols,
            'masked_cols': self.masked_cols,
            'special_features': self.special_features,
            'drop_feats_cols': self.drop_feats_cols,
            'gen_target': self.gen_target,
            'calc_features': self.calc_features,
            'correlated_cols': self.correlated_cols,
            'gen_cat_col': self.gen_cat_col,
            'pot_leak_cols': self.pot_leak_cols,
            'clip_data': self.clip_data,
            'include_target': self.include_target,
        }

class FeatureGenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gen_cols: dict=None, target_col: str=None, **kwargs):
        """
        Generates features from existing columns in the dataframe based on specified functions.
        :param gen_cols: dictionary of new feature column labels and their corresponding value generation functions
        and alternative values (to fillna) - e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'feat_col': (val_gen_func, alter_val)}
        :type dict:
        :param target_col: the column label of the target column - either as a hint or may be encountered as part of any generation function.
        :param kwargs:
        :type kwargs:
        """
        assert gen_cols is not None and not (not gen_cols), 'A valid dictionary of new feature column labels and their corresponding value generation functions and optional default values expected as input'
        self.gen_cols = gen_cols
        self.target_col = target_col

    def fit(self, X, y=None):
        LOGGER.debug('FeatureGenTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return self

    def transform(self, X):
        LOGGER.debug('FeatureGenTransformer: Transforming dataset, shape = {}', X.shape)
        new_X = self.__do_transform(X)
        LOGGER.debug('FeatureGenTransformer: Transformed dataset, out shape = {}', new_X.shape)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('FeatureGenTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y)
        LOGGER.debug('FeatureGenTransformer: Fit-transformed dataset, out shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X = generate_features(X, y, gen_cols=self.gen_cols, target_col=self.target_col, **fit_params)
        return new_X

    def get_params(self, deep=True):
        return {
            'gen_cols': self.gen_cols,
            'target_col': self.target_col,
        }

class SelectiveFunctionTransformer(FunctionTransformer):
    def __init__(self, rel_cols: list, **kwargs):
        super().__init__(**kwargs)
        self.rel_cols = rel_cols

    def fit(self, X, y=None):
        LOGGER.debug('SelectiveFunctionTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        to_fit = safe_copy(X[self.rel_cols])
        super().fit(to_fit, y)
        return self

    def transform(self, X):
        LOGGER.debug('SelectiveFunctionTransformer: Transforming dataset, shape = {}', X.shape)
        new_X = self.__do_transform(X)
        LOGGER.debug('SelectiveFunctionTransformer: Transformed dataset, out shape = {}', new_X.shape)
        return new_X

    def fit_transform(self, X, y=None, **kwargs):
        LOGGER.debug('SelectiveFunctionTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, **kwargs)
        LOGGER.debug('SelectiveFunctionTransformer: Fit-transformed dataset, out shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **kwargs):
        new_X = safe_copy(X)
        for col in self.rel_cols:
            to_transform = safe_copy(X[col])
            fitted_X = super().transform(to_transform)
            if isinstance(fitted_X, np.ndarray):
                fitted_X = pd.DataFrame(fitted_X, columns=[col])
            new_X[col] = fitted_X[col].values if isinstance(fitted_X, pd.DataFrame) else fitted_X
        new_X.fillna(0, inplace=True)
        return new_X

    def __inverse_transform(self, X):
        new_X = safe_copy(X)
        for col in self.rel_cols:
            to_inverse = safe_copy(X[col])
            inversed_X = super().inverse_transform(to_inverse)
            if isinstance(inversed_X, np.ndarray):
                inversed_X = pd.DataFrame(inversed_X, columns=self.rel_cols)
            new_X[col] = inversed_X[col].values if isinstance(inversed_X, pd.DataFrame) else inversed_X
        return new_X

class GeospatialTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms latitude-longitude point to a geohashed fixed neighborhood.
    """
    def __init__(self, lat_col: str, long_col: str, to_col: str, drop_geo_cols: bool=True,
                 precision: int=6, smoothing: float=5.0, min_samples_leaf: int=10, **kwargs):
        """
        Transforms latitude-longitude point to a geohashed fixed neighborhood.
        :param lat_col: the column labels for desired latitude column
        :type lat_col: str
        :param long_col: the column labels for desired longitude column
        :type long_col: str
        :param to_col: the new generated column label for the geohashed fixed neighborhood
        :param drop_geo_cols: drops the latitude and longitude columns
        :param precision: geohash precision - default is 6
        :param smoothing: smoothing effect to balance categorical average vs prior - higher value means stronger regularization.
        :param min_samples_leaf: used for regularization the weighted average between category mean and global mean is taken
        :param kwargs:
        :type kwargs:
        """
        assert lat_col is not None and not (not lat_col), 'A valid column label is expected for latitude column'
        assert long_col is not None and not (not long_col), 'A valid column label is expected for longitude'
        assert to_col is not None and not (not to_col), 'A valid column label is expected for the generated geohashed fixed neighborhood'
        super().__init__(**kwargs)
        self.lat_col = lat_col
        self.long_col = long_col
        self.to_col = to_col
        self.drop_geo_cols = drop_geo_cols
        self.precision = precision
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.target_encoder: TargetEncoder
        self.fitted = False

    def fit(self, X, y=None):
        LOGGER.debug('GeospatialTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # fit the encoder on the training data
        self.__do_fit(X, y, )
        LOGGER.debug('GeospatialTransformer: Fitted dataset, out shape = {}, {}', X.shape, y.shape if y is not None else None)
        return self

    def transform(self, X, y=None):
        LOGGER.debug('GeospatialTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, )
        LOGGER.debug('GeospatialTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('GeospatialTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.__do_fit(X, y, )
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('GeospatialTransformer: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X = self.__generate_geohashes(X, **fit_params)
        if y is not None:
            new_X = self.target_encoder.fit_transform(new_X, y,)
        else:
            new_X = self.target_encoder.transform(new_X,)
        #feature_names = self.target_encoder.get_feature_names_out()
        #new_X = pd.DataFrame(new_X, columns=feature_names)
        if self.drop_geo_cols:
            new_X.drop(columns=[self.lat_col, self.long_col], inplace=True)
        return new_X

    def __generate_geohashes(self, X, **fit_params):
        new_X = safe_copy(X)
        # notes: precision of 5 translates to ≤ 4.89km × 4.89km; 6 translates to ≤ 1.22km × 0.61km; 7 translates to ≤ 153m × 153m
        new_X[self.to_col] = new_X.transform(lambda x: gh.encode(x[self.lat_col], x[self.long_col], precision=self.precision), axis=1)
        return new_X

    def __do_fit(self, X, y=None, **fit_params):
        if not self.fitted:
            new_X = self.__generate_geohashes(X, **fit_params)
            # generate expected values based on category aggregates
            self.target_encoder = TargetEncoder(cols=[self.to_col], return_df=True,
                                                smoothing=self.smoothing, min_samples_leaf=self.min_samples_leaf, )
            # fit the encoder
            new_y = safe_copy(y)
            self.target_encoder.fit(new_X, new_y)
            self.fitted = True
        return self

class CategoricalTargetEncoder(ColumnTransformer):
    def __init__(self, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        # if configuring more than one column transformer make sure verbose_feature_names_out=True
        # to ensure the prefixes ensure uniqueness in the feature names
        __data_handler: DataPrepProperties = AppProperties().get_subscriber('data_handler')
        conf_target_encs = __data_handler.get_target_encoder() # a list with a single tuple (with target encoder pipeline)
        super().__init__(transformers=conf_target_encs,
                         remainder=remainder, force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True if len(conf_target_encs) > 1 else False,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.num_transformers = len(conf_target_encs)
        self.feature_names = conf_target_encs[0][2]
        self.fitted = False

    def fit(self, X, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('CategoricalTargetEncoder: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        super().fit(X, y, **fit_params)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('CategoricalTargetEncoder: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('CategoricalTargetEncoder: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        #self.fit(X, y, **fit_params) # cannot make this call as a superclass call creates an unending loop
        LOGGER.debug('CategoricalTargetEncoder: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('CategoricalTargetEncoder: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if y is None or self.fitted:
            transformed_X = super().transform(X, **fit_params)
        else:
            transformed_X = super().fit_transform(X, y, **fit_params)
        feature_names_out = super().get_feature_names_out().tolist()
        if self.num_transformers > 1:
            feature_names_out.reverse()
            # sort out any potential duplicates - noting how column transformers concatenate transformed and
            # passthrough columns
            feature_names = [feature_name.split('__')[-1] for feature_name in feature_names_out]
            duplicate_feature_idxs = []
            desired_feature_names_s = set()
            desired_feature_names = []
            for idx, feature_name in enumerate(feature_names):
                if feature_name not in desired_feature_names_s:
                    desired_feature_names_s.add(feature_name)
                    desired_feature_names.append(feature_name)
                else:
                    duplicate_feature_idxs.append(idx)
            desired_feature_names.reverse()
            duplicate_feature_idxs = [len(feature_names) - 1 - idx for idx in duplicate_feature_idxs]
            transformed_X = np.delete(transformed_X, duplicate_feature_idxs, axis=1)
        else:
            desired_feature_names = feature_names_out
        new_X = pd.DataFrame(transformed_X, columns=desired_feature_names)
        # re-order columns, so the altered columns appear at the end
        for feature_name in self.feature_names:
            if feature_name in desired_feature_names:
                desired_feature_names.remove(feature_name)
            try:
                new_X[feature_name] = pd.to_numeric(new_X[feature_name], )
            except ValueError as ignore:
                LOGGER.warning('Potential dtype issue: {}', ignore)
        desired_feature_names.extend(self.feature_names)
        return new_X[desired_feature_names]

"""
Adapted from https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/transformers/feature_augmenter.html
"""
class TSFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
        Sklearn-compatible estimator, for calculating and adding many features calculated from a given time series
        to the data. It is basically a wrapper around :func:`~tsfresh.feature_extraction.extract_features`.

        The features include basic ones like min, max or median, and advanced features like fourier
        transformations or statistical tests. For a list of all possible features, see the module
        :mod:`~tsfresh.feature_extraction.feature_calculators`. The column name of each added feature contains the name
        of the function of that module, which was used for the calculation.

        For this estimator, two datasets play a crucial role:

        1. the time series container with the timeseries data. This container (for the format see :ref:`data-formats-label`)
           contains the data which is used for calculating the
           features. It must be groupable by ids which are used to identify which feature should be attached to which row
           in the second dataframe.

        2. the input data X, where the features will be added to. Its rows are identifies by the index and each index in
           X must be present as an id in the time series container.

        Imagine the following situation: You want to classify 10 different financial shares and you have their development
        in the last year as a time series. You would then start by creating features from the metainformation of the
        shares, e.g. how long they were on the market etc. and filling up a table - the features of one stock in one row.
        This is the input array X, which each row identified by e.g. the stock name as an index.

        >>> df = pandas.DataFrame(index=["AAA", "BBB", ...])
        >>> # Fill in the information of the stocks
        >>> df["started_since_days"] = ... # add a feature

        You can then extract all the features from the time development of the shares, by using this estimator.
        The time series container must include a column of ids, which are the same as the index of X.

        >>> time_series = read_in_timeseries() # get the development of the shares
        >>> from cheutils import TSFeatureAugmenter
        >>> augmenter = TSFeatureAugmenter(column_id="id")
        >>> augmenter.fit(time_series, y=None)
        >>> df_with_time_series_features = augmenter.transform(df)

        The settings for the feature calculation can be controlled with the settings object.
        If you pass ``None``, the default settings are used.
        Please refer to :class:`~tsfresh.feature_extraction.settings.ComprehensiveFCParameters` for
        more information.

        This estimator does not select the relevant features, but calculates and adds all of them to the DataFrame. See the
        :class:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter` for calculating and selecting
        features.

        For a description what the parameters column_id, column_sort, column_kind and column_value mean, please see
        :mod:`~tsfresh.feature_extraction.extraction`.
        """

    def __init__(self, default_fc_parameters=None, kind_to_fc_parameters=None, column_id=None,
                 column_sort=None, column_kind=None, column_value=None,
                 chunksize=tsfresh.defaults.CHUNKSIZE, n_jobs=tsfresh.defaults.N_PROCESSES,
                 show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                 disable_progressbar=tsfresh.defaults.DISABLE_PROGRESSBAR,
                 impute_function=tsfresh.defaults.IMPUTE_FUNCTION, profile=tsfresh.defaults.PROFILING,
                 profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                 profiling_sorting=tsfresh.defaults.PROFILING_SORTING, drop_rel_cols: dict=None,
                 include_target:bool=False, ):
        """
        Create a new TSFeatureAugmenter instance.
        :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
               which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
               more information.
        :type default_fc_parameters: dict

        :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
                default_fc_parameters. If you put a kind as a key here, the fc_parameters
                object (which is the value), will be used instead of the default_fc_parameters. This means that kinds,
                for which kind_of_fc_parameters doe not have any entries, will be ignored by the feature selection.
        :type kind_to_fc_parameters: dict
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring
        :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
        :type n_jobs: int
        :param chunksize: The size of one chunk that is submitted to the worker
            process for the parallelisation.  Where one chunk is defined as a
            singular time series for one id and one kind. If you set the chunksize
            to 10, then it means that one task is to calculate all features for 10
            time series.  If it is set it to None, depending on distributor,
            heuristics are used to find the optimal chunksize. If you get out of
            memory exceptions, you can try it with the dask distributor and a
            smaller chunksize.
        :type chunksize: None or int
        :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
        :type show_warnings: bool
        :param disable_progressbar: Do not show a progressbar while doing the calculation.
        :type disable_progressbar: bool
        :param impute_function: None, if no imputing should happen or the function to call for imputing
            the result dataframe. Imputing will never happen on the input data.
        :type impute_function: None or function
        :param profile: Turn on profiling during feature extraction
        :type profile: bool
        :param profiling_sorting: How to sort the profiling results (see the documentation of the profiling package for
               more information)
        :type profiling_sorting: basestring
        :param profiling_filename: Where to save the profiling results.
        :type profiling_filename: basestring
        :param drop_rel_cols: flags indicating whether to drop the time series source features
        :type drop_rel_cols: dict
        """
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting
        self.extracted_features = None # holder for extracted features
        self.extracted_global_aggs = {}
        self.drop_rel_cols = drop_rel_cols
        self.include_target = include_target

    def fit(self, X=None, y=None):
        LOGGER.debug('TSFeatureAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        timeseries_container = safe_copy(X)
        if timeseries_container is None:
            raise RuntimeError('You have to provide a time series using the set_timeseries_container function before.')
        if self.include_target:
            timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
        # extract the features
        self.extracted_features = extract_features(timeseries_container,
                                                  default_fc_parameters=self.default_fc_parameters,
                                                  kind_to_fc_parameters=self.kind_to_fc_parameters,
                                                  column_id=self.column_id,
                                                  column_sort=self.column_sort,
                                                  column_kind=self.column_kind,
                                                  column_value=self.column_value,
                                                  chunksize=self.chunksize,
                                                  n_jobs=self.n_jobs,
                                                  show_warnings=self.show_warnings,
                                                  disable_progressbar=self.disable_progressbar,
                                                  impute_function=self.impute_function,
                                                  profile=self.profile,
                                                  profiling_filename=self.profiling_filename,
                                                  profiling_sorting=self.profiling_sorting, )
        del timeseries_container
        self.extracted_features.index.rename(self.column_id, inplace=True)
        cols = list(self.extracted_features.columns)
        col_map = {col: col.replace('__', '_') for col in cols}
        self.extracted_features.rename(columns=col_map, inplace=True)
        self.extracted_features = self.extracted_features.bfill()
        self.extracted_features.fillna(value=0, inplace=True)
        for col in list(self.extracted_features.columns):
            self.extracted_global_aggs[col] = self.extracted_features.reset_index()[col].agg('median')
        return self

    def transform(self, X, **fit_params):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        To save some computing time, you should only include those time serieses in the container, that you
        need.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSFeatureAugmenter: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('TSFeatureAugmenter: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('TSFeatureAugmenter: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if self.extracted_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        new_X = pd.merge(X, self.extracted_features, left_on=self.column_id, right_index=True, how='left')
        feat_cols = list(self.extracted_features.columns)
        for col in feat_cols:
            new_X[col] = new_X[col].fillna(self.extracted_global_aggs.get(col))
        if self.drop_rel_cols is not None and not (not self.drop_rel_cols):
            to_drop_cols = []
            for key, item in self.drop_rel_cols.items():
                if item is not None and item:
                    to_drop_cols.append(key)
            if to_drop_cols is not None and not (not to_drop_cols):
                new_X.drop(columns=to_drop_cols, inplace=True)
        return new_X

class TSLagFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
        Sklearn-compatible estimator, for calculating and adding many features calculated from a given time series
        to the data. It is basically a wrapper around :func:`~tsfresh.feature_extraction.extract_features`.

        The features include basic ones like min, max or median, and advanced features like fourier
        transformations or statistical tests. For a list of all possible features, see the module
        :mod:`~tsfresh.feature_extraction.feature_calculators`. The column name of each added feature contains the name
        of the function of that module, which was used for the calculation.

        For this estimator, two datasets play a crucial role:

        1. the time series container with the timeseries data. This container (for the format see :ref:`data-formats-label`)
           contains the data which is used for calculating the
           features. It must be groupable by ids which are used to identify which feature should be attached to which row
           in the second dataframe.

        2. the input data X, where the features will be added to. Its rows are identifies by the index and each index in
           X must be present as an id in the time series container.

        Imagine the following situation: You want to classify 10 different financial shares and you have their development
        in the last year as a time series. You would then start by creating features from the metainformation of the
        shares, e.g. how long they were on the market etc. and filling up a table - the features of one stock in one row.
        This is the input array X, which each row identified by e.g. the stock name as an index.

        >>> df = pandas.DataFrame(index=["AAA", "BBB", ...])
        >>> # Fill in the information of the stocks
        >>> df["started_since_days"] = ... # add a feature

        You can then extract all the features from the time development of the shares, by using this estimator.
        The time series container must include a column of ids, which are the same as the index of X.

        >>> time_series = read_in_timeseries() # get the development of the shares
        >>> from cheutils import TSFeatureAugmenter
        >>> augmenter = TSLagFeatureAugmenter(column_id="id")
        >>> augmenter.fit(time_series, y=None)
        >>> df_with_time_series_features = augmenter.transform(df)

        The settings for the feature calculation can be controlled with the settings object.
        If you pass ``None``, the default settings are used.
        Please refer to :class:`~tsfresh.feature_extraction.settings.ComprehensiveFCParameters` for
        more information.

        This estimator does not select the relevant features, but calculates and adds all of them to the DataFrame. See the
        :class:`~tsfresh.transformers.relevant_feature_augmenter.RelevantFeatureAugmenter` for calculating and selecting
        features.

        For a description what the parameters column_id, column_sort, column_kind and column_value mean, please see
        :mod:`~tsfresh.feature_extraction.extraction`.
        """

    def __init__(self, lag_features: dict, default_fc_parameters=None, kind_to_fc_parameters=None, column_id=None,
                 column_sort=None, column_kind=None, column_value=None, column_ts_date: str=None,
                 chunksize=tsfresh.defaults.CHUNKSIZE, n_jobs=tsfresh.defaults.N_PROCESSES,
                 show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                 disable_progressbar=tsfresh.defaults.DISABLE_PROGRESSBAR,
                 impute_function=tsfresh.defaults.IMPUTE_FUNCTION, profile=tsfresh.defaults.PROFILING,
                 profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                 profiling_sorting=tsfresh.defaults.PROFILING_SORTING,
                 drop_rel_cols: dict=None, lag_target: bool=False, ):
        """
        Create a new TSLagFeatureAugmenter instance.
        :param lag_features: dictionary of calculated column labels to hold lagging calculated values with their corresponding column lagging calculation functions - e.g., {'sort_by_cols': ['sort_by_col1', 'sort_by_col2'], period=1, 'freq': 'D', 'drop_rel_cols': False, }
        :type lag_features: dict

        :param default_fc_parameters: mapping from feature calculator names to parameters. Only those names
               which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
               more information.
        :type default_fc_parameters: dict

        :param kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
                default_fc_parameters. If you put a kind as a key here, the fc_parameters
                object (which is the value), will be used instead of the default_fc_parameters. This means that kinds,
                for which kind_of_fc_parameters doe not have any entries, will be ignored by the feature selection.
        :type kind_to_fc_parameters: dict
        :param column_id: The column with the id. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring
        :param column_ts_date: The column with the time series date feature relevant for sorting; if not specified assumed to be the same as column_sort
        :type column_ts_date: basestring
        :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
        :type n_jobs: int
        :param chunksize: The size of one chunk that is submitted to the worker
            process for the parallelisation.  Where one chunk is defined as a
            singular time series for one id and one kind. If you set the chunksize
            to 10, then it means that one task is to calculate all features for 10
            time series.  If it is set it to None, depending on distributor,
            heuristics are used to find the optimal chunksize. If you get out of
            memory exceptions, you can try it with the dask distributor and a
            smaller chunksize.
        :type chunksize: None or int
        :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
        :type show_warnings: bool
        :param disable_progressbar: Do not show a progressbar while doing the calculation.
        :type disable_progressbar: bool
        :param impute_function: None, if no imputing should happen or the function to call for imputing
            the result dataframe. Imputing will never happen on the input data.
        :type impute_function: None or function
        :param profile: Turn on profiling during feature extraction
        :type profile: bool
        :param profiling_sorting: How to sort the profiling results (see the documentation of the profiling package for
               more information)
        :type profiling_sorting: basestring
        :param profiling_filename: Where to save the profiling results.
        :type profiling_filename: basestring
        :param drop_rel_cols: flags to inidcate whether to drop the time series feature columns
        :type drop_rel_cols: dict
        :param lag_target: flag indicating whether to include lagged target values as features
        """
        assert lag_features is not None and not (not lag_features), 'Lag features specification must be provided'
        self.lag_features = lag_features
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value
        self.column_ts_date = column_ts_date if column_ts_date is not None else column_sort
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting
        self.drop_rel_cols = drop_rel_cols
        self.extracted_features = None # holder for extracted features
        self.extracted_global_aggs = {}
        self.lag_target = lag_target
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('TSLagFeatureAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        timeseries_container: pd.DataFrame = safe_copy(X)
        if self.lag_target:
            timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
        periods = self.lag_features.get('periods')
        freq = self.lag_features.get('freq')
        sort_by_cols = self.lag_features.get('sort_by_cols')
        timeseries_container.sort_values(by=sort_by_cols, inplace=True)
        timeseries_container.set_index(self.column_ts_date, inplace=True)
        timeseries_container = timeseries_container.shift(periods=periods + 1, freq=freq)
        if timeseries_container is None:
            raise RuntimeError('You have to provide a time series container/dataframe before.')
        # extract the features
        self.extracted_features = extract_features(timeseries_container,
                                                  default_fc_parameters=self.default_fc_parameters,
                                                  kind_to_fc_parameters=self.kind_to_fc_parameters,
                                                  column_id=self.column_id,
                                                  column_sort=self.column_sort,
                                                  column_kind=self.column_kind,
                                                  column_value=self.column_value,
                                                  chunksize=self.chunksize,
                                                  n_jobs=self.n_jobs,
                                                  show_warnings=self.show_warnings,
                                                  disable_progressbar=self.disable_progressbar,
                                                  impute_function=self.impute_function,
                                                  profile=self.profile,
                                                  profiling_filename=self.profiling_filename,
                                                  profiling_sorting=self.profiling_sorting, )
        self.extracted_features = self.extracted_features.bfill()
        self.extracted_features.fillna(0, inplace=True)
        self.extracted_features.index.rename(self.column_id, inplace=True)
        cols = list(self.extracted_features.columns)
        col_map = {col: col.replace('__', '_lag_') for col in cols}
        self.extracted_features.rename(columns=col_map, inplace=True)
        for col in list(self.extracted_features.columns):
            self.extracted_global_aggs[col] = self.extracted_features.reset_index()[col].agg('median')
        del timeseries_container
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        To save some computing time, you should only include those time serieses in the container, that you
        need. You can set the timeseries container with the method :func:`set_timeseries_container`.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSLagFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSLagFeatureAugmenter: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('TSLagFeatureAugmenter: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('TSLagFeatureAugmenter: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if self.extracted_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        new_X = pd.merge(X, self.extracted_features, left_on=self.column_id, right_index=True, how='left')
        feat_cols = list(self.extracted_features.columns)
        for col in feat_cols:
            new_X[col] = new_X[col].fillna(self.extracted_global_aggs.get(col))
        if self.drop_rel_cols is not None and not (not self.drop_rel_cols):
            to_drop_cols = []
            for key, item in self.drop_rel_cols.items():
                if item is not None and item:
                    to_drop_cols.append(key)
            if to_drop_cols is not None and not (not to_drop_cols):
                new_X.drop(columns=to_drop_cols, inplace=True)
        return new_X

class TSRollingLagFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    See also TSLagFeatureAugmenter.
    """
    def __init__(self, roll_cols: list, roll_target: bool=False, window: int=15,
                 shift_periods: int=15, freq: str='D', column_ts_date: str=None,
                 filter_by:str='date', agg_func: str='mean', ):
        """
        Create a new TSRollingLagFeatureAugmenter instance.
        :param roll_cols: The columns to aggregate - if not specified it is assumed that the target variable is rolled
        :type roll_cols: list
        :param roll_target: If True, the target variable is also included in the rolling window calculations
        :type roll_target: bool
        :param window: the rolling window
        :type window: int
        :param shift_periods: any lag periods as necessary
        :type shift_periods: int
        :param freq: the frequency of the rolling window
        :type freq: str
        :param column_ts_date: The column with the time series date feature relevant for sorting; if not specified assumed to be the same as column_sort
        :type column_ts_date: basestring
        :param filter_by: filter the data by this column label
        :type filter_by: basestring
        :param agg_func: The aggregation function to apply to each column
        :type agg_func: basestring
        """
        assert roll_cols is not None, 'Valid roll columns or features must be specified'
        self.roll_cols = roll_cols
        self.roll_target = roll_target
        self.window = window
        self.shift_periods = shift_periods
        self.freq = freq
        self.column_ts_date = column_ts_date
        self.filter_by = filter_by
        self.agg_func = agg_func
        self.extracted_features = None # holder for extracted features
        self.extracted_global_aggs = {}
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('TSRollingLagFeatureAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        all_cols_to_roll = self.roll_cols.copy()
        timeseries_container: pd.DataFrame = safe_copy(X)
        if self.roll_target:
            # roll target variable/feature
            timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
            all_cols_to_roll.append(y.name)
        timeseries_container.set_index(self.column_ts_date, inplace=True)
        # extract the features
        self.extracted_features = timeseries_container.groupby(self.filter_by)[all_cols_to_roll].apply(lambda x: x.shift(self.shift_periods, freq=self.freq)).rolling(window=self.window + 1, min_periods=1).agg(self.agg_func)
        self.extracted_features = self.extracted_features.bfill()
        self.extracted_features.fillna(0, inplace=True)
        self.extracted_features = self.extracted_features.reset_index()
        col_map = {roll_col: roll_col.replace(roll_col, roll_col + '_' + str(self.shift_periods) + '_lag_rolling_' + self.agg_func) for roll_col in all_cols_to_roll}
        self.extracted_features.rename(columns=col_map, inplace=True)
        cols = list(self.extracted_features.columns)
        self.extracted_features.loc[:, cols[-len(all_cols_to_roll):]] = self.extracted_features[cols[-len(all_cols_to_roll):]].bfill()
        for col in cols[-len(all_cols_to_roll):]:
            self.extracted_global_aggs[col] = self.extracted_features[col].agg(self.agg_func)
        del timeseries_container
        """is_duplicate = self.extracted_features.duplicated(subset=[self.filter_by, self.column_ts_date], keep='last')
        try:
            self.extracted_features[is_duplicate].to_excel('duplicate_extracted_features.xlsx')
            self.extracted_features[~is_duplicate].to_excel('not_duplicate_extracted_features.xlsx')
        except Exception as ignore:
            LOGGER.warning('Could not save duplicate extracted features:\n{}', ignore)"""
        self.extracted_features.drop_duplicates(subset=[self.filter_by, self.column_ts_date], keep='last', inplace=True)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        To save some computing time, you should only include those time serieses in the container, that you
        need. You can set the timeseries container with the method :func:`set_timeseries_container`.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSRollingLagFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSRollingLagFeatureAugmenter: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('TSRollingLagFeatureAugmenter: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('TSRollingLagFeatureAugmenter: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if self.extracted_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        common_key = [self.filter_by, self.column_ts_date]
        new_X = pd.merge(X, self.extracted_features, how='left', left_on=common_key, right_on=common_key)
        feat_cols = list(self.extracted_features.columns)
        for feat_col in feat_cols:
            if feat_col not in common_key:
                new_X.loc[:, feat_col] = new_X[feat_col].fillna(self.extracted_global_aggs.get(feat_col))
        return new_X

class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, group_by: list,
                 l_quartile: float = 0.25, u_quartile: float = 0.75, pos_thres: bool=False, ):
        """
        Create a new TSRollingLagFeatureAugmenter instance.
        :param rel_cols: the list of columns to clip
        :param group_by: list of columns to group by or filter the data by
        :param l_quartile: the lower quartile (float between 0 and 1)
        :param u_quartile: the upper quartile (float between 0 and 1 but greater than l_quartile)
        :param pos_thres: enforce positive clipping boundaries or thresholds values
        """
        assert rel_cols is not None, 'Valid numeric feature columns must be specified'
        assert group_by is not None or not (not group_by), 'Valid numeric feature columns must be specified'
        self.rel_cols = rel_cols
        self.group_by = group_by
        self.l_quartile = l_quartile
        self.u_quartile = u_quartile
        self.pos_thres = pos_thres
        self.extracted_cat_thres = None # holder for extracted category thresholds
        self.extracted_global_thres = {}
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('ClipOutliersTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # generate category thresholds
        self.extracted_cat_thres = get_outlier_cat_thresholds(X, self.rel_cols, self.group_by,
                                                              self.l_quartile, self.u_quartile, self.pos_thres)
        for rel_col in self.rel_cols:
            col_iqr = iqr(X[rel_col])
            qvals = get_quantiles(X, rel_col, [self.l_quartile, self.u_quartile])
            l_thres = qvals[0] - 1.5 * col_iqr
            u_thres = qvals[1] + 1.5 * col_iqr
            l_thres = max(0, l_thres) if self.pos_thres else l_thres
            u_thres = max(0, u_thres) if self.pos_thres else u_thres
            self.extracted_global_thres[rel_col] = (l_thres, u_thres)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('ClipOutliersTransformer: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('ClipOutliersTransformer: Transformed dataset, shape = {}, {}', X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('ClipOutliersTransformer: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('ClipOutliersTransformer: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if not self.fitted or self.extracted_cat_thres is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # apply clipping appropriately
        new_X = safe_copy(X)
        cat_grps = new_X.groupby(self.group_by)
        clipped_subset = []
        for grp_name, cat_grp in cat_grps:
            cur_thres = self.extracted_cat_thres.get(grp_name)
            if cur_thres is not None:
                lower_thres, upper_thres = cur_thres
            else:
                l_thres = []
                u_thres = []
                for col in self.rel_cols:
                    l_thres.append(self.extracted_global_thres.get(col)[0])
                    u_thres.append(self.extracted_global_thres.get(col)[1])
                lower_thres, upper_thres = l_thres, u_thres
            clipped_subset.append(cat_grp[self.rel_cols].clip(lower=lower_thres, upper=upper_thres))
        clipped_srs = pd.concat(clipped_subset, ignore_index=False)
        new_X.loc[:, self.rel_cols] = clipped_srs
        return new_X