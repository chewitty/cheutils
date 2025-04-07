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
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties, AppPropertiesHandler
from cheutils.exceptions import PropertiesException, FeatureGenException
from cheutils.data_prep_support import apply_replace_patterns, apply_calc_feature, force_joblib_cleanup
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_categorical_dtype
from scipy.stats import iqr
from typing import cast

LOGGER = LoguruWrapper().get_logger()

@singleton
class DataPropertiesHandler(AppPropertiesHandler):
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

    def _load_feat_sel_passthrough(self):
        key = 'model.feat_selection.passthrough'
        self.__data_prep_properties['feat_sel_passthrough'] = self.__app_props.get_bol(key)

    def _load_feat_sel_override(self):
        key = 'model.feat_selection.override'
        self.__data_prep_properties['feat_sel_override'] = self.__app_props.get_bol(key)

    def _load_feat_sel_selected(self):
        key = 'model.feat_selection.selected'
        self.__data_prep_properties['feat_sel_selected'] = self.__app_props.get_list(key)

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

    def get_feat_sel_passthrough(self):
        value = self.__data_prep_properties.get('feat_sel_passthrough')
        if value is None:
            self._load_feat_sel_passthrough()
        return self.__data_prep_properties.get('feat_sel_passthrough')

    def get_feat_sel_override(self):
        value = self.__data_prep_properties.get('feat_sel_override')
        if value is None:
            self._load_feat_sel_override()
        return self.__data_prep_properties.get('feat_sel_override')

    def get_feat_sel_selected(self):
        value = self.__data_prep_properties.get('feat_sel_selected')
        if value is None:
            self._load_feat_sel_selected()
        return self.__data_prep_properties.get('feat_sel_selected')

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