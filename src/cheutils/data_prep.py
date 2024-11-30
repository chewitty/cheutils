import importlib
import pandas as pd
import numpy as np
import geolib.geohash as gh
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from cheutils.common_utils import apply_clipping, parse_special_features, safe_copy
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties

LOGGER = LoguruWrapper().get_logger()
APP_PROPS = AppProperties()
CONFIG_TRANSFORMERS = APP_PROPS.get_list_properties('model.selective_column.transformers')
if (CONFIG_TRANSFORMERS is not None) or not (not CONFIG_TRANSFORMERS):
    LOGGER.debug('Preparing configured column transformers: \n{}', CONFIG_TRANSFORMERS)
    SELECTIVE_TRANSFORMERS = []
    for item in CONFIG_TRANSFORMERS:
        name = item.get('name')
        tf_params = item.get('transformer_params')
        tf_params = {} if tf_params is None else tf_params
        cols = list(item.get('columns'))
        tf_class = getattr(importlib.import_module(item.get('transformer_package')),
                           item.get('transformer_name'))
        try:
            tf = tf_class(**tf_params)
            SELECTIVE_TRANSFORMERS.append((name, tf, cols))
        except TypeError as err:
            LOGGER.error('Problem encountered instantiating transformer: {}, {}', name, err)

class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms datetimes, generating additional prefixed 'dow', 'wk', 'qtr', 'wkend' features for all relevant columns
    (specified) in the dataframe; drops the datetime column by default but can be retained as desired.
    """
    def __init__(self, rel_cols: list, prefixes: list, drop_rel_cols: list=None, **kwargs):
        """
        Transforms datetimes, generating additional prefixed 'dow', 'wk', 'qtr', 'wkend' features for all relevant
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
            new_X[rel_col] = pd.to_datetime(new_X[rel_col], errors='coerce',
                                            utc=True)  # to be absolutely sure it is datetime
            new_X.loc[:, prefix + 'dow'] = new_X[rel_col].dt.dayofweek
            null_dayofweek = new_X[prefix + 'dow'].isna()
            nulldofwk = new_X[null_dayofweek]
            new_X[prefix + 'dow'] = new_X[prefix + 'dow'].astype(int)
            new_X.loc[:, prefix + 'wk'] = new_X[rel_col].apply(lambda x: pd.Timestamp(x).week)
            new_X[prefix + 'wk'] = new_X[prefix + 'wk'].astype(int)
            # new_X.loc[:, prefix + 'doy'] = new_X[rel_col].dt.dayofyear
            # new_X[prefix + 'doy'] = new_X[prefix + 'doy'].astype(int)
            new_X.loc[:, prefix + 'qtr'] = new_X[rel_col].dt.quarter
            new_X[prefix + 'qtr'] = new_X[prefix + 'qtr'].astype(int)
            new_X.loc[:, prefix + 'wkend'] = np.where(new_X[rel_col].dt.dayofweek.isin([5, 6]), 1, 0)
            new_X[prefix + 'wkend'] = new_X[prefix + 'wkend'].astype(int)
        if len(self.rel_cols) > 0:
            if self.drop_rel_cols is None or not self.drop_rel_cols:
                new_X.drop(columns=self.rel_cols, inplace=True)
            else:
                to_drop_cols = []
                for index, to_drop_cols in enumerate(self.rel_cols):
                    if self.drop_rel_cols[index]:
                        to_drop_cols.append(to_drop_cols)
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

"""
Meta-transformer for selecting features based on recursive feature selection.
"""
class FeatureSelectionTransformer(RFE):
    """
    Returns features based on ranking with recursive feature elimination.
    """
    def __init__(self, estimator=None, random_state: int=100, **kwargs):
        self.random_state = random_state
        self.estimator = estimator
        super().__init__(self.estimator, ** kwargs)
        self.target = None
        self.selected_cols = None

    def fit(self, X, y=None, **fit_params):
        LOGGER.debug('FeatureSelectionTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y  # possibly passed in chain
        #self.estimator.fit(X, y)
        #LOGGER.debug('FeatureSelectionTransformer: Feature coefficients = {}', self.estimator.coef_)
        return super().fit(X, y, **fit_params)

    def transform(self, X, y=None, **fit_params):
        LOGGER.debug('FeatureSelectionTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y=None)
        LOGGER.debug('FeatureSelectionTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        LOGGER.debug('FeatureSelectionTransformer: Transformed features selected = {}', self.selected_cols)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('FeatureSelectionTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('FeatureSelectionTransformer: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        LOGGER.debug('FeatureSelectionTransformer: Fit-transformed features selected = {}', self.selected_cols)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if y is None:
            transformed_X = super().transform(X)
        else:
            transformed_X = super().fit_transform(X, y, **fit_params)
        self.selected_cols = list(X.columns[self.get_support()])
        new_X = pd.DataFrame(transformed_X, columns=self.selected_cols)
        return new_X

    def get_selected_features(self):
        """
        Return the selected features or column labels.
        :return:
        """
        return self.selected_cols

    def get_target(self):
        return self.target

class DropSelectedColsTransformer(BaseEstimator, TransformerMixin):
    """
    Drops selected columns from the dataframe.
    """
    def __init__(self, rel_cols: list, **kwargs):
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.target = None

    def fit(self, X, y=None):
        LOGGER.debug('DropSelectedColsTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        return self

    def transform(self, X, y=None):
        LOGGER.debug('DropSelectedColsTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        new_X = self.__do_transform(X, y)
        LOGGER.debug('DropSelectedColsTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        LOGGER.debug('DropSelectedColsTransformer: Columns dropped = {}', self.rel_cols)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('DropSelectedColsTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        new_X = self.__do_transform(X, y, **fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        def drop_selected(df: pd.DataFrame, rel_cols: list):
            """
            Drop rows with missing data
            :param df: dataframe with the specified columns, which may not contain any target class labels
            :param rel_cols: list of column labels corresponding to columns of the specified dataframe
            :return: revised dataframe with the specified columns dropped
            """
            assert df is not None, 'A valid DataFrame expected as input'
            clean_df = df.copy(deep=True)
            clean_df = clean_df.drop(columns=rel_cols)
            LOGGER.debug('Dropped columns = {}', rel_cols)
            return clean_df
        new_X = drop_selected(X, rel_cols=self.rel_cols)
        return new_X

    def get_target(self):
        """
        Returns the transformed target if any
        :return:
        """
        return self.target

class SelectiveColumnTransformer(ColumnTransformer):
    def __init__(self, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose_feature_names_out=False, verbose=False, n_jobs=None, **kwargs):
        super().__init__(transformers=SELECTIVE_TRANSFORMERS, remainder=remainder,
                         force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=verbose_feature_names_out,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.feature_names = None

    def fit(self, X, y=None, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.feature_names = list(X.columns)
        super().fit(X, y, **fit_params)
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('SelectiveColumnTransformer: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.feature_names = list(X.columns)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('SelectiveColumnTransformer: Fit-transformed dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if y is None:
            transformed_X = super().transform(X, **fit_params)
        else:
            transformed_X = super().fit_transform(X, y, **fit_params)
        new_X = pd.DataFrame(transformed_X[:, :-1], columns=self.feature_names)
        return new_X

"""
The imblearn.FunctionSampler and imblearn.pipeline.Pipeline need to be used in order to correctly add this to a data pipeline
"""
def pre_process(X, y=None, date_cols: list=None, int_cols: list=None, float_cols: list=None,
                masked_cols: dict=None, special_features: dict=None, drop_feats_cols: bool=True,
                calc_features: dict=None, gen_target: dict=None, correlated_cols: list=None,
                pot_leak_cols: list=None, drop_missing: bool=False, clip_data: dict=None,
                include_target: bool=False,):
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
    :param calc_features: dictionary of calculated column labels with their corresponding column generation functions - e.g., {'col_label1': col_gen_func1, 'col_label2': col_gen_func2}
    :type calc_features: dict
    :param gen_target: dictionary of target column label and target generation function (e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'target_col': 'target_collabel', 'target_gen_func': target_gen_func}
    :type gen_target: dict
    :param correlated_cols: columns that are moderately to highly correlated and should be dropped
    :type correlated_cols: list
    :param pot_leak_cols: columns that could potentially introduce data leakage and should be dropped
    :type pot_leak_cols: list
    :param drop_missing: drop rows with missing data if True; default is False
    :type drop_missing: bool
    :param clip_data: clip the data based on categories defined by the filterby key and whether to enforec positive threshold defined by the pos_thres key - e.g., clip_data = {'rel_cols': ['col1', 'col2'], 'filterby': 'col_label1', 'pos_thres': False}
    :type clip_data: dict
    :param include_target: include the target Series in the returned first item of the tuple if True; default is False
    :return: Processed dataframe and updated target Series
    :rtype: tuple(pd.DataFrame, pd.Series or None)
    """
    LOGGER.debug('Preprocessing dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
    new_X = safe_copy(X)
    new_y = safe_copy(y)
    if drop_missing:
        def drop_missing(df: pd.DataFrame, target_sr: pd.Series = None):
            """
            Drop rows with missing data
            :param df: dataframe with the specified columns, which may not contain any target class labels
            :param target_sr: optional target class labels corresponding to the dataframe
            :return: revised dataframe and corresponding target series where present
            """
            assert df is not None, 'A valid DataFrame expected as input'
            clean_df = safe_copy(df)
            clean_target_sr = safe_copy(target_sr)
            null_rows = clean_df.isna().any(axis=1)
            clean_df = clean_df.dropna()
            # do not reset index here
            # clean_df.reset_index(drop=True, inplace=True)
            LOGGER.debug('Preprocessing, rows with missing data = {}', len(df) - len(clean_df))
            if target_sr is not None:
                clean_target_sr = clean_target_sr[~null_rows]
                # do not reset index here
                # clean_target_sr.reset_index(drop=True)
            return clean_df, clean_target_sr
        new_X, new_y = drop_missing(X, target_sr=new_y)
    if date_cols is not None:
        for col in date_cols:
            if col in new_X.columns:
                new_X[col] = pd.to_datetime(new_X[col], errors='coerce', utc=True)
    if int_cols is not None:
        for col in int_cols:
            if col in new_X.columns:
                new_X[col] = new_X[col].astype(int)
    if float_cols is not None:
        for col in float_cols:
            if col in new_X.columns:
                new_X[col] = new_X[col].astype(float)
    # process any data clipping
    if clip_data:
        rel_cols = clip_data.get('rel_cols')
        filterby = clip_data.get('filterby')
        pos_thres = clip_data.get('pos_thres')
        new_X = apply_clipping(new_X, rel_cols=rel_cols, filterby=filterby, pos_thres=pos_thres)
    # process any special features
    def process_feature(col, feat_mappings, sep:str=','):
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
                new_X[col] = new_X[col].str.replace(regex_pat, regex_repl, regex=True)
            # then process features mappings
            feat_mappings = special_features.get(col).get('feat_mappings')
            sep = special_features.get(col).get('sep')
            process_feature(col, feat_mappings, sep=sep if sep is not None else ',')
        if drop_feats_cols:
            to_drop = [col for col in rel_cols if col in new_X.columns]
            new_X.drop(columns=to_drop, inplace=True)
    # generate any calculated columns as needed
    if calc_features is not None:
        for col, col_gen_func in calc_features.items():
            new_X[col] = new_X.apply(col_gen_func, axis=1)
    # apply any masking logic
    if masked_cols is not None:
        for col, mask in masked_cols.items():
            new_X[col] = np.where(new_X.apply(mask, axis=1), 1, 0)
    # generate any target variables as needed
    # do this safely so that if any missing features is encountered, as with real unseen data situation where
    # future variable is not available at the time of testing, then ignore the target generation as it ought
    # to be predicted
    new_X, new_y = generate_target(new_X, new_y, gen_target=gen_target, include_target=include_target, )
    if correlated_cols is not None or not (not correlated_cols):
        to_drop = [col for col in correlated_cols if col in new_X.columns]
        new_X.drop(columns=to_drop, inplace=True)
    if pot_leak_cols is not None or not (not pot_leak_cols):
        to_drop = [col for col in pot_leak_cols if col in new_X.columns]
        new_X.drop(columns=to_drop, inplace=True)
    LOGGER.debug('Preprocessed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
    return new_X, new_y

def generate_target(X: pd.DataFrame, y: pd.Series=None, gen_target: dict=None, include_target: bool=False, **kwargs):
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
    new_X = safe_copy(X)
    new_y = safe_copy(y)
    try:
        if gen_target is not None:
            tmp_X = safe_copy(new_X)
            if new_y is not None:
                tmp_X[new_y.name] = new_y
            target_col = gen_target.get('target_col')
            target_gen_func = gen_target.get('target_gen_func')
            new_y = tmp_X.apply(target_gen_func, axis=1)
            new_y.name = target_col
            if include_target:
                new_X[target_col] = safe_copy(new_y)
            del tmp_X
    except Exception as warnEx:
        LOGGER.warning('Something went wrong with target variable generation, skipping: {}', warnEx)
        pass
    return new_X, new_y

class DataPrepTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols: list=None, int_cols: list=None, float_cols: list=None,
                 masked_cols: dict=None, special_features: dict=None, drop_feats_cols: bool=True,
                 calc_features: dict=None, gen_target: dict=None, correlated_cols: list=None,
                 pot_leak_cols: list=None, drop_missing: bool=False, clip_data: dict=None, include_target: bool=False, **kwargs):
        """
        Preprocessing dataframe columns to ensure consistent data types and formatting, and optionally extracting any special features described by dictionaries of feature mappings - e.g., special_features = {'col_label1': {'feat_mappings': {'Trailers': 'trailers', 'Deleted Scenes': 'deleted_scenes', 'Behind the Scenes': 'behind_scenes', 'Commentaries': 'commentaries'}, 'sep': ','}, }.
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
        :param calc_features: dictionary of calculated column labels with their corresponding column generation functions - e.g., {'col_label1': col_gen_func1, 'col_label2': col_gen_func2}
        :param gen_target: dictionary of target column label and target generation function (e.g., a lambda expression to be applied to rows (i.e., axis=1), such as {'target_col': 'target_collabel', 'target_gen_func': target_gen_func}
        :param correlated_cols: columns that are moderately to highly correlated and should be dropped
        :param pot_leak_cols: columns that could potentially introduce data leakage and should be dropped
        :param drop_missing: drop rows with missing data if True; default is False
        :param clip_data: clip the data based on categories defined by the filterby key and whether to enforec positive threshold defined by the pos_thres key - e.g., clip_data = {'rel_cols': ['col1', 'col2'], 'filterby': 'col_label1', 'pos_thres': False}
        :param include_target: include the target Series in the returned first item of the tuple if True (usually during exploratory analysis only); default is False (when as part of model pipeline)
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
        self.correlated_cols = correlated_cols
        self.pot_leak_cols = pot_leak_cols
        self.drop_missing = drop_missing
        self.clip_data = clip_data
        self.include_target = include_target
        self.target = None

    def fit(self, X, y=None):
        LOGGER.debug('DataPrepTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        return self

    def transform(self, X):
        LOGGER.debug('DataPrepTransformer: Transforming dataset, shape = {}', X.shape)
        # be sure to patch in any generated target column
        new_X, new_y = self.__do_transform(X)
        self.target = new_y
        LOGGER.debug('DataPrepTransformer: Transforming dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('DataPrepTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # be sure to patch in any generated target column
        self.target = y
        new_X, new_y = self.__do_transform(X, y)
        self.target = new_y
        LOGGER.debug('DataPrepTransformer: Fit-transformed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X, new_y = pre_process(X, y, date_cols=self.date_cols, int_cols=self.int_cols, float_cols=self.float_cols,
                                   masked_cols=self.masked_cols, special_features=self.special_features,
                                   drop_feats_cols=self.drop_feats_cols, gen_target=self.gen_target,
                                   calc_features=self.calc_features, correlated_cols=self.correlated_cols,
                                   pot_leak_cols=self.pot_leak_cols, drop_missing=self.drop_missing,
                                   clip_data=self.clip_data, include_target=self.include_target,)
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
            'pot_leak_cols': self.pot_leak_cols,
            'drop_missing': self.drop_missing,
            'clip_data': self.clip_data,
            'include_target': self.include_target,
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
        LOGGER.debug('SelectiveFunctionTransformer: Fit=transformed dataset, out shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **kwargs):
        new_X = safe_copy(X)
        for col in self.rel_cols:
            to_transform = safe_copy(X[col])
            fitted_X = super().transform(to_transform)
            if isinstance(fitted_X, np.ndarray):
                fitted_X = pd.DataFrame(fitted_X, columns=[col])
            new_X[col] = fitted_X[col].values if isinstance(fitted_X, pd.DataFrame) else fitted_X
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
    def __init__(self, lat_col: str, long_col: str, to_col: str, drop_geo_cols: bool=True, agg_func: str='median', **kwargs):
        """
        Transforms latitude-longitude point to a geohashed fixed neighborhood.
        :param lat_col: the column labels for desired latitude column
        :type lat_col: str
        :param long_col: the column labels for desired longitude column
        :type long_col: str
        :param to_col: the new generated column label for the geohashed fixed neighborhood
        :param drop_geo_cols: drops the latitude and longitude columns
        :param agg_func: aggregate function - e.g., 'mean', 'median', 'sum', 'min', 'max' - the default is 'median'
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
        self.geo_data_cols = [lat_col, long_col, to_col]
        self.drop_geo_cols = drop_geo_cols
        self.expected_spatial_features = None
        self.agg_func = agg_func

    def fit(self, X, y=None):
        LOGGER.debug('GeospatialTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # generate expected weights based on
        self.expected_spatial_features = GeospatialTransformer.__do_fit(X, y, self.geo_data_cols, agg_func=self.agg_func, )
        LOGGER.debug('GeospatialTransformer: Fitted dataset, out shape = {}, {}', X.shape, y.shape if y is not None else None)
        return self

    def transform(self, X, y=None):
        LOGGER.debug('GeospatialTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y,)
        LOGGER.debug('GeospatialTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('GeospatialTransformer: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        if self.expected_spatial_features is None:
            self.expected_spatial_features = GeospatialTransformer.__do_fit(X, y, self.geo_data_cols, agg_func=self.agg_func, )
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('GeospatialTransformer: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        gen_geohashes = GeospatialTransformer.__generate_geohashes(X, y, self.geo_data_cols, agg_func=self.agg_func, **fit_params)
        new_X = safe_copy(X)
        #new_X[self.to_col] = gen_geohashes.apply(lambda x: self.expected_spatial_features.loc[x[self.to_col]], axis=1)
        new_X[self.to_col] = gen_geohashes.apply(lambda x: GeospatialTransformer.__get_feature_value(self.expected_spatial_features, x[self.to_col]), axis=1)
        if self.drop_geo_cols:
            to_drop = [self.lat_col, self.long_col]
            new_X.drop(columns=to_drop, inplace=True)
        return new_X

    @staticmethod
    def __generate_geohashes(X, y=None, geo_data_cols=None, agg_func: str='median', precision: int=6, **fit_params):
        assert geo_data_cols is not None and not (not geo_data_cols), 'A valid list of latitude, longitude, and geohash column labels is expected'
        if geo_data_cols is None:
            geo_data_cols = ['latitude', 'longitude', 'geohash']
        new_X = safe_copy(X)
        # precision of 5 translates to ≤ 4.89km × 4.89km; 6 translates to ≤ 1.22km × 0.61km; 7 translates to ≤ 153m × 153m
        new_X[geo_data_cols[2]] = new_X.apply(lambda x: gh.encode(x[geo_data_cols[0]], x[geo_data_cols[1]], precision=precision), axis=1)
        return new_X

    @staticmethod
    def __do_fit(X, y=None, geo_data_cols=None, agg_func: str='median', **fit_params):
        gen_geohashes = GeospatialTransformer.__generate_geohashes(X, y, geo_data_cols, agg_func=agg_func, **fit_params)
        new_y = safe_copy(y)
        new_y = new_y.reset_index(drop=True) if isinstance(new_y, pd.Series) else new_y
        gen_geohashes['train_target'] = new_y.values if isinstance(new_y, pd.Series) else new_y
        expected_spatial_features = gen_geohashes.groupby(geo_data_cols[2]).agg({'train_target': agg_func})
        return expected_spatial_features

    @staticmethod
    def __get_feature_value(expected_spatial_features, geohash: str, missing_func: str='mode'):
        assert expected_spatial_features is not None, 'Valid expected_spatial_features is expected'
        assert geohash is not None and not (not geohash), 'A valid geohash is expected'
        try:
            return expected_spatial_features.loc[geohash].values[0]
        except KeyError as missing_err:
            # compute estimated spatial feature based proximity search of neighboring geohashes
            # focusing on all points that fall within a specific set of Geohashes
            # Geohash is useful for proximity searches because locations that are close to each other have similar Geohash prefixes
            # However: we must rely entirely on any insights extracted at training to avoid data leakage
            all_exp_hashes = expected_spatial_features.index.to_list()
            neighbors_vals = [expected_spatial_features.loc[the_gh] for the_gh in all_exp_hashes if the_gh.startswith(geohash[:len(geohash)-1])]
            if len(neighbors_vals) > 0:
                if missing_func == 'mode':
                    return np.median(neighbors_vals)
                else:
                    return np.mean(neighbors_vals)
            else:
                # defaults to mean of expected features
                if missing_func == 'mode':
                    return expected_spatial_features['train_target'].median()
                else:
                    return expected_spatial_features['train_target'].mean()

    @staticmethod
    def __get_neighbours(level: int, geohashes: list)-> list:
        assert isinstance(level, int) and level >= 1, 'Level must be a positive integer greater than 0'
        assert isinstance(geohashes, list) and all(isinstance(h, str) for h in geohashes), 'geohashes must be a list of strings'
        try:
            visited = set(geohashes)
            queue = geohashes.copy()
            next_level_items = []
            while level > 0:
                for hash in queue:
                    neighbors = gh.neighbours(hash)
                    next_level_items.extend(neighbors)
                    visited.update(neighbors)
                queue = next_level_items.copy()
                next_level_items.clear()
                level -= 1
            return list(visited)
        except ValueError as e:
            LOGGER.warning(f'Error encountered attempting to get geohash neighbors: {e}')
            return []
