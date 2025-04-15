import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.common_utils import apply_clipping, parse_special_features, safe_copy, get_outlier_cat_thresholds, get_quantiles
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties
from cheutils.data_properties import DataPropertiesHandler
from cheutils.exceptions import FeatureGenException
from cheutils.data_prep_support import apply_replace_patterns, apply_calc_feature, force_joblib_cleanup
from joblib import Parallel, delayed
from pandas.api.types import is_datetime64_any_dtype
from scipy.stats import iqr
from typing import cast
from abc import ABC, abstractmethod

LOGGER = LoguruWrapper().get_logger()
class OutputTidying(ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'tidy') and
                callable(subclass.tidy) or
                NotImplemented)

    @abstractmethod
    def tidy(self, transformed_X: [pd.DataFrame, np.ndarray], feature_names_out: list, **params) -> (pd.DataFrame, list):
        raise NotImplementedError

class OutputWrapper(OutputTidying):
    def __init__(self, num_transformers: int=1, feature_names: list=None,):
        super().__init__()
        self.num_transformers = num_transformers
        self.feature_names = feature_names

    def tidy(self, transformed_X: [pd.DataFrame, np.ndarray], feature_names_out: list, **params) -> (pd.DataFrame, list):
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
        # re-order columns, so the altered columns appear at the end
        add_back_feats = []
        for feature_name in self.feature_names:
            if feature_name in desired_feature_names:
                desired_feature_names.remove(feature_name)
                add_back_feats.append(feature_name)
        desired_feature_names.extend(add_back_feats)
        return new_X[desired_feature_names]

class BasicTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformers, remainder='passthrough', force_int_remainder_cols: bool = False,
                 verbose=False, n_jobs=None, worker: OutputTidying=None, **kwargs):
        super().__init__()
        self.transformers = transformers
        self.remainder = remainder
        self.force_int_remainder_cols = force_int_remainder_cols
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.worker = worker
        self.underlying_transformer = ColumnTransformer(transformers=transformers, remainder=remainder,
                                                        force_int_remainder_cols=force_int_remainder_cols,
                                                        verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.fitted = False

    def fit(self, X, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('BasicTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.underlying_transformer.fit(X, y, **fit_params)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('BasicTransformer: Transforming dataset, shape = {}', X.shape)
        new_X = self.underlying_transformer.transform(X)
        if self.worker is not None:
            feature_names_out = self.underlying_transformer.get_feature_names_out().tolist()
            new_X = self.worker.tidy(new_X, feature_names_out=feature_names_out, **fit_params)
        LOGGER.debug('BasicTransformer: Transformed dataset, out shape = {}', new_X.shape)
        return new_X

class SelectiveScaler(BasicTransformer):
    def __init__(self, transformers, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        assert transformers is not None and len(transformers) > 0, 'Valid transfomers required'
        super().__init__(transformers=transformers, remainder=remainder,
                         force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True, verbose=verbose, n_jobs=n_jobs,
                         worker=OutputWrapper(num_transformers=len(transformers), feature_names=transformers[0][2]),
                         **kwargs)

class SelectiveEncoder(BasicTransformer):
    def __init__(self, transformers, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        assert transformers is not None and len(transformers) > 0, 'Valid transfomers required'
        super().__init__(transformers=transformers, remainder=remainder,
                         force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True, verbose=verbose, n_jobs=n_jobs,
                         worker=OutputWrapper(num_transformers=len(transformers), feature_names=transformers[0][2]),
                         **kwargs)

class SelectiveBinarizer(BasicTransformer):
    def __init__(self, transformers, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        assert transformers is not None and len(transformers) > 0, 'Valid transfomers required'
        super().__init__(transformers=transformers, remainder=remainder,
                         force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True, verbose=verbose, n_jobs=n_jobs,
                         worker=OutputWrapper(num_transformers=len(transformers), feature_names=transformers[0][2]),
                         **kwargs)

class SelectiveTargetEncoder(BasicTransformer):
    def __init__(self, transformers, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        super().__init__(transformers=transformers, remainder=remainder,
                         force_int_remainder_cols=force_int_remainder_cols,
                         verbose_feature_names_out=True, verbose=verbose, n_jobs=n_jobs,
                         worker=OutputWrapper(num_transformers=len(transformers), feature_names=transformers[0][2]),
                         **kwargs)

class TSSelectiveTargetEncoder(SelectiveTargetEncoder):
    def __init__(self, transformers, lag_features: dict, column_ts_index, remainder='passthrough', force_int_remainder_cols: bool=False,
                 verbose=False, n_jobs=None, **kwargs):
        """
        create instance of TSSelectiveTargetEncoder.
        :param transformers: configured transformers
        :type transformers:
        :param lag_features: dictionary of calculated column labels to hold lagging calculated values with their corresponding column lagging calculation functions - e.g., {'sort_by_cols': ['sort_by_col1', 'sort_by_col2'], period=1, 'freq': 'D', 'drop_rel_cols': False, }
        :type lag_features: dict
        :param column_ts_index: The column with the time series date feature relevant for sorting; if not specified assumed to be the same as column_sort
        :type column_ts_index: basestring
        :param remainder:
        :type remainder:
        :param force_int_remainder_cols:
        :type force_int_remainder_cols:
        :param verbose:
        :type verbose:
        :param n_jobs:
        :type n_jobs:
        :param kwargs:
        :type kwargs:
        """
        assert lag_features is not None and not (not lag_features), 'Lag features specification must be provided'
        assert column_ts_index is not None, 'A date feature/column must be provided'
        super().__init__(transformers=transformers,
                         remainder=remainder, force_int_remainder_cols=force_int_remainder_cols,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)
        self.lag_features = lag_features
        self.column_ts_index = column_ts_index
        self.fitted = False

    def fit(self, X, y=None, **fit_params):
        if self.fitted:
            return self
        target_name = y.name
        periods = self.lag_features.get('periods')
        freq = self.lag_features.get('freq')
        sort_by_cols = self.lag_features.get('sort_by_cols')
        timeseries_container: pd.DataFrame = safe_copy(X)
        timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
        timeseries_container.sort_values(by=sort_by_cols, inplace=True)
        timeseries_container['index'] = timeseries_container[self.column_ts_index]
        timeseries_container.set_index('index', inplace=True)
        timeseries_container = timeseries_container.shift(periods=periods + 1, freq=freq).bfill()
        new_y = pd.Series(timeseries_container[target_name].values, index=y.index, name=target_name)
        del timeseries_container
        self.fitted = True
        return super().fit(X, new_y, **fit_params)

class PreOrPostDataPrep(TransformerMixin, BaseEstimator):
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
        super().__init__(**kwargs)
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
        LOGGER.debug('PreOrPostDataPrep: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
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
        LOGGER.debug('PreOrPostDataPrep: Transforming dataset, shape = {}', X.shape)
        # be sure to patch in any generated target column
        new_X, new_y = self.__do_transform(X)
        self.target = new_y if new_y is not None else self.target
        LOGGER.debug('PreOrPostDataPrep: Transformed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
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
        LOGGER.debug('PreOrPostDataPrep: Pre-processing dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X
        new_y = y
        # process columns with  strings to replace patterns
        if self.replace_patterns is not None:
            if len(self.replace_patterns) > 1:
                result_out = Parallel(n_jobs=-1)(delayed(apply_replace_patterns)(new_X, replace_dict) for replace_dict in self.replace_patterns)
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
            new_X = apply_clipping(new_X, rel_cols=rel_cols, group_by=filterby, pos_thres=pos_thres)
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
        LOGGER.debug('PreOrPostDataPrep: Pre-processed dataset, out shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
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
        LOGGER.debug('PreOrPostDataPrep: Post-processing dataset, out shape = {}', X.shape)
        new_X = X
        if correlated_cols is not None or not (not correlated_cols):
            to_drop = [col for col in correlated_cols if col in new_X.columns]
            new_X.drop(columns=to_drop, inplace=True)
        if pot_leak_cols is not None or not (not pot_leak_cols):
            to_drop = [col for col in pot_leak_cols if col in new_X.columns]
            new_X.drop(columns=to_drop, inplace=True)
        LOGGER.debug('PreOrPostDataPrep: Post-processed dataset, out shape = {}', new_X.shape)
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
            results_out = Parallel(n_jobs=-1)(delayed(apply_calc_feature)(X, col, col_gen_func_dict) for col, col_gen_func_dict in calc_features.items())
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

class OutlierClipper(TransformerMixin, BaseEstimator):
    def __init__(self, rel_cols: list, group_by: list,
                 l_quartile: float = 0.25, u_quartile: float = 0.75, pos_thres: bool=False, **kwargs):
        """
        Create a new OutlierClipper instance.
        :param rel_cols: the list of columns to clip
        :param group_by: list of columns to group by or filter the data by
        :param l_quartile: the lower quartile (float between 0 and 1)
        :param u_quartile: the upper quartile (float between 0 and 1 but greater than l_quartile)
        :param pos_thres: enforce positive clipping boundaries or thresholds values
        """
        assert rel_cols is not None or not (not rel_cols), 'Valid numeric feature columns must be specified'
        assert group_by is not None or not (not group_by), 'Valid numeric feature columns must be specified'
        super().__init__(**kwargs)
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
        LOGGER.debug('OutlierClipper: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
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
        LOGGER.debug('OutlierClipper: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('OutlierClipper: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
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

def get_scaler(remainder='passthrough', force_int_remainder_cols: bool=False,
               verbose=False, n_jobs=None, **kwargs):
    # if configuring more than one column transformer make sure verbose_feature_names_out=True
    # to ensure the prefixes ensure uniqueness in the feature names
    __data_handler: DataPropertiesHandler = cast(DataPropertiesHandler, AppProperties().get_subscriber('data_handler'))
    conf_transformers = __data_handler.get_scalers()
    return SelectiveScaler(transformers=conf_transformers, remainder=remainder,
                           force_int_remainder_cols=force_int_remainder_cols, verbose=verbose, )

def get_encoder(remainder='passthrough', force_int_remainder_cols: bool=False,
               verbose=False, n_jobs=None, **kwargs):
    # if configuring more than one column transformer make sure verbose_feature_names_out=True
    # to ensure the prefixes ensure uniqueness in the feature names
    __data_handler: DataPropertiesHandler = cast(DataPropertiesHandler, AppProperties().get_subscriber('data_handler'))
    conf_transformers = __data_handler.get_encoders()
    return SelectiveEncoder(transformers=conf_transformers, remainder=remainder,
                           force_int_remainder_cols=force_int_remainder_cols, verbose=verbose, )

def get_binarizer(remainder='passthrough', force_int_remainder_cols: bool=False,
                  verbose=False, n_jobs=None, **kwargs):
    # if configuring more than one column transformer make sure verbose_feature_names_out=True
    # to ensure the prefixes ensure uniqueness in the feature names
    __data_handler: DataPropertiesHandler = cast(DataPropertiesHandler, AppProperties().get_subscriber('data_handler'))
    conf_transformers = __data_handler.get_binarizers()
    return SelectiveBinarizer(transformers=conf_transformers, remainder=remainder,
                              force_int_remainder_cols=force_int_remainder_cols, verbose=verbose, )

def get_target_encoder(lag_features: dict=None, column_ts_index: str=None,
                       remainder='passthrough', force_int_remainder_cols: bool=False,
                       verbose=False, n_jobs=None, **kwargs):
    # if configuring more than one column transformer make sure verbose_feature_names_out=True
    # to ensure the prefixes ensure uniqueness in the feature names
    __data_handler: DataPropertiesHandler = cast(DataPropertiesHandler, AppProperties().get_subscriber('data_handler'))
    conf_transformers = __data_handler.get_target_encoder()
    if lag_features is not None:
        return TSSelectiveTargetEncoder(transformers=conf_transformers, lag_features=lag_features, column_ts_index=column_ts_index,
                                        remainder=remainder, force_int_remainder_cols=force_int_remainder_cols, verbose=verbose, )
    else:
        return SelectiveTargetEncoder(transformers=conf_transformers, remainder=remainder,
                                      force_int_remainder_cols=force_int_remainder_cols, verbose=verbose, )