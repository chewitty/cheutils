import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.common_utils import safe_copy
from cheutils.project_tree import save_excel
from cheutils.loggers import LoguruWrapper
import tsfresh.defaults
from tsfresh.feature_extraction import extract_features

LOGGER = LoguruWrapper().get_logger()

class TSBasicFeatureAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, col_periods: list, group_by: list, ts_index_col: str=None, freq: str=None,
                 include_target: bool=False, **kwargs):
        assert rel_cols is not None and not (not rel_cols), 'Features requiring lag must be provided'
        assert col_periods is not None and not (not col_periods), 'Lag periods to be applied to each feature must be provided'
        assert group_by is not None and not (not group_by), 'Feature(s) to id or group by must be provided'
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.col_periods = col_periods
        self.group_by = group_by
        self.ts_index_col = ts_index_col if ts_index_col is not None else 'date'
        self.freq = freq
        self.include_target = include_target
        self.lagged_features = None
        self.suffix = '_lag'
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('TSBasicFeatureAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        base_feats = self.group_by.copy()
        base_feats.append(self.ts_index_col)
        self.lagged_features = safe_copy(X[base_feats])
        self.lagged_with_target = safe_copy(X[base_feats])
        timeseries_container = safe_copy(X)
        if self.include_target:
            timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
            self.lagged_with_target = pd.concat([self.lagged_with_target, safe_copy(y)], axis=1)
        if self.ts_index_col in timeseries_container.columns:
            timeseries_container.set_index(self.ts_index_col, inplace=True)
        timeseries_container = timeseries_container.sort_values(base_feats)
        for col in self.rel_cols:
            for period in self.col_periods:
                suffix = f'{self.suffix}_{period}'
                if self.freq is not None:
                    self.lagged_features.loc[:, col + suffix] = timeseries_container.groupby(self.group_by)[col].shift(period, freq=self.freq).reset_index()[col].values
                else:
                    self.lagged_features.loc[:, col + suffix] = timeseries_container.groupby(self.group_by)[col].shift(period).reset_index()[col].values
        self.lagged_features.bfill(inplace=True)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('TSBasicFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSBasicFeatureAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if self.lagged_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        column_id = self.group_by.copy()
        column_id.append(self.ts_index_col)
        timeseries_container = safe_copy(X)
        if self.include_target:
            target_col = str(self.lagged_with_target.columns[-1])
            timeseries_container = pd.concat([timeseries_container, self.lagged_with_target[target_col]], axis=1)
            is_surplus = timeseries_container[column_id].isna().any(axis=1)
            timeseries_container = timeseries_container[~is_surplus]
            timeseries_container[target_col] = timeseries_container[target_col].fillna(self.lagged_with_target[target_col])
            timeseries_container[target_col] = timeseries_container[target_col].fillna(self.lagged_with_target[target_col].mean())
        if self.ts_index_col in timeseries_container.columns:
            timeseries_container.set_index(self.ts_index_col, inplace=True)
        timeseries_container = timeseries_container.sort_values(column_id)
        new_X = safe_copy(X)
        for col in self.rel_cols:
            for period in self.col_periods:
                suffix = f'{self.suffix}_{period}'
                if self.freq is not None:
                    shifted_data = timeseries_container.groupby(self.group_by, level=0)[col].shift(period, freq=self.freq).reset_index()[col]
                else:
                    shifted_data = timeseries_container.groupby(self.group_by, level=0)[col].shift(period).reset_index()[col]
                if isinstance(shifted_data, pd.DataFrame):
                    shifted_data = shifted_data.loc[:, ~shifted_data.columns.duplicated(keep='last')]
                new_X.loc[:, col + suffix] = shifted_data.values
                new_X.loc[:, col + suffix] = new_X[col + suffix].fillna(self.lagged_features[col + suffix])
                new_X.loc[:, col + suffix] = new_X[col + suffix].fillna(self.lagged_features[col + suffix].mean())
        #new_X.set_index(X.index, inplace=True)
        del timeseries_container
        return new_X

"""
Adapted from https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/transformers/feature_augmenter.html
"""
class TSFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
        Sklearn-compatible estimator, for calculating and adding many features from a given time series
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

        >>> from cheutils import TSFeatureAugmenter
        >>> time_series = read_in_timeseries() # get the development of the shares
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
                 include_target:bool=False, **kwargs):
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
        :param column_id: The name of the id column to group by. See :mod:`~tsfresh.feature_extraction.extraction`.
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
        super().__init__(**kwargs)
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

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSFeatureAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
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
                 column_sort=None, column_kind=None, column_value=None, ts_index_col: str=None,
                 chunksize=tsfresh.defaults.CHUNKSIZE, n_jobs=tsfresh.defaults.N_PROCESSES,
                 show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                 disable_progressbar=tsfresh.defaults.DISABLE_PROGRESSBAR,
                 impute_function=tsfresh.defaults.IMPUTE_FUNCTION, profile=tsfresh.defaults.PROFILING,
                 profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                 profiling_sorting=tsfresh.defaults.PROFILING_SORTING,
                 drop_rel_cols: dict=None, lag_target: bool=False, **kwargs):
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
        :param column_id: The name of the id column to group by. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_id: basestring
        :param column_sort: The column with the sort data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_sort: basestring
        :param column_kind: The column with the kind data. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_kind: basestring
        :param column_value: The column with the values. See :mod:`~tsfresh.feature_extraction.extraction`.
        :type column_value: basestring
        :param ts_index_col: The column with the time series date feature relevant for sorting; if not specified assumed to be the same as column_sort
        :type ts_index_col: basestring
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
        super().__init__(**kwargs)
        self.lag_features = lag_features
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value
        self.ts_index_col = ts_index_col if ts_index_col is not None else column_sort
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
        self.target_name = None
        self.fitted = False

    def fit(self, X=None, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('TSLagFeatureAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target_name = y.name
        self.extracted_features = self.__extract_features(X, y, **fit_params)
        for col in list(self.extracted_features.columns):
            self.extracted_global_aggs[col] = self.extracted_features.reset_index()[col].agg('median')
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSLagFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSLagFeatureAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __extract_features(self, X, y=None, **fit_params):
        timeseries_container: pd.DataFrame = safe_copy(X)
        if self.lag_target:
            if y is not None:
                timeseries_container = pd.concat([timeseries_container, safe_copy(y)], axis=1)
            else:
                timeseries_container.loc[:, self.target_name] = 0
        periods = self.lag_features.get('periods')
        freq = self.lag_features.get('freq')
        sort_by_cols = list(self.lag_features.get('sort_by_cols'))
        sort_by_cols.append(self.ts_index_col)
        timeseries_container.sort_values(by=sort_by_cols, inplace=True)
        timeseries_container['index'] = timeseries_container[self.ts_index_col]
        timeseries_container.set_index('index', inplace=True)
        timeseries_container = timeseries_container.shift(periods=periods + 1, freq=freq).bfill()
        if timeseries_container is None:
            raise RuntimeError('You have to provide a time series container/dataframe before.')
        # extract the features
        extracted_features = extract_features(timeseries_container,
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
        extracted_features = extracted_features.bfill()
        extracted_features.fillna(0, inplace=True)
        extracted_features.index.rename(self.column_id, inplace=True)
        cols = list(extracted_features.columns)
        col_map = {col: col.replace('__', '_lag_') for col in cols}
        extracted_features.rename(columns=col_map, inplace=True)
        del timeseries_container
        return extracted_features

    def __do_transform(self, X, y=None, **fit_params):
        if self.extracted_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        new_X = pd.merge(X, self.extracted_features, left_on=self.column_id, right_index=True, how='left')
        new_X.set_index(X.index, inplace=True)
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
                 shift_periods: int=15, freq: str=None, ts_index_col: str=None,
                 filter_by:str='date', agg_func: str=None, suffix: str='_rol', **kwargs):
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
        :param ts_index_col: The column with the time series date feature relevant for sorting; if not specified assumed to be the same as column_sort
        :type ts_index_col: basestring
        :param filter_by: filter the data by this column label - the name of the id column to group by
        :type filter_by: basestring
        :param agg_func: The aggregation function to apply to each column
        :type agg_func: basestring
        :param suffix: the suffix to add to the column names
        """
        assert roll_cols is not None, 'Valid roll columns or features must be specified'
        super().__init__(**kwargs)
        self.roll_cols = roll_cols
        self.roll_target = roll_target
        self.window = window
        self.shift_periods = shift_periods
        self.freq = freq
        self.ts_index_col = ts_index_col
        self.filter_by = filter_by
        self.agg_func = agg_func if agg_func is not None else ['mean', 'std']
        self.suffix = suffix
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
        timeseries_container.set_index(self.ts_index_col, inplace=True)
        # extract the features
        self.extracted_features = timeseries_container.groupby(self.filter_by)[all_cols_to_roll].apply(lambda x: x.shift(self.shift_periods, freq=self.freq)).rolling(window=self.window + 1, min_periods=1).agg(self.agg_func)
        self.extracted_features = self.extracted_features.bfill()
        self.extracted_features.fillna(0, inplace=True)
        if isinstance(self.agg_func, list) and len(self.agg_func) > 1:
            self.extracted_features = self.extracted_features.reset_index(names=[self.filter_by, self.ts_index_col])
            self.extracted_features.columns = self.extracted_features.columns.map('_'.join)
            col_map = {roll_col + '_mean': roll_col.replace(roll_col, roll_col + '_' + str(self.shift_periods) + self.suffix + '_mean') for roll_col in all_cols_to_roll}
            col_map_std = {roll_col + '_std': roll_col.replace(roll_col, roll_col + '_' + str(self.shift_periods) + self.suffix + '_std') for roll_col in all_cols_to_roll}
            for entry_key, entry_value in col_map_std.items():
                col_map[entry_key] = entry_value
        else:
            self.extracted_features = self.extracted_features.reset_index()
            col_map = {roll_col: roll_col.replace(roll_col, roll_col + '_' + str(self.shift_periods) + self.suffix + '_' + self.agg_func) for roll_col in all_cols_to_roll}
        self.extracted_features.rename(columns=col_map, inplace=True)
        cols = list(self.extracted_features.columns)
        self.extracted_features.loc[:, cols[-len(all_cols_to_roll):]] = self.extracted_features[cols[-len(all_cols_to_roll):]].bfill()
        if isinstance(self.agg_func, list) and len(self.agg_func) > 1:
            self.extracted_features.columns = [col_name.rstrip('_') for col_name in self.extracted_features.columns]
            cols = list(self.extracted_features.columns)[-len(all_cols_to_roll)*len(self.agg_func):]
            for col_indx in range(len(all_cols_to_roll)):
                indx = col_indx
                for agg_func in self.agg_func:
                    self.extracted_global_aggs[cols[col_indx+indx]] = self.extracted_features[cols[col_indx+indx]].agg(agg_func)
                    indx += 1
        else:
            for col in cols[-len(all_cols_to_roll):]:
                self.extracted_global_aggs[col] = self.extracted_features[col].agg(self.agg_func)
        del timeseries_container
        self.extracted_features.drop_duplicates(subset=[self.filter_by, self.ts_index_col], keep='last', inplace=True)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        """
        Add the features calculated using the timeseries_container and add them to the corresponding rows in the input
        pandas.DataFrame X.

        :param X: the DataFrame to which the calculated timeseries features will be added. This is *not* the
               dataframe with the timeseries itself.
        :type X: pandas.DataFrame

        :return: The input DataFrame, but with added features.
        :rtype: pandas.DataFrame
        """
        LOGGER.debug('TSRollingLagFeatureAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('TSRollingLagFeatureAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if self.extracted_features is None:
            raise RuntimeError('You have to call fit on the transformer before')
        # add newly created features to dataset
        common_key = [self.filter_by, self.ts_index_col]
        new_X = pd.merge(X, self.extracted_features, how='left', left_on=common_key, right_on=common_key)
        feat_cols = list(self.extracted_features.columns)
        for feat_col in feat_cols:
            if feat_col not in common_key:
                new_X.loc[:, feat_col] = new_X[feat_col].fillna(self.extracted_global_aggs.get(feat_col))
        return new_X