import pandas as pd
import numpy as np
import cheutils
from fast_ml import eda
from fast_ml.utilities import display_all
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
from cheutils.common_utils import apply_clipping
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

def parse_special_features(special_feat_str, feature_mappings: dict):
    """
    Returns a dictionary with flags indicating the special features included in the movie
    :param special_feat_str: any relevant string that may contain some special features to be matched - e.g., "Trailers, commentaries, feat1, feat2,"
    :type special_feat_str: str
    :param feature_mappings: complete dictionary of features or tokens to be matched in feature string and their corresponding desired labels - e.g., {"feat1": "label1", "feat2": "label2", "Trailers": "trailers"}
    :type feature_mappings: dict
    :return: list of flags indicating the special features, identified by the feature mapping keys, included in the input string correctly matched
    :rtype: list
    """
    assert special_feat_str is not None, "Special features expected"
    assert feature_mappings is not None, 'Special feature mappings expected'
    feat_split = special_feat_str.split(",")
    feat_keys = list(feature_mappings.keys())
    feat_mappings = list(feature_mappings.values())
    feat_pattern = {mapping: 0 for mapping in feat_mappings}
    for feat_key in feat_keys:
        if feat_key in feat_split:
            feat_pattern[feature_mappings.get(feat_key)] = 1
    return list(feat_pattern.values())

def summarize(df: pd.DataFrame, display: bool = True):
    """
    Generate useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent
    :param df: specified dataframe
    :param display: whether to display the dataframe
    :return: summary dataframe or None
    :rtype:
    """
    summary_df = eda.df_info(df)
    if display:
        display_all(summary_df)
    else:
        return summary_df

def drop_missing(df: pd.DataFrame, target_sr: pd.Series = None):
    """
    Drop rows with missing data
    :param df: dataframe with the specified columns, which may not contain any target class labels
    :param target_sr: optional target class labels corresponding to the dataframe
    :return: revised dataframe and corresponding target series where present
    """
    assert df is not None, 'A valid DataFrame expected as input'
    clean_df = df.copy(deep=True)
    clean_target_sr = target_sr.copy(deep=True) if (target_sr is not None) else None
    null_rows = clean_df.isna().any(axis=1)
    clean_df = clean_df.dropna()
    # do not reset index here
    #clean_df.reset_index(drop=True, inplace=True)
    LOGGER.debug('Rows with missing data = {}', len(df) - len(clean_df))
    if target_sr is not  None:
        clean_target_sr = clean_target_sr[~null_rows]
        # do not reset index here
        #clean_target_sr.reset_index(drop=True)
    return clean_df, clean_target_sr

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

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms dates.
    """
    def __init__(self, rel_cols: list, prefixes: list, **kwargs):
        super().__init__(**kwargs)
        self.target = None
        self.rel_cols = rel_cols
        self.prefixes = prefixes
        self.trans_cols = rel_cols

    def fit(self, X, y=None):
        LOGGER.debug('DateTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y  # possibly passed in chain
        return self

    def transform(self, X, y=None):
        LOGGER.debug('DateTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X.copy(deep=True)
        new_X.reset_index(drop=True, inplace=True)
        # otherwise also generate the following features
        for rel_col, prefix in zip(self.rel_cols, self.prefixes):
            new_X.loc[:, prefix + 'dow'] = new_X[rel_col].dt.dayofweek
            null_dayofweek = new_X[prefix + 'dow'].isna()
            nulldofwk = new_X[null_dayofweek]
            new_X[prefix + 'dow'] = new_X[prefix + 'dow'].astype(int)
            new_X.loc[:, prefix + 'wk'] = new_X[rel_col].apply(lambda x: pd.Timestamp(x).week)
            new_X[prefix + 'wk'] = new_X[prefix + 'wk'].astype(int)
            #new_X.loc[:, prefix + 'doy'] = new_X[rel_col].dt.dayofyear
            #new_X[prefix + 'doy'] = new_X[prefix + 'doy'].astype(int)
            new_X.loc[:, prefix + 'qtr'] = new_X[rel_col].dt.quarter
            new_X[prefix + 'qtr'] = new_X[prefix + 'qtr'].astype(int)
            new_X.loc[:, prefix + 'wkend'] = np.where(new_X[rel_col].dt.dayofweek.isin([5, 6]), 1, 0)
            new_X[prefix + 'wkend'] = new_X[prefix + 'wkend'].astype(int)
        if len(self.rel_cols) > 0:
            new_X.drop(columns=self.rel_cols, inplace=True)
        LOGGER.debug('DateTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def get_date_cols(self):
        """
        Returns the transformed date columns, if any
        :return:
        """
        return self.trans_cols

    def get_target(self):
        return self.target

class SpecialFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms dates.
    """
    def __init__(self, rel_col: str, feature_mappings: dict, **kwargs):
        super().__init__(**kwargs)
        self.target = None
        self.rel_col = rel_col
        self.feature_mappings = feature_mappings
        self.trans_cols = rel_col

    def fit(self, X, y=None):
        LOGGER.debug('SpecialFeaturesTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y  # possibly passed in chain
        return self

    def transform(self, X, y=None):
        LOGGER.debug('SpecialFeaturesTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X.copy(deep=True)
        new_X.reset_index(drop=True, inplace=True)
        # otherwise also generate the following features
        num_mappings = len(self.feature_mappings.values())
        created_features = new_X[self.rel_col].apply(lambda x: parse_special_features(x, self.feature_mappings))
        new_feat_values = {mapping: [] for mapping in self.feature_mappings.values()}
        for index, col in enumerate(self.feature_mappings.values()):
            for row in range(created_features.shape[0]):
                new_feat_values.get(col).append(created_features[row][index])
            new_X.loc[:, col] = new_feat_values.get(col)
        if self.rel_col is not None:
            new_X.drop(columns=[self.rel_col], inplace=True)
        LOGGER.debug('SpecialFeaturesTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def get_date_cols(self):
        """
        Returns the transformed date columns, if any
        :return:
        """
        return self.trans_cols

    def get_target(self):
        return self.target

class SelectiveRobustScaler(RobustScaler):
    """
    Selectively scale only columns specified in dataframe
    """
    def __init__(self, rel_cols: list, **kwargs):
        super().__init__(**kwargs)
        assert (rel_cols is not None) or not (not rel_cols), 'A valid list or non-empty list of column labels expected as input'
        self.rel_cols = rel_cols
        self.target = None
        self.available_cols = None
        self.fill_back_cols = None

    def fit(self, X, y=None):
        LOGGER.debug('SelectiveRobustScaler: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y  # possibly passed in chain
        self.fill_back_cols = list(X.columns)
        self.available_cols = [col for col in self.rel_cols if col in X.columns]
        if len(self.available_cols) > 0:
            return super().fit(X[self.available_cols], y)
        else:
            return self

    def transform(self, X, y=None):
        LOGGER.debug('SelectiveRobustScaler: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # Later use to add back remaining columns
        if isinstance(X, pd.Series):
            target_X = super().transform(pd.DataFrame({X.name: X}))
            target_X = target_X[X.name]
            return target_X
        tmp_X = X.reset_index(drop=True)
        if len(self.available_cols) > 0:
            transformed_X = super().transform(X[self.available_cols])
            orig_cols = list(X.columns) if (X is not None) else self.fill_back_cols
            #LOGGER.debug('Columns at transform = {}', orig_cols)
            new_X = pd.DataFrame(transformed_X, columns=self.available_cols)
            #LOGGER.debug('Original shape = {}', tmp_X.shape)
            for col in orig_cols:
                if col in self.available_cols:
                    pass
                else:
                    new_X.loc[:, col] = tmp_X[col]
        else:
            new_X = tmp_X
        return new_X

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
        self.estimator.fit(X, y)
        #LOGGER.debug('FeatureSelectionTransformer: Feature coefficients = {}', self.estimator.coef_)
        return super().fit(X, y, **fit_params)

    def transform(self, X, y=None):
        LOGGER.debug('FeatureSelectionTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        transformed_X = super().transform(X)
        self.selected_cols = list(self.estimator.feature_names_in_[self.support_])
        new_X = pd.DataFrame(transformed_X, columns=self.selected_cols)
        LOGGER.debug('FeatureSelectionTransformer: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        LOGGER.debug('FeatureSelectionTransformer: Features selected = {}', self.selected_cols)
        return new_X

    def get_selected_features(self):
        """
        Return the selected features or column labels.
        :return:
        """
        return self.selected_cols

    def get_target(self):
        return self.target

class DropMissingDataTransformer(BaseEstimator, TransformerMixin):
    """
    Drops rows with missing data from the dataframe.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = None

    def fit(self, X, y=None):
        LOGGER.debug('DropMissingDataTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        return self

    def transform(self, X, y=None):
        LOGGER.debug('DropMissingDataTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # it appears the y is never really passed on from fit_transform and so we use a record already held
        new_X, self.target = drop_missing(X, target_sr=self.target)
        y = self.target
        return new_X, y

    def get_target(self):
        """
        Returns the transformed target if any
        :return:
        """
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
        new_X = drop_selected(X, rel_cols=self.rel_cols)
        return new_X

    def get_target(self):
        """
        Returns the transformed target if any
        :return:
        """
        return self.target

class ClipDataTransformer(BaseEstimator, TransformerMixin):
    """
    Clip data values assessed as outliers.
    """
    def __init__(self, rel_cols: list = None, filterby:str = None, pos_thres: bool=False, **kwargs):
        super().__init__()
        self.rel_cols = rel_cols
        self.filterby = filterby
        self.pos_thres = pos_thres
        self.target = None

    def fit(self, X, y=None):
        LOGGER.debug('ClipDataTransformer: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.target = y
        return self

    def transform(self, X, y=None):
        LOGGER.debug('ClipDataTransformer: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = apply_clipping(X, rel_cols=self.rel_cols, filterby=self.filterby, pos_thres=self.pos_thres, )
        return new_X

    def get_clipped_cols(self):
        """
        Returns the clipped columns, if any
        :return:
        """
        return self.rel_cols

    def get_target(self):
        return self.target
