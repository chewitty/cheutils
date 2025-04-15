import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class FeatureTrendsAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, periods: list, impute_vals:list = None, **kwargs):
        """
        Create a new FeatureTrendsAugmenter instance.
        :param rel_cols: the list of columns with series features to encode using a sine and cosine transformation
        with the corresponding matching periods
        :param periods: list of corresponding period values, matching the relevant series feature columns specified
        :param impute_vals: list of corresponding values to impute missing values for the corresponding features
        """
        assert rel_cols is not None or not (not rel_cols), 'Valid numeric periodic feature columns must be specified'
        assert periods is not None or not (not periods), 'Valid periods for the periodic features must be specified'
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.periods = periods
        self.impute_vals = impute_vals if impute_vals is not None and not (not impute_vals) else [0]*len(rel_cols)
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('FeatureTrendsAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('FeatureTrendsAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('FeatureTrendsAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if not self.fitted:
            raise RuntimeError('You have to call fit on the transformer before')
        # apply trend features
        new_X = X
        for idx, rel_col in enumerate(self.rel_cols):
            new_X.loc[:, rel_col + '_' + str(self.periods[idx]) + 'p_trend'] = new_X[rel_col].diff(self.periods[idx]).fillna(self.impute_vals[idx])
        return new_X