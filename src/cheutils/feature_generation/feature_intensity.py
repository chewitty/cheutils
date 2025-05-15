import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.common_utils import get_quantiles
from cheutils.loggers import LoguruWrapper
from scipy.stats import iqr

LOGGER = LoguruWrapper().get_logger()

class FeatureIntensityAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, group_by: list, suffix: str='_intensity', agg_func=None, **kwargs):
        """
        Create a new FeatureIntensityAugmenter instance.
        :param rel_cols: the list of columns with features to compute intensities
        :param group_by: any necessary category to group aggregate stats by - default is None
        :param suffix: suffix to add to column name
        :param agg_func: aggregation function, if any
        """
        assert rel_cols is not None or not (not rel_cols), 'Valid numeric feature columns must be specified'
        assert group_by is not None or not (not group_by), 'Valid group or category identifiers must be specified'
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.group_by = group_by
        self.suffix = suffix if agg_func is None else '_inten_proxy'
        self.agg_func = agg_func if agg_func is not None else 'sum'
        self.feature_aggs = None
        self.fitted = False

    def fit(self, X=None, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('FeatureIntensityAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.feature_aggs = X.groupby(self.group_by)[self.rel_cols].transform(self.agg_func) + 1e-6
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('FeatureIntensityAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('FeatureIntensityAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if not self.fitted:
            raise RuntimeError('You have to call fit on the transformer before')
        new_X = X
        for rel_col in self.rel_cols:
            if self.agg_func is not None:
                new_X.loc[:, rel_col + self.suffix] = self.feature_aggs[rel_col] / (self.feature_aggs[rel_col].max())
            else:
                new_X.loc[:, rel_col + self.suffix] = new_X[rel_col] / self.feature_aggs[rel_col]
        return new_X