import pandas as pd
import numpy as np
from cheutils.interceptor import PipelineInterceptor
from cheutils.common_utils import get_quantiles, safe_copy
from cheutils.loggers import LoguruWrapper
from scipy.stats import iqr

LOGGER = LoguruWrapper().get_logger()

class FeatureIntensityInterceptor(PipelineInterceptor):
    def __init__(self, rel_cols: list, group_by: list, suffix: str='_intensity', agg_func=None, **kwargs):
        """
        Create a new FeatureIntensityInterceptor instance.
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
        self.suffix = suffix
        self.agg_func = agg_func

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> pd.DataFrame:
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('FeatureIntensityInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = safe_copy(X)
        agg_func = self.agg_func if self.agg_func is not None else 'sum'
        feature_aggs = X.groupby(self.group_by)[self.rel_cols].transform(agg_func) + 1e-6
        for rel_col in self.rel_cols:
            if self.agg_func is not None:
                new_X.loc[:, rel_col + self.suffix] = feature_aggs[rel_col] / (feature_aggs[rel_col].max())
            else:
                new_X.loc[:, rel_col + self.suffix] = new_X[rel_col] / feature_aggs[rel_col]
        LOGGER.debug('FeatureIntensityInterceptor: dataset out, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X