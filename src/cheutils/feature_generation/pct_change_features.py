import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from cheutils.interceptor import PipelineInterceptor
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class PctChangeInterceptor(PipelineInterceptor):
    def __init__(self, rel_cols: list, date_index: str, group_by: list = None, freq = None):
        """
        Create a new PctChangeInterceptor instance.
        :param rel_cols: the list of columns on which to compute percentage changes
        :type rel_cols:
        :param date_index: the date column to use as index for pct_change computation
        :type date_index:
        :param group_by: any required grouping as needed
        :type group_by:
        :param freq: the frequency to compute pct_change for
        """
        assert rel_cols is not None and not (not rel_cols), 'Valid list of features required'
        assert date_index is not None, 'Date feature (datetime format) must be provided'
        super().__init__()
        self.rel_cols = rel_cols
        self.date_index = date_index
        self.group_by = group_by
        self.freq = freq
        self.fitted = False

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('PctChangeInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X
        for rel_col in self.rel_cols:
            if self.group_by is not None:
                new_X.loc[:, rel_col + '_pct_change'] = new_X.groupby(self.group_by)[rel_col].pct_change(freq=self.freq).bfill()
            else:
                new_X.loc[:, rel_col + '_pct_change'] = new_X[rel_col].pct_change(freq=self.freq).bfill()
        LOGGER.debug('PctChangeInterceptor: dataset out, shape = {}, {}\nFeatures applied:\n{}', new_X.shape, y.shape if y is not None else None, self.rel_cols)
        return new_X, y