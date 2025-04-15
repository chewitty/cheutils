import pandas as pd
from cheutils import safe_copy
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor

LOGGER = LoguruWrapper().get_logger()

class NumericDataInterceptor(PipelineInterceptor):
    """
    Attempts to transform all features to numeric types prior to inputting to the pipeline estimator.
    """
    def __init__(self):
        super().__init__()

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        """
        Transforms all features or relevant columns to numeric types in readiness for the last pipeline step or estimator.
        :param X: dataframe with features
        :type X: pd.DataFrame
        :param y: series with target values
        :type y: pd.Series
        :return: a tuple of transformed X and y (which remains untouched in this case)
        :rtype: (pd.DataFrame, pd.Series)
        """
        assert X is not None, 'Valid dataframe with data required'
        new_X = safe_copy(X)
        new_y = y
        for col in new_X.columns:
            try:
                new_X[col] = pd.to_numeric(new_X[col], )
            except ValueError as ignore:
                LOGGER.warning('Potential dtype issue: {}', ignore)
        return new_X, new_y