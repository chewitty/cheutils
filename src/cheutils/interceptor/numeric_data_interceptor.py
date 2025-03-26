import pandas as pd
from cheutils import safe_copy
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor

LOGGER = LoguruWrapper().get_logger()

class NumericDataInterceptor(PipelineInterceptor):
    def __init__(self):
        super().__init__()

    def apply(self, X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        new_X = safe_copy(X)
        new_y = y
        for col in new_X.columns:
            try:
                new_X[col] = pd.to_numeric(new_X[col], )
            except ValueError as ignore:
                LOGGER.warning('Potential dtype issue: {}', ignore)
        return new_X, new_y