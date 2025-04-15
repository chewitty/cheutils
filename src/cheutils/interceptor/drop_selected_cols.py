import pandas as pd
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor

LOGGER = LoguruWrapper().get_logger()

class DropSelectedColsInterceptor(PipelineInterceptor):
    def __init__(self, selected_cols: list, **kwargs):
        super().__init__(**kwargs)
        assert selected_cols is not None and not (not selected_cols), 'Valid selected features/columns required'
        self.selected_cols = selected_cols

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('DropSelectedColsInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        desired_cols = list(X.columns)
        for col in self.selected_cols:
            if col in desired_cols:
                desired_cols.remove(col)
        new_X = X[desired_cols]
        new_y = y
        LOGGER.debug('DropSelectedColsInterceptor: dataset out, shape = {}, {}\nFeatures dropped:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
        return new_X, new_y