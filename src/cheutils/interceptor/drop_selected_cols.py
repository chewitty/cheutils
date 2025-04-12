import pandas as pd
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor

LOGGER = LoguruWrapper().get_logger()

class DropSelectedColsInterceptor(PipelineInterceptor):
    def __init__(self, selected_cols: list):
        super().__init__()
        assert selected_cols is not None and not (not selected_cols), 'Valid selected features/columns required'
        self.selected_cols = selected_cols

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('DropSelectedColsInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X.drop(columns=self.selected_cols) if self.selected_cols is not None and not (not self.selected_cols) else X
        new_y = y
        LOGGER.debug('DropSelectedColsInterceptor: dataset out, shape = {}, {}\nFeatures dropped:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
        return new_X, new_y