import pandas as pd
from cheutils import safe_copy
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor

LOGGER = LoguruWrapper().get_logger()

class SelectedFeaturesInterceptor(PipelineInterceptor):
    def __init__(self, selected_features: list):
        super().__init__()
        assert selected_features is not None and not (not selected_features), 'Valid selected features list required'
        self.selected_features = selected_features

    def apply(self, X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('SelectedFeaturesInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X[self.selected_features]
        new_y = y
        LOGGER.debug('SelectedFeaturesInterceptor: dataset out, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_features)
        return new_X, new_y

