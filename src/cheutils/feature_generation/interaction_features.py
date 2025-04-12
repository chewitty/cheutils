import pandas as pd
from cheutils.interceptor import PipelineInterceptor
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class InteractionFeaturesInterceptor(PipelineInterceptor):
    def __init__(self, left_cols: list, right_cols: list, ):
        super().__init__()
        assert left_cols is not None and not(not left_cols), 'Valid left columns/features must be provided'
        assert right_cols is not None and not (not right_cols), 'Valid right columns/features must be provided'
        assert len(left_cols) == len(right_cols), 'Left and right columns must have same length'
        self.left_cols = left_cols
        self.right_cols = right_cols
        self.interaction_feats = []

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('InteractionFeaturesInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X
        for c1, c2 in zip(self.left_cols, self.right_cols):
            n = f'{c1}_with_{c2}'
            new_X.loc[:, n] = new_X[c1] * new_X[c2]
            self.interaction_feats.append(n)
        LOGGER.debug('InteractionFeaturesInterceptor: dataset out, shape = {}, {}\nInteraction features:\n{}', new_X.shape, y.shape if y is not None else None, self.interaction_feats)
        return new_X, y