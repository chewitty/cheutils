import pandas as pd
from cheutils.interceptor import PipelineInterceptor
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class InteractionFeaturesInterceptor(PipelineInterceptor):
    def __init__(self, left_cols: list, right_cols: list, selected_feats: list=None, **kwargs):
        """
        Creates an instance of the interaction features interceptor, that could be useful for a data pipeline. Each
        interaction involves two features - a left and right feature.
        :param left_cols: left features
        :type left_cols: list
        :param right_cols: right features - must be same length as left_cols
        :type right_cols: list
        :param selected_feats: a priori features selected by an a priori feature selection process (i.e., known beforehand); used to limit qualifying interactions.
        :type selected_feats: list
        :param kwargs:
        :type kwargs:
        """
        assert left_cols is not None and not(not left_cols), 'Valid left columns/features must be provided'
        assert right_cols is not None and not (not right_cols), 'Valid right columns/features must be provided'
        assert len(left_cols) == len(right_cols), 'Left and right columns must have same length'
        super().__init__(**kwargs)
        self.left_cols = left_cols
        self.right_cols = right_cols
        self.interaction_feats = None
        self.selected_feats = selected_feats
        self.separator = '_with_'

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('InteractionFeaturesInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.interaction_feats = []
        new_X = X
        def parse_selected():
            qualify_feats = [tuple(quali_feat.split(self.separator)) for quali_feat in self.selected_feats if self.separator in quali_feat]
            quali_left, quali_right = [[*quali_feat] for quali_feat in zip(*qualify_feats)]
            return quali_left, quali_right
        quali_left_cols, quali_right_cols = self.left_cols, self.right_cols
        if self.selected_feats is not None and not (not self.selected_feats):
            quali_left_cols, quali_right_cols = parse_selected()
        interaction_srs = [new_X]
        for c1, c2 in zip(quali_left_cols, quali_right_cols):
            n = f'{c1}{self.separator}{c2}'
            new_sr = new_X[c1] * new_X[c2]
            new_sr.name = n
            interaction_srs.append(new_sr)
            self.interaction_feats.append(n)
        new_X = pd.concat(interaction_srs, axis=1)
        LOGGER.debug('InteractionFeaturesInterceptor: dataset out, shape = {}, {}\nInteraction features:\n{}', new_X.shape, y.shape if y is not None else None, self.interaction_feats)
        return new_X, y