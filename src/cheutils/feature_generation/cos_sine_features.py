import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class PeriodicFeaturesAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, periods: list, **kwargs):
        """
        Create a new PeriodicFeaturesAugmenter instance.
        :param rel_cols: the list of columns with periodic features to encode using a sine and cosine transformation
        with the corresponding matching periods
        :param periods: list of appropriate period values, matching the relevant periodic feature columns specified
        """
        assert rel_cols is not None or not (not rel_cols), 'Valid numeric periodic feature columns must be specified'
        assert periods is not None or not (not periods), 'Valid periods for the periodic features must be specified'
        super().__init__(**kwargs)
        self.rel_cols = rel_cols
        self.periods = periods
        self.sine_transformers = {}
        self.cosine_transformers = {}
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('PeriodicFeaturesAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.__sine_transformers()
        self.__cosine_transformers()
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('PeriodicFeaturesAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('PeriodicFeaturesAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def __sine_transformers(self):
        for rel_col, period in zip(self.rel_cols, self.periods):
            self.sine_transformers[rel_col] = FunctionTransformer(func=lambda x: np.sin(x / float(period) * 2 * np.pi), )

    def __cosine_transformers(self):
        for rel_col, period in zip(self.rel_cols, self.periods):
            self.cosine_transformers[rel_col] = FunctionTransformer(func=lambda x: np.cos(x / float(period) * 2 * np.pi), )

    def __do_transform(self, X, y=None, **fit_params):
        if not self.fitted:
            raise RuntimeError('You have to call fit on the transformer before')
        # apply sine and cosine transformations appropriately
        new_X = X
        for rel_col in self.rel_cols:
            sine_tf = self.sine_transformers.get(rel_col)
            input = new_X[[rel_col]]
            input.loc[:, rel_col] = input[rel_col].astype(float)
            if sine_tf is not None:
                new_X.loc[:, rel_col + '_sin'] = sine_tf.fit_transform(input)[rel_col]
            cose_tf = self.cosine_transformers.get(rel_col)
            if cose_tf is not None:
                new_X.loc[:, rel_col + '_cos'] = cose_tf.fit_transform(input)[rel_col]
        return new_X