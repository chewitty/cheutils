import pandas as pd
import numpy as np
import geolib.geohash as gh
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.common_utils import safe_copy
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

class GeohashAugmenter(BaseEstimator, TransformerMixin):
    """
    Transforms latitude-longitude point to a geohashed fixed neighborhood.
    """
    def __init__(self, lat_col: str, long_col: str, to_col: str, drop_geo_cols: bool=True,
                 precision: int=6, smoothing: float=5.0, min_samples_leaf: int=10, **kwargs):
        """
        Transforms latitude-longitude point to a geohashed fixed neighborhood.
        :param lat_col: the column labels for desired latitude column
        :type lat_col: str
        :param long_col: the column labels for desired longitude column
        :type long_col: str
        :param to_col: the new generated column label for the geohashed fixed neighborhood
        :param drop_geo_cols: drops the latitude and longitude columns
        :param precision: geohash precision - default is 6
        :param smoothing: smoothing effect to balance categorical average vs prior - higher value means stronger regularization.
        :param min_samples_leaf: used for regularization the weighted average between category mean and global mean is taken
        :param kwargs:
        :type kwargs:
        """
        assert lat_col is not None and not (not lat_col), 'A valid column label is expected for latitude column'
        assert long_col is not None and not (not long_col), 'A valid column label is expected for longitude'
        assert to_col is not None and not (not to_col), 'A valid column label is expected for the generated geohashed fixed neighborhood'
        super().__init__(**kwargs)
        self.lat_col = lat_col
        self.long_col = long_col
        self.to_col = to_col
        self.drop_geo_cols = drop_geo_cols
        self.precision = precision
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.target_encoder: TargetEncoder
        self.fitted = False

    def fit(self, X, y=None):
        LOGGER.debug('GeohashAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        # fit the encoder on the training data
        self.__do_fit(X, y, )
        LOGGER.debug('GeohashAugmenter: Fitted dataset, out shape = {}, {}', X.shape, y.shape if y is not None else None)
        return self

    def transform(self, X, y=None):
        LOGGER.debug('GeohashAugmenter: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = self.__do_transform(X, y, )
        LOGGER.debug('GeohashAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('GeohashAugmenter: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.__do_fit(X, y, )
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('GeohashAugmenter: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        new_X = self.__generate_geohashes(X, **fit_params)
        if y is not None:
            new_X = self.target_encoder.fit_transform(new_X, y,)
        else:
            new_X = self.target_encoder.transform(new_X,)
        #feature_names = self.target_encoder.get_feature_names_out()
        #new_X = pd.DataFrame(new_X, columns=feature_names)
        if self.drop_geo_cols:
            new_X.drop(columns=[self.lat_col, self.long_col], inplace=True)
        return new_X

    def __generate_geohashes(self, X, **fit_params):
        new_X = safe_copy(X)
        # notes: precision of 5 translates to ≤ 4.89km × 4.89km; 6 translates to ≤ 1.22km × 0.61km; 7 translates to ≤ 153m × 153m
        new_X[self.to_col] = new_X.transform(lambda x: gh.encode(x[self.lat_col], x[self.long_col], precision=self.precision), axis=1)
        return new_X

    def __do_fit(self, X, y=None, **fit_params):
        if not self.fitted:
            new_X = self.__generate_geohashes(X, **fit_params)
            # generate expected values based on category aggregates
            self.target_encoder = TargetEncoder(cols=[self.to_col], return_df=True,
                                                smoothing=self.smoothing, min_samples_leaf=self.min_samples_leaf, )
            # fit the encoder
            new_y = safe_copy(y)
            self.target_encoder.fit(new_X, new_y)
            self.fitted = True
        return self