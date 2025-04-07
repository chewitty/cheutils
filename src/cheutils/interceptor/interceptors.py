import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor
from cheutils.interceptor.numeric_data import NumericDataInterceptor

LOGGER = LoguruWrapper().get_logger()

class DataPipelineInterceptor(BaseEstimator, TransformerMixin):
    def __init__(self, interceptors: list=None, apply_numeric: bool=False, ):
        """
        Create a new DataPipelineInterceptor instance.
        :param interceptors: the list of data pipeline interceptors to be applied in order
        :param apply_numeric: indicates if the numeric data interceptor should be applied as the final step
        """
        self.interceptors = interceptors if interceptors is not None else []
        self.apply_numeric = apply_numeric
        if self.apply_numeric:
            # add the numeric interceptor as last step as needed
            self.interceptors.append(NumericDataInterceptor())
        assert all(isinstance(n, PipelineInterceptor) for n in self.interceptors), 'Valid PipelineInterceptors expected'

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('DataPipelineInterceptor: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X, new_y = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('DataPipelineInterceptor: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('DataPipelineInterceptor: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X, new_y = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('DataPipelineInterceptor: Fit-transformed dataset, shape = {}, {}', new_X.shape, new_y.shape if new_y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params) -> (pd.DataFrame, pd.Series):
        """
        Apply the data pipeline interceptors in order, with the numeric interceptor as last step as needed.
        :param X: dataframe with data to transform
        :type X:
        :param y: series with target values - in most cases, these target values are untouched
        :type y:
        :param fit_params: any additional special parameters that may be required by the specific interceptor processing
        :type fit_params:
        :return: the transformed X and y (which may be untouched)
        :rtype: (pd.DataFrame, pd.Series)
        """
        new_X = X
        new_y = y
        for interceptor in self.interceptors:
            new_X, new_y = interceptor.apply(new_X, new_y, **fit_params)
        return new_X, new_y