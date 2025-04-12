import pandas as pd
from cheutils.project_tree import save_excel
from cheutils.properties_util import AppProperties
from cheutils.loggers import LoguruWrapper
from cheutils.interceptor.pipelineInterceptor import PipelineInterceptor
from cheutils.data_properties import DataPropertiesHandler
from typing import cast

LOGGER = LoguruWrapper().get_logger()

def feature_selector(selector: str, estimator, passthrough: bool=False, ):
    """
    Meta-transformer for selecting features based on recursive feature selection, select from model, or any other equivalent class.
    @see https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python?noredirect=1&lq=1
    """
    assert selector is not None, 'A valid configured feature selector option must be provided'
    __data_handler: DataPropertiesHandler = cast(DataPropertiesHandler, AppProperties().get_subscriber('data_handler'))
    tf_base_class, tf_params = __data_handler.get_feat_selectors(estimator).get(selector) # a tuple (trans_class, trans_params)
    do_passthrough = passthrough | __data_handler.get_feat_sel_passthrough()
    override_sel = __data_handler.get_feat_sel_override()
    override_with_cols = __data_handler.get_feat_sel_selected()
    class FeatureSelector(tf_base_class):
        """
        Returns features based on ranking with recursive feature elimination.
        """
        def __init__(self, estimator=None, passthrough: bool=False, override_sel: bool=False,
                     override_cols: list=None, random_state: int=100, **kwargs):
            self.random_state = random_state
            self.estimator = estimator
            super().__init__(self.estimator, ** kwargs)
            self.target = None
            self.selected_cols = None
            self.passthrough = passthrough
            self.override_sel = override_sel
            self.override_cols = override_cols
            self.fitted = False
            self.selector = None

        def fit(self, X, y=None, **fit_params):
            if self.passthrough:
                self.selected_cols = list(X.columns)
            if self.fitted and not self.passthrough:
                return self
            LOGGER.debug('FeatureSelector: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            self.target = y  # possibly passed in chain
            super().fit(X, y, **fit_params)
            self.fitted = True
            return self

        def transform(self, X, y=None, **fit_params):
            LOGGER.debug('FeatureSelector: Transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            new_X = self.__do_transform(X, y=None)
            LOGGER.debug('FeatureSelector: Transformed dataset, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
            return new_X

        def fit_transform(self, X, y=None, **fit_params):
            LOGGER.debug('FeatureSelector: Fit-transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
            self.fit(X, y, **fit_params)
            self.target = y
            new_X = self.__do_transform(X, y, **fit_params)
            LOGGER.debug('FeatureSelector: Fit-transformed dataset, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_cols)
            return new_X

        def __do_transform(self, X, y=None, **fit_params):
            if self.selected_cols is None and not self.passthrough:
                if self.override_sel:
                    self.selected_cols = self.override_cols
                    new_X = X[self.selected_cols]
                if self.selected_cols is None:
                    if y is None:
                        transformed_X = super().transform(X)
                    else:
                        transformed_X = super().fit_transform(X, y, **fit_params)
                    self.selected_cols = list(X.columns[self.get_support()])
                    new_X = pd.DataFrame(transformed_X, columns=self.selected_cols)
                    self.__save_importances()
                else:
                    new_X = X
            else:
                new_X = pd.DataFrame(X, columns=self.selected_cols)
            return new_X

        def get_selected_features(self):
            """
            Return the selected features or column labels.
            :return:
            """
            return self.selected_cols

        def get_target(self):
            return self.target

        def __save_importances(self):
            try:
                importances = None
                try:
                    importances = self.estimator_.coef_
                except AttributeError:
                    importances = self.estimator_.feature_importances_
                if importances is not None:
                    importances = pd.Series(importances[:len(self.selected_cols)], index=self.selected_cols, name='importance')
                    save_excel(importances, file_name='feature_importances.xlsx', tag_label=selector, index=True)
            except Exception as ignore:
                LOGGER.warning('Could not save feature importances: {}', ignore)

    tf_instance = None
    try:
        tf_instance = FeatureSelector(estimator=estimator, passthrough=do_passthrough,
                                      override_sel=override_sel, override_cols=override_with_cols,
                                      **tf_params, )
        tf_instance.selector = selector
    except TypeError as err:
        LOGGER.error('Problem encountered instantiating feature selection transformer: {}, {}', selector, err)
    return tf_instance

class FeatureSelectionInterceptor(PipelineInterceptor):
    def __init__(self, selected_features: list):
        super().__init__()
        assert selected_features is not None and not (not selected_features), 'Valid selected features list required'
        self.selected_features = selected_features

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> (pd.DataFrame, pd.Series):
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('FeatureSelectionInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        new_X = X[self.selected_features]
        new_y = y
        LOGGER.debug('FeatureSelectionInterceptor: dataset out, shape = {}, {}\nFeatures selected:\n{}', new_X.shape, y.shape if y is not None else None, self.selected_features)
        return new_X, new_y

