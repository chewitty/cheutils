import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import cast
from cheutils import ModelProperties, AppProperties
from cheutils.common_utils import safe_copy
from cheutils.sqlite_util import get_promising_interactions_from_sqlite_db, save_promising_interactions_to_sqlite_db
from cheutils.interceptor import PipelineInterceptor
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

def parse_promising_interactions(promising_interactions: list) -> (list, list):
    """
    Creates left and right lists of corresponding feature interactions from list of interaction tuples
    :param promising_interactions:
    :type promising_interactions:
    :return:
    :rtype:
    """
    assert promising_interactions is not None, 'Promising interactions must not be None'
    quali_left, quali_right = [[*quali_feat] for quali_feat in zip(*promising_interactions)]
    return quali_left, quali_right

def extract_interactions(selected_feats: list, separator: str= '_with_') -> (list, list):
    """
    Creates a tuple of lists of left and right feature interactions
    :param selected_feats: list of features, to extract qualifying left/right feature interactions, if name contains the separator
    :type selected_feats: list
    :param separator:
    :type separator:
    :return:
    :rtype:
    """
    assert selected_feats is not None and not (not selected_feats), 'Valid list of tuples containing left, right feature interactions must be provided'
    qualify_feats = [tuple(quali_feat.split(separator)) for quali_feat in selected_feats if separator in quali_feat]
    if qualify_feats is not None and not (not qualify_feats):
        quali_left, quali_right = parse_promising_interactions(qualify_feats)
    else:
        quali_left, quali_right = [], []
    return quali_left, quali_right

def augment_with_interactions(X: pd.DataFrame, quali_left_cols: list, quali_right_cols: list, separator: str='_with_', ) -> pd.DataFrame:
    new_X = X
    interaction_srs = [new_X]
    interaction_feats = []
    for c1, c2 in zip(quali_left_cols, quali_right_cols):
        n = f'{c1}{separator}{c2}'
        new_sr = new_X[c1] * new_X[c2]
        new_sr.name = n
        interaction_srs.append(new_sr)
        interaction_feats.append((c1, c2))
    if len(interaction_srs) > 1:
        new_X = pd.concat(interaction_srs, axis=1)
    LOGGER.debug('\nInteraction features:\n{}', interaction_feats)
    return new_X

class PromisingInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, candidate_feats: list, scoring, transform_pipeline: Pipeline=None,
                 cv=3, error_margin: float=0.01, selected_feats: list=None, separator: str='_with_', tb_name: str=None, n_jobs: int=-1, ):
        """
        Create an instance of PromisingInteractions - which can be used to run a standalone transformation to identify feature interactions.
        >>> from cheutils import PromisingInteractions
        >>> model_option = MAIN_ESTIMATOR
        >>> model_prefix = model_option
        >>> baseline_model = get_estimator(model_option=model_option)
        >>> baseline_pipeline_steps = standard_pipeline_steps.copy()
        >>> transform_pipeline = Pipeline(steps=baseline_pipeline_steps, verbose=True)
        >>> baseline_pipeline_steps.append((model_prefix, baseline_model))
        >>> baseline_pipeline = Pipeline(steps=baseline_pipeline_steps, verbose=True)
        >>> candidate_feats = APP_PROPS.get_list('model.feature.interaction.candidates')
        >>> # if no pipeline the transform_pipeline should be ignored
        >>> interaction_tf = PromisingInteractions(baseline_pipeline, candidate_feats, transform_pipeline=transform_pipeline, scoring=SCORING, cv=3, n_jobs=-1, )
        >>> # the transformed dataframe includes identified interactions
        >>> X_train_transformed = interaction_tf.fit_transform(X_train, y_train)
        :param estimator: the estimator, which can be a pipeline
        :type estimator:
        :param candidate_feats: the candidate interaction features worth investigating if they would likely improve model performance if included.
        :type candidate_feats:
        :param scoring: the scoring - e.g., make_scorer(rmsle, greater_is_better=False)
        :type scoring:
        :param transform_pipeline: the transform pipeline - usually, the same pipeline without the last estimator (classifier or regressor)
        :type transform_pipeline:
        :param cv: number of folds or splitting strategy
        :type cv:
        :param error_margin: error margin tolerable (betweeon 0 and 0.10, recommended) on the calculated baseline score
        :type error_margin:
        :param selected_feats: important features previously identified through feature selection
        :type selected_feats:
        :param separator: feature interaction str separator - defaults to '_with_'
        :param tb_name: any optional underlying sqlite table name
        :type tb_name:
        :param n_jobs: optional n_jobs
        :type n_jobs:
        """
        assert estimator is not None, 'Valid estimator or pipeline required'
        assert candidate_feats is not None and not (not candidate_feats), 'Valid candidate features required'
        assert scoring is not None, 'A valid scoring function required'
        super().__init__()
        self.estimator = estimator
        self.candidate_feats = candidate_feats
        self.scoring = scoring
        self.transform_pipeline = estimator if transform_pipeline is None else transform_pipeline # optionally provided when pipeline of transformers is terminated with an estimator
        self.cv = cv
        self.error_margin = error_margin
        self.selected_feats = selected_feats # any a priori features from a feature selection process
        self.tb_name = tb_name
        self.n_jobs = n_jobs
        self.baseline_score = None
        self.fitted = False
        self.promising_interactions = None
        self.separator = separator
        self.transformed_X = None

    def fit(self, X, y=None, **fit_params):
        if self.fitted:
            return self
        LOGGER.debug('PromisingInteractions: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
        self.promising_interactions = get_promising_interactions_from_sqlite_db(tb_name=self.tb_name, model_prefix=__model_handler.get_model_option()) if self.tb_name is not None else get_promising_interactions_from_sqlite_db(model_prefix=__model_handler.get_model_option())
        # if there is a prevailing set of interaction features that was previously cached, then use those for efficiency
        if self.promising_interactions is not None and not (not self.promising_interactions):
            if self.transformed_X is None:
                self.transformed_X = self.transform_pipeline.fit_transform(X, y, **fit_params) if isinstance(self.transform_pipeline, Pipeline) else X
            self.fitted = True
            return self
        # otherwise, re-generate those and save in sqlite for future reuse.
        cv_score = cross_val_score(self.estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs,
                                   error_score='raise',
                                   verbose=10, )
        LOGGER.debug('PromisingInteractions: Baseline CV Score = \n{}, \n{}, \n{}', -cv_score, -cv_score.mean(), -cv_score.std())
        self.baseline_score = abs(cv_score.mean())
        self.transformed_X = self.transform_pipeline.fit_transform(X, y, **fit_params) if isinstance(self.transform_pipeline, Pipeline) else X
        __model_handler: ModelProperties = cast(ModelProperties, AppProperties().get_subscriber('model_handler'))
        promising_feats = []
        train = safe_copy(X)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        ct = ColumnTransformer([('poly', poly, self.candidate_feats)], remainder='drop',
                               force_int_remainder_cols=True, verbose=True, n_jobs=__model_handler.get_n_jobs())
        train_polys = ct.fit_transform(self.transformed_X)
        feature_names_out = ct.get_feature_names_out().tolist()
        feature_names = [feature_name.split('__')[-1].replace('^', '_pow_').replace(' ', '_with_') for feature_name in feature_names_out]
        feature_interactions = [feature_name for feature_name in feature_names if feature_name not in self.candidate_feats]
        train_interactions = pd.DataFrame(train_polys, columns=feature_names, )[feature_interactions]
        for feat_interaction in feature_interactions:
            train[feat_interaction] = train_interactions[feat_interaction].values
            cv_score = cross_val_score(self.estimator, train, y, scoring=__model_handler.get_cross_val_scoring(),
                                       cv=__model_handler.get_cross_val_num_folds(),
                                       n_jobs=__model_handler.get_n_jobs(), error_score='raise')
            mean_score = -np.nanmean(cv_score) if is_classifier(self.estimator) else abs(np.nanmean(cv_score))
            LOGGER.debug('Mean score = {}, [{}]\n', round(mean_score, 4), feat_interaction)
            test_val = self.baseline_score * (1 + self.error_margin) if is_classifier(
                    self.estimator) else self.baseline_score * (1 - self.error_margin)
            if mean_score >= test_val if is_classifier(self.estimator) else mean_score < test_val:
                promising_feats.append(feat_interaction)
            train.drop(columns=[feat_interaction], inplace=True)
        del train
        del train_interactions
        # cache the promising interaction features to SQLite
        if len(promising_feats) > 0:
            if self.tb_name is not None:
                save_promising_interactions_to_sqlite_db(promising_interactions=promising_feats, tb_name=self.tb_name, model_prefix=__model_handler.get_model_option(), )
            else:
                save_promising_interactions_to_sqlite_db(promising_interactions=promising_feats, model_prefix=__model_handler.get_model_option(), )
        self.promising_interactions = get_promising_interactions_from_sqlite_db(tb_name=self.tb_name, model_prefix=__model_handler.get_model_option()) if self.tb_name is not None else get_promising_interactions_from_sqlite_db(model_prefix=__model_handler.get_model_option())
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('PromisingInteractions: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        if self.transformed_X is not None:
            new_X = self.transformed_X
        else:
            new_X = self.transform_pipeline.transform(X, **fit_params) if isinstance(self.transform_pipeline, Pipeline) else X
        if self.promising_interactions is not None and not (not self.promising_interactions):
            # add promising features to dataframe
            quali_left, quali_right = parse_promising_interactions(self.promising_interactions)
            # if there was a previously found selected features from a feature selection process provided
            # then limit the promising interactions by the qualifying interactions included in those
            if self.selected_feats is not None and not (not self.selected_feats):
                quali_left, quali_right = extract_interactions(self.selected_feats, separator=self.separator)
            new_X = augment_with_interactions(new_X, quali_left, quali_right, separator=self.separator)
        LOGGER.debug('PromisingInteractions: Transformed dataset, shape = {}', new_X.shape)
        return new_X

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

    def apply(self, X: pd.DataFrame, y: pd.Series, **params) -> pd.DataFrame:
        assert X is not None, 'Valid dataframe with data required'
        LOGGER.debug('InteractionFeaturesInterceptor: dataset in, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.interaction_feats = []
        quali_left_cols, quali_right_cols = self.left_cols, self.right_cols
        if self.selected_feats is not None and not (not self.selected_feats):
            quali_left_cols, quali_right_cols = extract_interactions(self.selected_feats, separator=self.separator)
        new_X = augment_with_interactions(X, quali_left_cols, quali_right_cols, separator=self.separator)
        LOGGER.debug('InteractionFeaturesInterceptor: dataset out, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X
