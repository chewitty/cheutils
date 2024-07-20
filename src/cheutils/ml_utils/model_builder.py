import numpy as np
from cheutils.decorator_debug import debug_func
from cheutils.decorator_timer import track_duration
from cheutils.ml_utils.pipeline_details import show_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


@track_duration(name='fit')
@debug_func(enable_debug=True, prefix='fit')
def fit(pipeline: Pipeline, X, y, **kwargs):
    """
    Fit the model based on the pipeline
    :param pipeline:
    :param X:
    :param y:
    :return:
    """
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    assert y is not None, "A valid y expected"
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    if name is not None:
        show_pipeline(pipeline, name=name, save_to_file=True)
    else:
        show_pipeline(pipeline)
    pipeline.fit(X, y, **kwargs)

@track_duration(name='predict')
@debug_func(enable_debug=True, prefix='predict')
def predict(pipeline: Pipeline, X):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    return np.round(pipeline.predict(X),0).astype(int)

def predict_proba(pipeline: Pipeline, X):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert X is not None, "A valid X expected"
    return pipeline.predict_proba(X)


def direct_predict(estimator, X):
    assert estimator is not None, "A valid estimator instance expected"
    assert X is not None, "A valid X expected"
    return estimator.predict(X)

def exclude_nulls(X, y):
    """
    Return dataset ready for predictions, scoring, and reporting - i.e., the prediction step does not need null values.
    :param X:
    :param y:
    :return:
    """
    assert X is not None, "A valid X expected"
    assert y is not None, "A valid y expected"
    # The prediction step does not need null values
    X_pred = X.copy(deep=True)
    y_pred = y.copy(deep=True)
    X_pred.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)
    null_rows = X_pred.isna().any(axis=1)
    X_pred.dropna(inplace=True)
    y_pred = y_pred[~null_rows]
    print("Shape of dataset available for predictions", X_pred.shape, y_pred.shape)
    return X_pred, y_pred

@track_duration(name='score')
@debug_func(enable_debug=True, prefix='score')
def score(y_true, y_pred, kind:str="mse"):
    assert y_true is not None, "A valid y_true expected"
    assert y_pred is not None, "A valid y_pred expected"
    if kind == "mse":
        mse_score = mean_squared_error(y_true, y_pred)
        return mse_score
    elif kind == "r2":
        r2score = r2_score(y_true, y_pred)
        return r2score
    else:
        raise ValueError("Score not yet implemented")
@track_duration(name='tune_model')
@debug_func(enable_debug=True, prefix='tune_model')
def tune_model(pipeline: Pipeline, X, y, params_grid: dict, cv, grid_search: bool = False,
               scoring: str = "neg_mean_squared_error", random_state=100, debug: bool = False, scan_params: bool=False, **kwargs):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert params_grid is not None, "A valid parameter grid (a dict) expected"
    assert (cv is not None), "A valid cv, either the number of folds or an instance of something like StratifiedKFold"
    n_iters = 1000
    n_jobs = -1
    search_cv = None
    if grid_search:
        if debug:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=2, error_score="raise",)
        else:
            search_cv = GridSearchCV(estimator=pipeline, param_grid=params_grid,
                                     scoring=scoring, cv=cv, n_jobs=n_jobs,)
    else:
        if debug:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state, verbose=2, error_score="raise",)
        else:
            search_cv = RandomizedSearchCV(estimator=pipeline, param_distributions=params_grid,
                                           scoring=scoring, cv=cv, n_iter=n_iters, n_jobs=n_jobs,
                                           random_state=random_state,)
    name = None
    if "name" in kwargs:
        name = kwargs.get("name")
        del kwargs["name"]
    if name is not None:
        show_pipeline(search_cv, name=name, save_to_file=True)
    else:
        show_pipeline(search_cv)
    search_cv.fit(X, y)
    if not scan_params:
        return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_
    else:
        return search_cv.cv_results_
@track_duration(name='cv')
@debug_func(enable_debug=True, prefix='cv')
def cross_val_model(pipeline: Pipeline, X, y, scoring, cv=5, **fit_params):
    assert pipeline is not None, "A valid pipeline instance expected"
    assert (cv is not None), "A valid cv, either the number of folds or an instance of something like StratifiedKFold"
    cv_scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, **fit_params)
    return cv_scores



