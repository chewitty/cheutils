import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

def rmsle(y_true, y_pred):
    """
    The Root Mean Squared Logarithmic Error (RMSLE) evaluation metric. Note that, when using Root Mean Squared Logarithmic Error (RMSLE),
    it's crucial to log-transform the target variable before evaluating the model. RMSLE is essentially the RMSE of the
    log-transformed target, so minimizing RMSLE is the same as minimizing the RMSE of the logged target. This transformation
    helps to reduce the impact of large errors and makes the metric more sensitive to proportional differences.
    :param y_true: True values, assumed to be log-transformed
    :type y_true:
    :param y_pred: Predicted values, assumed to be log-transformed
    :type y_pred:
    :return: Root Mean Squared Logarithmic Error (RMSLE) as a float.
    :rtype:
    """
    y_true_in = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_in = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    y_true_in, y_pred_in = ensure_no_negatives(y_true_in, y_pred_in)
    err = root_mean_squared_log_error(y_true_in, y_pred_in)
    return err

def ensure_no_negatives(y_true, y_pred):
    # Ensure no negative values
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return y_true, y_pred