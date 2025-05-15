import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

def rmsle(y_true, y_pred):
    """
    The Root Mean Squared Logarithmic Error (RMSLE) evaluation metric. Note that, when using Root Mean Squared Logarithmic Error (RMSLE),
    it's crucial to log-transform the target variable before evaluating the model. RMSLE is essentially the RMSE of the
    log-transformed target, so minimizing RMSLE is the same as minimizing the RMSE of the logged target. This transformation
    helps to reduce the impact of large errors and makes the metric more sensitive to proportional differences.
    :param y_true: True values
    :type y_true:
    :param y_pred: Predicted values
    :type y_pred:
    :return: Root Mean Squared Logarithmic Error (RMSLE) as a float.
    :rtype:
    """
    y_true_in = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_in = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    err = np.sqrt(mean_squared_log_error(y_true_in, y_pred_in))
    #LOGGER.debug('\nRMSLE score = {}\n', err)
    return err

def nan_rmsle(y_true, y_pred):
    """
    The Root Mean Squared Logarithmic Error (RMSLE) evaluation metric where the y_pred is non-negative by capping at zero.
    :param y_true: True values
    :type y_true:
    :param y_pred: Predicted values
    :type y_pred:
    :return: Root Mean Squared Logarithmic Error (RMSLE) as a float.
    :rtype:
    """
    y_true_in = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_in = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    err = np.sqrt(mean_squared_log_error(y_true_in, np.maximum(0, y_pred_in)))
    #LOGGER.debug('\nRMSLE score = {}\n', err)
    return err