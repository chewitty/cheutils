import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    """
    The Root Mean Squared Logarithmic Error (RMSLE) evaluation metric.
    :param y_true: True values
    :type y_true:
    :param y_pred: Predicted values
    :type y_pred:
    :return: Root Mean Squared Logarithmic Error (RMSLE) as a float.
    :rtype:
    """
    err = np.sqrt(mean_squared_log_error(y_true.values if isinstance(y_true, pd.Series) else y_true,
                                         y_pred.values if isinstance(y_pred, pd.Series) else y_pred))
    return err