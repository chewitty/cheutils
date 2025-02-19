import numpy as np

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
    diffs = np.log(y_true + 1) - np.log(y_pred + 1)
    squares = np.power(diffs, 2)
    err = np.sqrt(np.mean(squares))
    return err