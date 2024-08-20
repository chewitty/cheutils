"""
Set of plotting and visualization utilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PredictionErrorDisplay
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from cheutils.project_tree import save_current_fig

def plot_reg_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = None, save_to_file: str = None, **kwargs):
    """
    Plot the prediction error of a regression model given the true and predicted targets (actuals vs predicted.
    :param y_true: True target values
    :type y_true:
    :param y_pred: Predicted target values
    :type y_pred:
    :param title:
    :type title:
    :param save_to_file:
    :type save_to_file:
    :return:
    :rtype:
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    PredictionErrorDisplay.from_predictions(y_true, y_pred, kind='actual_vs_predicted',
                                            ax=ax, scatter_kwargs={'alpha': 0.5})
    # Add the score in the legend of each axis
    for name, score in __compute_score(y_true, y_pred).items():
        ax.plot([], [], ' ', label=f'{name}={score}')
    ax.legend(loc='best')
    ax.set_title('Scatter plot of Actuals vs Predicted' if title is None else title)
    plt.tight_layout()
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()


def plot_reg_residuals(y_true: pd.Series, y_pred: pd.Series, title: str = None, save_to_file: str = None, **kwargs):
    """
    Plot the prediction error of a regression model given the true and predicted targets (residuals vs predicted.
    The residuals are difference between observed and predicted values.
    :param y_true:
    :type y_true:
    :param y_pred:
    :type y_pred:
    :param title:
    :type title:
    :param save_to_file:
    :type save_to_file:
    :return:
    :rtype:
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    PredictionErrorDisplay.from_predictions(y_true, y_pred, kind='residual_vs_predicted',
                                            ax=ax, scatter_kwargs={'alpha': 0.5})
    # Add the score in the legend of each axis
    for name, score in __compute_score(y_true, y_pred).items():
        ax.plot([], [], ' ', label=f'{name}={score}')
    ax.legend(loc='best')
    ax.set_title('Residuals plot (i.e., Actual - Predicted)' if title is None else title)
    plt.tight_layout()
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()

def plot_reg_predictions_dist(y_true: pd.Series, y_pred: pd.Series, title: str = None, save_to_file: str = None, **kwargs):
    """
    Plot the prediction error of a regression model given the true and predicted targets (actuals vs predicted).
    The width of each violin represents the density of the data points; the white dot is the median,
    and the box represents the interquartile range (IQR), whereas, the whiskers are 1.5 times the IQR.
    Ideally the mass of each violin should be centered on the true value.
    as a violin plot.
    :param y_true: True target values
    :type y_true:
    :param y_pred: Predicted target values
    :type y_pred:
    :param title:
    :type title:
    :param save_to_file:
    :type save_to_file:
    :return:
    :rtype:
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x=y_true, y=y_pred, alpha=0.5)
    # Add the score in the legend of each axis
    for name, score in __compute_score(y_true, y_pred).items():
        ax.plot([], [], ' ', label=f'{name}={score}')
    ax.legend(loc='best')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Violin plot of Actuals vs Predicted' if title is None else title)
    plt.tight_layout()
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()

def plot_reg_residuals_dist(y_true: pd.Series, y_pred: pd.Series, title: str = None, save_to_file: str = None, **kwargs):
    """
    Plot a distribution of the prediction error of a regression model given the true and predicted targets (residuals vs predicted.
    The residuals are difference between observed and predicted values. A good model would have residuals that are normally
    distributed around 0. If the residuals are not centered around 0 or have a skewed distribution, it may indicate
    that the model is systematically overestimating or underestimating the target variable.
    :param y_true:
    :type y_true:
    :param y_pred:
    :type y_pred:
    :param title:
    :type title:
    :param save_to_file:
    :type save_to_file:
    :return:
    :rtype:
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=y_true-y_pred, ax=ax, alpha=0.5, **kwargs)
    # Add the score in the legend of each axis
    for name, score in __compute_score(y_true, y_pred).items():
        ax.plot([], [], ' ', label=f'{name}={score}')
    ax.legend(loc='best')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of residuals (i.e., Actual - Predicted)' if title is None else title)
    plt.tight_layout()
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()

def __compute_score(y_true: pd.Series, y_pred: pd):
    return {'R2': f'{r2_score(y_true, y_pred):.2f}', 'MSE': f'{abs(mean_squared_error(y_true, y_pred)):.2f}', }


def plot_pie(df: pd.DataFrame, data_col: str = 'rental_duration', label_col: str = 'index',
             title: str = 'Dataset Split', legend_title: str = 'Rental Duration', save_to_file: str = None, **kwargs):
    """
    Create a pie chart of the given data.
    :param df:
    :param data_col:
    :param label_col:
    :param title:
    :param legend_title:
    :param save_to_file:
    :return:
    """
    # only a binary classification case
    my_palette = {'1': 'bisque', '0': 'lightslategray'}
    if 'palette' in kwargs:
        my_palette = kwargs.get('palette')
    high_color = 'bisque' if (my_palette.get('1') is None) else my_palette.get('1')
    not_High_color = 'lightslategray' if (my_palette.get('0') is None) else my_palette.get('0')
    colors = (high_color, not_High_color)
    explode = (0.05, 0.0)
    wp = {'linewidth': 1, 'edgecolor': 'grey'}
    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(df[data_col], autopct=lambda x: __label_pie_wedges(x, df[data_col]),
                                      explode=explode, labels=df[label_col], shadow=False, colors=colors,
                                      startangle=90, wedgeprops=wp, textprops=dict(color='black'))
    ax.legend(wedges, df[label_col], title=legend_title, loc='center left', bbox_to_anchor=(1, 0, 0.2, 1))
    plt.setp(autotexts, size=10, weight='bold')
    ax.set_title(title)
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()


def plot_hyperparameter(metrics_df: pd.DataFrame, param_label: str = 'param', metric_label: str = 'scores',
                        save_to_file: str = None, **kwargs):
    """
    Plots a scatter plot showing the metric over a range of parameter values
    :param metrics_df: a DataFrame that has each hyperparameter combination and the resulting metric scores
    :param metric_label: column label specifying the columns of the dataframe containing the metric scores
    :param param_label: column label specifying the columns of the dataframe containing the metric scores
    :param save_to_file: file name to save the plot (default should be an SVG file name (i.e., ending in .svg)
    :return:
    :rtype:
    """
    assert metrics_df is not None, 'metric scores by range of parameter values should be provided'
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=metrics_df, x=param_label, y=metric_label)
    #plt.scatter(metrics_df[param_label], metrics_df[metric_label])
    plt.gca().set(xlabel='{}'.format(param_label), ylabel=metric_label,
                  title=metric_label.title() + ' for different {} values'.format(param_label))
    plt.tight_layout()
    if save_to_file is not None:
        save_current_fig(file_name=save_to_file)
    plt.show()


def __label_pie_wedges(num_rows, allvalues):
    absolute = int(np.round((num_rows / 100. * np.sum(allvalues)), 0))
    return "{:.0f}%\n(N={:d})".format(num_rows, absolute)
