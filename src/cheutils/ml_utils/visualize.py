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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.tree import plot_tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from cheutils.project_tree import save_current_fig
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

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

def plot_confusion_matrix(y_actual, y_pred, class_labels: list, title: str=None, svg_file_name: str=None,):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    conf_matrix = confusion_matrix(y_actual, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    # set the plot title using the axes object
    plot_title = title if title is not None and not (not title) else 'Confusion Matrix'
    ax.set(title=plot_title)
    disp.plot(ax=ax, values_format='.0f')
    ax.grid(False)
    if svg_file_name is not None and not (not svg_file_name):
        save_current_fig(svg_file_name)
    plt.show()

def print_classification_report(y_actual, y_pred):
    assert y_actual is not None, 'A valid y_actual is required'
    assert y_pred is not None, 'A valid y_pred is required'
    LOGGER.info('\n{}', classification_report(y_actual, y_pred))

def plot_no_skill_line(y_test, label: str = 'No Skill', linestyle: str = '--', color: str = 'blue',
                       show_axes: bool = False, flush: bool=False):
    """
    Plot the no skill line for the precision-recall plot.
    :param y_test:
    :param label:
    :param linestyle:
    :param color:
    :param show_axes:
    :param flush: call plt.show() - default is False
    :return:
    """
    # A no-skill model is represented by a horizontal line with a precision that is the ratio of positive
    # examples in the dataset (e.g. TP / (TP + TN))
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    aug_label = label + '={0:0.2f}'.format(no_skill)
    plt.plot([0, 1], [no_skill, no_skill], linestyle=linestyle, label=aug_label, color=color)
    # axis labels
    if show_axes:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    # outside: expected to call plt.show() after all the plotting is done
    if flush:
        plt.show()

def plot_precision_recall(y_test, pred_probas, pos_label=None, label: str = 'Model', marker: str = '.',
                          color: str = 'green', show_axes: bool = True,
                          no_skill_line: bool = True, opt_thres: bool = True, flush: bool=False, **kwargs):
    """
    Plot the precision-recall curve or plot.
    :param y_test:
    :param pred_probas
    :param pos_label:
    :param label:
    :param linestyle:
    :param color:
    :param show_axes:
    :param no_skill_line:
    :param ax:
    :param flush: call plt.show() - default is False
    :param kwargs: can pass additional plot parameters: e.g., curve_type='train' to ensure (Train) is include in title
    or , show_title=False to turn off the title.
    :return:
    """
    pos_class_label = 1 if (pos_label is None) else pos_label
    mdl_pr, mdl_rc, thresholds = precision_recall_curve(y_test, pred_probas,
                                                        pos_label=pos_class_label)
    avg_prec = average_precision_score(y_test, pred_probas)
    if no_skill_line:
        plot_no_skill_line(y_test)
    aug_label = label + ': AP={0:0.2f}'.format(avg_prec)
    curve_type = ' (Train)' if ('curve_type' in kwargs) & ('train' == kwargs.get('curve_type')) else ' (Test)'
    no_title = True if ('no_title' in kwargs) and kwargs.get('no_title') else False
    if 'curve_type' in kwargs:
        del kwargs['curve_type']
    if 'no_title' in kwargs:
        del kwargs['no_title']
    plt.plot(mdl_rc, mdl_pr, marker=marker, label=aug_label, color=color, **kwargs)
    # plot the location of the optimal fscore
    # apply threshold tuning
    # convert to fscore
    fscore = (2 * mdl_pr * mdl_rc) / (mdl_pr + mdl_rc)
    fscore = np.nan_to_num(fscore, nan=0)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, precision = %.2f' % (thresholds[ix], mdl_pr[ix]))
    print('Best Threshold=%f, recall = %.2f' % (thresholds[ix], mdl_rc[ix]))
    print('Best Threshold=%f, F-Score = %.2f' % (thresholds[ix], fscore[ix]))
    if opt_thres:
        plt.scatter(mdl_rc[ix], mdl_pr[ix], marker='v', color='red', label='Opt.F-Score=' + '{0:0.02f}'.format(fscore[ix]))
        #print('Dimensions', thresholds.shape, mdl_pr.shape, mdl_rc.shape)
        #thres = np.pad(thresholds, (0, 1), 'constant', constant_values=(1))
        #pr_values = pd.DataFrame({'thres': thres, 'precision': mdl_pr, 'recall': mdl_rc}, columns=['thres', 'precision', 'recall'])
        #pr_values.to_excel('precision_recall_thres.xlsx')
    # axis labels
    if show_axes:
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    if not no_title:
        plt.title('Precision-Recall' + curve_type)
    plt.legend(loc='best')
    # outside: expected to call plt.show() after all the plotting is done
    if flush:
        plt.show()
    # return the optimal values
    return mdl_pr[ix], mdl_rc[ix], fscore[ix]

def plot_precision_recall_by_threshold(y_test, pred_probas, pos_label=None, label: str = 'Model', marker: str = '.',
                          color_p: str = 'blue', color_r: str = 'green', color_f: str = 'gold', show_axes: bool = True,
                                       desired_thres: dict={'precision': 0.80, 'recall': 0.40}, opt_thres: bool = True,
                                       flush: bool=False, **kwargs):
    """
    Plot the trade-off between precision and recall as a function of the decision thresholds.
    :param y_test:
    :param pred_probas:
    :param pos_label:
    :param label:
    :param marker:
    :param color_p:
    :param color_r:
    :param color_f:
    :param show_axes:
    :param desired_thres: the desired minimum scores for precision and recall
    :param opt_thres:
    :param flush: call plt.show() - default is False
    :param kwargs:
    :return:
    """
    pos_class_label = 1 if (pos_label is None) else pos_label
    mdl_pr, mdl_rc, thresholds = precision_recall_curve(y_test, pred_probas,
                                                        pos_label=pos_class_label)
    curve_type = ' (' + label + ': Train)' if ('curve_type' in kwargs) & ('train' == kwargs.get('curve_type')) else ' (' + label + ': Test)'
    no_title = True if ('no_title' in kwargs) and kwargs.get('no_title') else False
    if 'curve_type' in kwargs:
        del kwargs['curve_type']
    if 'no_title' in kwargs:
        del kwargs['no_title']
    # convert to fscore
    fscore = (2 * mdl_pr * mdl_rc) / (mdl_pr + mdl_rc)
    fscore = np.nan_to_num(fscore, nan=0)
    # locate the index of the largest f score
    ix1 = np.argmax(fscore)
    # determine threshold at which the optimal balance of precision and recall could be found given
    # the desired minimums specified
    pr_options = (mdl_pr >= desired_thres.get('precision')) & (mdl_pr < 1.0)
    rc_options = (mdl_rc >= desired_thres.get('recall')) & (mdl_rc < 1.0)
    matching_options = pr_options & rc_options
    ix2 = ix1  # initialize
    if matching_options.any():
        best_pr = mdl_pr[matching_options].max()
        best_rc = mdl_rc[matching_options].max()
        ix2 = np.where(mdl_pr == best_pr)[0][0] if (best_pr > best_rc) else np.where(mdl_rc == best_rc)[0][0]
    else:
        print('No matching optimal threshold found beyond optimal F-Score')
    # plot the optimal fscore threshold line
    if opt_thres:
        plt.plot(thresholds, mdl_pr[:-1], color_p,
                 label='Precision = [' + '{0:0.02f}, {1:0.02f}]'.format(mdl_pr[ix1], mdl_pr[ix2]))
        plt.plot(thresholds, mdl_rc[:-1], color_r,
                 label='Recall = [' + '{0:0.02f}, {1:0.02f}]'.format(mdl_rc[ix1], mdl_rc[ix2]))
        plt.plot(thresholds, fscore[:-1], color_f,
                 label='F-score = [' + '{0:0.02f}, {1:0.02f}]'.format(fscore[ix1], fscore[ix2]))
        aug_label0 = 'Def.Thres. 0 = {0:0.2f}'.format(0.5)
        plt.axvline(x=0.5, linestyle='--', label=aug_label0, color='grey')
        aug_label1 = 'Opt.Thres. 1 = {0:0.2f}'.format(thresholds[ix1])
        plt.axvline(x=thresholds[ix1], linestyle='--', label=aug_label1, color='red')
        aug_label2 = 'Opt.Thres. 2 = {0:0.2f}'.format(thresholds[ix2])
        plt.axvline(x=thresholds[ix2], linestyle='--', label=aug_label2, color='brown')
    # axis labels
    if show_axes:
        plt.ylabel("Precision/Recall/F-Score")
        plt.xlabel("Decision Threshold")
    if not no_title:
        plt.title('Precision/Recall/F-Score vs Decision Threshold' + curve_type)
    plt.legend(loc='best')
    # outside: expected to call plt.show() after all the plotting is done
    if flush:
        plt.show()
    # return the second optimal values
    return thresholds[ix1], thresholds[ix2], mdl_pr[ix2], mdl_rc[ix2], fscore[ix2]

def plot_decision_tree(estimator, feature_names: list, class_labels: list, font_size: int=8, proportion: bool=True,
                       rounded: bool=True, filled: bool=True, svg_file_name: str=None, **kwargs):
    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    plot_tree(estimator, feature_names=feature_names, class_names=class_labels, fontsize=font_size,
              proportion=proportion, rounded=rounded, filled=filled, ax=ax)
    if svg_file_name is not None and not (not svg_file_name):
        save_current_fig(svg_file_name)
    plt.show()

def __label_pie_wedges(num_rows, allvalues):
    absolute = int(np.round((num_rows / 100. * np.sum(allvalues)), 0))
    return "{:.0f}%\n(N={:d})".format(num_rows, absolute)
