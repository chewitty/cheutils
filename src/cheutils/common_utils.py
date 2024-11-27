import numpy as np
import copy
import pandas as pd
import re
import pingouin as pg
import datetime as dt
import inspect
from typing import Union, Any
from scipy.stats import iqr, mstats, trimboth
from scipy.stats.mstats import winsorize
from fast_ml import eda
from fast_ml.utilities import display_all
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

def summarize(df: pd.DataFrame, display: bool = True):
    """
    Generate useful summary - variables, datatype, number of unique values, sample of unique values, missing count, missing percent
    :param df: specified dataframe
    :param display: whether to display the dataframe
    :return: summary dataframe or None
    :rtype:
    """
    summary_df = eda.df_info(df)
    if display:
        display_all(summary_df)
    else:
        return summary_df

# Validation of all loaded data columns according to expectations
def validate_data(df, expectations):
    """
    Check all data types are consistent with expectations.
    :param df: data frame containing the data to check
    :param expectations: a dict containing the column-names as keys and their corresponding
    data types as values.
    :return: pandas Series containing pass/fail status for each column tested
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (expectations is not None) and (len(expectations.keys()) > 0), 'missing dict (at least one key)'
    # start the checks
    cols_to_check = expectations.keys()
    results = {col: True if df[col].dtype == expectations.get(col) else False for col in cols_to_check}
    results_sr = pd.Series(results, name='Status')
    return results_sr

def calc_prop_missing(df):
    """
    Calculate proportion of missing values in each column of the DataFrame.
    :param df: DataFrame of interest
    :return: Series with missing value porportions
    """
    assert df is not None, 'A valid DataFrame expected as input'
    prop_sr = df.isna().mean().sort_values(ascending=False)
    prop_sr.rename('prop', inplace=True)
    return prop_sr

def find_numeric(rel_str):
    """
    Return any numeric characters found - i.e., numbers from 0-9
    :param rel_str: string of interest
    :return: numeric digits or blank
    """
    assert rel_str is not None, 'A valid string expected as input'
    return re.findall('\d', rel_str)

def cat_to_numeric(data_in, drop_first: bool = False):
    """
    Return a DataFrame/Series with categorical columns encoded to integers.
    :param data_in: the DataFrame/Series of interest
    :param drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level.
    :return: DataFrame/Series with all categorical columns changed to integers and column names prefixed with 'cat'
    """
    assert data_in is not None, 'A valid DataFrame/Series expected as input'
    result_out = pd.get_dummies(data_in, prefix='', prefix_sep='', dtype=np.int32, drop_first=drop_first)
    return result_out

def apply_annova(df: pd.DataFrame, rel_col: list, between_col: str, alpha: float = 0.05):
    """
    Retrun the results of running annova on the relevant continuous variable columns to determine
    differences between groups specified by the values of the between column.
    :param df:
    :param rel_col:
    :param between_col:
    :param alpha: the significance level (defaults to 0.05)
    :return:
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (rel_col is not None) or not (not rel_col), 'A valid list of continuous value columns expected as input'
    assert between_col is not None, 'A valid categorical column whose values are the groups expected as input'
    annova_res = []
    for col in rel_col:
        cur_res = pg.welch_anova(data=df, dv=col, between=between_col)
        is_sig = cur_res['p-unc'].values[0] < alpha
        annova_res.append({'By': cur_res['Source'].values[0], 'rel_col': col, 'p-value': cur_res['p-unc'].values[0], 'sig': is_sig})
    annova_res_df = pd.DataFrame(annova_res, columns=['By', 'rel_col', 'p-value', 'sig'])
    return annova_res_df

def get_date(data_row, date_cols: list=None):
    """
    Gets a suitable datetime from the specified columns - and fixes incorrect leap year date
    :param data_row: the row of data
    :type data_row: the relevant date columns
    :param date_cols:
    :return:
    :rtype:
    """
    if date_cols is None:
        return np.nan
    row_in = data_row.copy()
    def is_leap_year(year):
        # divided by 100 means century year (ending with 00)
        # century year divided by 400 is leap year
        if (year % 400 == 0) and (year % 100 == 0):
            return True
        # not divided by 100 means not a century year
        # year divided by 4 is a leap year
        elif (year % 4 == 0) and (year % 100 != 0):
            return True
        else:
            return False
    def attempt_correction():
        if (row_in[date_cols[1]] == 2) & (row_in[date_cols[2]] > 28):
            row_in[date_cols[1]] = 3
            row_in[date_cols[2]] = 1
    # divided by 100 means century year (ending with 00)
    # century year divided by 400 is leap year
    if is_leap_year(row_in[date_cols[0]]):
        pass
    else:
        # not a leap year
        attempt_correction()
    date_str = str(int(row_in[date_cols[0]])) + '-' + str(int(row_in[date_cols[1]])) + '-' + str(int(row_in[date_cols[2]]))
    date_val = pd.to_datetime(date_str, errors='coerce')
    return date_val


def datestamp(fname, fmt='%Y-%m-%d') -> str:
    """
    Append the date to the filename.
    Parameters
    ----------
    fname :
        string
        The filename or full path to the filename

    fmt :
         string
         The format of the date portion (Default value = '%Y-%m-%d')

    Returns
    -------
    type
        string
        The revised filename containing the formatted date pattern

    """
    assert fname is not None, 'File name expected'
    # This creates a timestamped filename so we don't overwrite our good work
    fname_parts = fname.rsplit('.', 1)
    # print(fname_parts)
    revised_fmt = fmt
    if len(fname_parts) < 2:
        revised_fmt = f'{fname_parts[0]}-{fmt}'
    else:
        revised_fmt = f'{fname_parts[0]}-{fmt}.{fname_parts[-1]}'
    return dt.date.today().strftime(revised_fmt).format(fname=fname)


def label(fname, label: str = 'labeled'):
    """

    Parameters
    ----------
    fname :

    label: str :
         (Default value = 'labeled')

    Returns
    -------

    """
    assert fname is not None, 'File name expected'
    # split the fname by last occurence of the dot character
    fname_parts = fname.rsplit('.', 1)
    # print(fname_parts)
    name_label = label
    if len(fname_parts) < 2:
        revised_fmt = f'{fname_parts[0]}-{name_label}'
    else:
        revised_fmt = f'{fname_parts[0]}-{name_label}.{fname_parts[-1]}'
    return dt.date.today().strftime(revised_fmt).format(fname=fname)


def get_func_def(func):
    """
    Return the function definition as a string
    :param func: name with which this function was defined
    :return:
    :rtype:
    """
    return inspect.getsource(func)


def properties_to_frame(props: dict):
    """
    Dump the properties in the specified dict as a dataframe of key, value columns
    :param props:
    :type props:
    :return:
    :rtype:
    """
    assert props is not None, 'A valid properties dictionary is required'
    props_df = pd.DataFrame(data={'key': props.keys(), 'value': props.values()}, columns=['key', 'value'])
    return props_df

def get_quantiles(df: pd.DataFrame, rel_col: str, q_probs: list = None, ignore_nan: bool=True):
    """
    Return the calculated quantiles from the specified column in the dataframe.
    :param df: the DataFrame containing the relevant column
    :param rel_col: the relevant column to compute the quantiles from
    :param q_probs: a sequence of probabilities for the quantiles to compute - e.g., [0.25, 0.5, 0.75]; if not specified a list containing the lower, median,
    and upper bounds for detecting outliers is returned by default.
    :param ignore_nan: whether to ignore null values when computing the quantiles - if True, null values are ignored; if False, an exception is raised if null values are found.
    :return: list of quantile values for the relevant column
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert rel_col is not None, 'A valid column name expected as input'
    col_sr = df[rel_col].copy(deep=True)
    nulls_vals = col_sr.isna().sum()
    if not ignore_nan and nulls_vals > 0:
        raise ValueError(f'Column {rel_col} contains NaN values')
    col_sr.dropna(inplace=True)
    qr_vals = None
    if q_probs is not None:
        qr_vals = np.quantile(col_sr, q=q_probs)
    # otherwise, continue
    else:
        req_props = [0.25, 0.50, 0.75]
        qr_vals = np.quantile(col_sr, req_props)
    if len(qr_vals) == 1:
        return (qr_vals, )
    return qr_vals

def get_aggs(df: pd.DataFrame, rel_cols: list = None, by: list = None, aggfunc: str = 'mean'):
    """
    Calculates aggregates for the relevant columns.
    :param df: the relevant dataframe
    :param rel_cols: a list of relevant column labels to apply aggregate func to
    :param by: list of columns to group data by before applying aggregate func
    :param aggfunc: the aggregate function to apply to the relevant columns; defaults to 'mean'
    :return: the computed aggregates
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (rel_cols is not None) or not (not rel_cols), 'A valid list or non-empty list of column labels expected as input'
    assert (by is not None) or not (not by), 'A valid list or non-empty list of column labels to group by expected as input'
    group_aggs = df.groupby(by)[rel_cols].agg(aggfunc)
    return group_aggs

def get_correlated(df: pd.DataFrame, num_cols: list = None, corr_thres: float = 0.70):
    """
    Return list of column labels with highly correlated features, based on the specified correlation threshold.
    :param df: the dataframe of interest
    :param num_cols: list of numeric columns or features in the dataframe
    :param corr_thres: pairs of features or columns with correlations greater than this threshold should be flagged as highly correlated; defaults to 0.70.
    :return: list of columns or features that are highly correlated
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (num_cols is not None) or not (not num_cols), 'A valid list or non-empty list of column labels expected as input'
    # Set of all names of correlated columns
    col_corr = set()
    corr_mat = df[num_cols].corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > corr_thres:
                colname = corr_mat.columns[i]
                col_corr.add(colname)
    return list(col_corr)

def apply_impute(df: pd.DataFrame, rel_cols: list = None, by: list = None, aggfunc: str = 'mean'):
    """
    Apply imputation using the specified aggregation method on the relevant columns.
    :param df: the relevant dataframe
    :param rel_cols: the relevant column labels to apply aggregate func to
    :param by: list of columns to group data by before applying aggregate func
    :param aggfunc: the aggregate function to apply to the relevant columns; defaults to 'mean'
    :return: the transformed or imputed dataframe
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (rel_cols is not None) or not (not rel_cols), 'A valid list or non-empty list of column labels expected as input'
    assert (by is not None) or not (not by), 'A valid list or non-empty list of column labels to group by expected as input'
    # Impute median for calories
    imputed_df = df.copy(deep=True)
    # now execute the imputation accordingly now the full dataframe is in place
    for col in rel_cols:
        imputed_df.loc[:, col] = imputed_df[col].fillna(imputed_df.groupby(by, observed=True)[col].transform(aggfunc))
    return imputed_df

def apply_clipping(df: pd.DataFrame, rel_cols: list, filterby: str, pos_thres: bool=False, ):
    """
    Clips the relevant columns based on their category or group aggregate statistics and outlier rules - that is, an outlier is less than 1st_quartile - 1.5*iqr and more than 3rd_quartile + 1.5*iqr; additionally, enforce that the values must be positive as desired.
    :param df: the relevant dataframe
    :param rel_cols: the list of columns to clip
    :param filterby: column to group by or filter the data by
    :param pos_thres: enforce positive clipping boundaries or thresholds values
    :return: a clipped dataframe
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert (rel_cols is not None) or not (not rel_cols), 'A valid list or non-empty list of column labels expected as input'
    LOGGER.debug('Clipping the dataframe columns = {} filtered by {}', rel_cols, filterby)
    def cat_thresholds(rel_col: str, rel_cat: str, l_quartile: float = 0.25, u_quartile: float = 0.75):
        is_not_null = df[rel_col].notnull()
        cat_df = df[is_not_null]
        cat_df = cat_df[cat_df[filterby] == rel_cat]
        col_iqr = iqr(cat_df[rel_col])
        qvals = get_quantiles(cat_df, rel_col, [l_quartile, u_quartile])
        l_thres = qvals[0] - 1.5 * col_iqr
        u_thres = qvals[1] + 1.5 * col_iqr
        l_thres = max(0, l_thres) if pos_thres else l_thres
        u_thres = max(0, u_thres) if pos_thres else u_thres
        return l_thres, u_thres
    # Clip the  affected columns based on which category they are associated with
    clipped_df = df.copy(deep=True)
    filter_vals = df[filterby].unique()
    for col in rel_cols:
        clipped_subset = []
        for filter_val in filter_vals:
            l_thres, u_thres = cat_thresholds(col, filter_val)
            clipped_subset.append(clipped_df[clipped_df[filterby] == filter_val][col].clip(lower=l_thres, upper=u_thres))
        clipped_sr = pd.concat(clipped_subset, ignore_index=True)
        clipped_df.loc[:, col] = clipped_sr.values
    return clipped_df


def parse_special_features(special_feat_str, feature_mappings: dict, sep: str =',', ):
    """
    Returns a dictionary with flags indicating the special features included in the movie
    :param special_feat_str: any relevant string that may contain some special features to be matched - e.g., "Trailers, commentaries, feat1, feat2,"
    :type special_feat_str: str
    :param feature_mappings: complete dictionary of features or tokens to be matched in feature string and their corresponding desired labels - e.g., {"feat1": "label1", "feat2": "label2", "Trailers": "trailers"}
    :type feature_mappings: dict
    :param sep: separator character to be used in the feature string, defaults to ','
    :return: list of flags indicating the special features, identified by the feature mapping keys, included in the input string correctly matched
    :rtype: list
    """
    assert special_feat_str is not None, "Special features expected"
    assert feature_mappings is not None, 'Special feature mappings expected'
    feat_split = special_feat_str.split(sep)
    feat_keys = list(feature_mappings.keys())
    feat_mappings = list(feature_mappings.values())
    feat_pattern = {mapping: 0 for mapping in feat_mappings}
    for feat_key in feat_keys:
        if feat_key in feat_split:
            feat_pattern[feature_mappings.get(feat_key)] = 1
    return list(feat_pattern.values())

def safe_copy(data_in: Union[pd.DataFrame, pd.Series, list, np.ndarray, Any]):
    """
    Safely make a copy of the input data.
    :param data_in: input data, which can be any of dataframe, series, list, or numpy array
    :type data_in:
    :return: deep copy of the data where possible
    :rtype: Any
    """
    if isinstance(data_in, (pd.DataFrame, pd.Series)):
        new_data = data_in.copy(deep=True) if (data_in is not None) else None
    elif isinstance(data_in, (list, np.ndarray)):
        new_data = copy.deepcopy(data_in) if (data_in is not None) else None
    else:
        new_data = data_in.copy() if (data_in is not None) else None
    return new_data

def winsorize_it(data_in: Union[pd.Series, list, np.ndarray], limits: list = [0.05, 0.05], **kwargs):
    """
    Apply winsorize to the input data - i.e., replaces extreme values (outliers) at both ends of the data with the
    nearest values within the dataset. It doesn't discard data but instead adjusts the most extreme values to reduce
    their impact. Therefore, prefer this, when you want to preserve the data structure (i.e., keep the sample size
    the same) but still reduce the effect of extreme values. For datasets where removing data points (as in trimming)
    would result in an unrepresentative or incomplete sample, winsorization preserves all values, ensuring the dataset remains usable.
    :param data_in: input data, which can be any of series, list, or numpy array
    :type data_in:
    :param limits: scipe of the winsorization - defaults to 5% limits for both ends
    :type limits:
    :param kwargs: any other scipy.stats.mstats.winsorize parameters
    :return: Returns a Winsorized version of the input data.
    :rtype:
    """
    assert data_in is not None, 'Input data expected'
    assert len(limits) == 2, 'Limits must be a list of two values'
    assert limits[0] >= 0.0, 'Lower limit must be greater than or equal to zero'
    assert limits[1] <= 1.0, 'Upper limit must be less than or equal to 1; much better to use a small proportion - e.g., 0.05'
    assert limits[0] <= limits[1], 'Lower limit must be less than or equal to upper limit'
    data_out = None
    if isinstance(data_in, pd.Series):
        data_out = mstats.winsorize(data_in, limits=limits, **kwargs)
    elif isinstance(data_in, (list, np.ndarray)):
        data_out = mstats.winsorize(np.array(data_in), limits=limits, **kwargs)
    return data_out

def trim_both(data_in: Union[pd.Series, list, np.ndarray], proportiontocut: float = 0.05, **kwargs):
    """
    Slice off a proportion of items from both ends of the input data - removes (trims) the lowest and highest percentage
    of data points - i.e., discards a portion of data at both ends. Note that, this does not retains the overall structure
    of the dataset as it discards extreme data points. Note suited for datasets where removing data points
    (as in trimming) would result in unrepresentative or incomplete sample.
    :param data_in: input data, which can be any of series, list, or numpy array
    :type data_in:
    :param proportiontocut: proportion of data to trim from both ends (in the range 0-1) - defaults to 5%
    :type proportiontocut:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    assert data_in is not None, 'Input data expected'
    assert proportiontocut >= 0.0, 'Proportion to cut must be greater than or equal to zero'
    assert proportiontocut <= 1.0, 'Proportion to cut must be less than or equal to one'
    data_out = None
    if isinstance(data_in, pd.Series):
        data_out = mstats.trimboth(data_in, proportiontocut=proportiontocut, **kwargs)
    elif isinstance(data_in, (list, np.ndarray)):
        data_out = mstats.trimboth(np.array(data_in), proportiontocut=proportiontocut, **kwargs)
    return data_out
