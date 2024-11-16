import numpy as np
import pandas as pd
import re
import pingouin as pg
import datetime as dt
import inspect
from scipy.stats import iqr
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

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
    Clips the relevant columns based on outlier rules - that is, an outlier is less than 1st_quartile - 1.5*iqr and more than 3rd_quartile + 1.5*iqr; additionally, enforce that the values must be positive as desired.
    :param df: the relevant dataframe
    :param rel_cols: the list of columns to clip
    :param filterby: column to filter the data by
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
