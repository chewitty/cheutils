import numpy as np
import pandas as pd
import re
import pingouin as pg
import datetime as dt
import inspect

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

def find_numeric(row_str):
    """
    Return any numeric characters found - i.e., numbers from 0-9
    :param row_str: string of interest
    :return: numeric digits or blank
    """
    return re.findall('\d', row_str)

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

def quantilefy(df: pd.DataFrame, rel_col: str, q: list = None):
    """
    Return the calcuated quantiles specified by the list otherwise a list containing the lower, median,
    and upper bounds for detecting outliers by default.
    :param df: the DataFrame containing the relevant column
    :param rel_col: the relevant column
    :param q: a sequence of probabilities for the quantiles to compute
    :return: list of quantile values
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert rel_col is not None, 'A valid column name expected as input'
    col_sr = df[rel_col]
    assert col_sr.isna().sum() == 0, 'Column contains NaN values'
    qr_vals = None
    if q is not None:
        qr_vals = np.quantile(col_sr, q=q)
    # otherwise, continue
    else:
        req_props = [0.25, 0.50, 0.75]
        qr_vals = np.quantile(col_sr, req_props)
    return qr_vals

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


def datestamped(fname, fmt='%Y-%m-%d') -> str:
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


def dump_properties(props: dict):
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