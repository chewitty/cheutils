import os
import pandas as pd
from cheutils.loggers import LoguruWrapper
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype, is_string_dtype

LOGGER = LoguruWrapper().get_logger()

def apply_replace_patterns(df: pd.DataFrame, replace_dict: dict):
    assert df is not None, 'A valid dataframe is required'
    assert replace_dict is not None, 'A valid replace_dict is required'
    LOGGER.debug('Doing replace patterns ...')
    rel_col = replace_dict.get('rel_col')
    col_repl_dict = replace_dict.get('replace_dict')
    is_regex = replace_dict.get('regex')
    data_sr = df[rel_col].replace(col_repl_dict, regex=is_regex)
    return rel_col, data_sr

def apply_type(df: pd.DataFrame, rel_col: str, req_type: str='int'):
    assert df is not None, 'A valid dataframe is required'
    assert rel_col in df.columns, 'A valid column is required'
    if req_type == 'int' and not is_integer_dtype(df[rel_col]):
        df.loc[:, rel_col] = df[rel_col].astype(int)
    elif req_type == 'float' and not is_float_dtype(df[rel_col]):
        df.loc[:, rel_col] = df[rel_col].astype(float)
    elif req_type == 'str' and not is_string_dtype(df[rel_col]):
        df.loc[:, rel_col] = df[rel_col].astype(str)
    elif req_type == 'bool' and not is_bool_dtype(df[rel_col]):
        df.loc[:, rel_col] = df[rel_col].astype(bool)
    elif req_type == 'datetime' and not is_datetime64_any_dtype(df[rel_col]):
        df.loc[:, rel_col] = pd.to_datetime(df[rel_col], errors='coerce', utc=True)
    elif req_type == 'categorical' and not is_categorical_dtype(df[rel_col]):
        df.loc[:, rel_col] = df[rel_col].astype('category')
    return rel_col, df[rel_col]

def apply_calc_feature(df: pd.DataFrame, rel_col: str, col_gen_func_dict):
    assert df is not None, 'A valid dataframe is required'
    assert rel_col is not None, 'A valid column is required'
    col_gen_func = col_gen_func_dict.get('func')
    func_kwargs: dict = col_gen_func_dict.get('kwargs')
    inc_target = col_gen_func_dict.get('inc_target')
    if inc_target is not None and inc_target:
        if (func_kwargs is not None) or not (not func_kwargs):
            calc_feat = df.agg(col_gen_func, **func_kwargs, target=self.target, axis=1, )
        else:
            calc_feat = df.agg(col_gen_func, target=self.target, axis=1, )
    else:
        if (func_kwargs is not None) or not (not func_kwargs):
            calc_feat = df.agg(col_gen_func, **func_kwargs, axis=1)
        else:
            calc_feat = df.agg(col_gen_func, axis=1)
    return rel_col, calc_feat