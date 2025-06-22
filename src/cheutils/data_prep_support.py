import pandas as pd
import dill as pickle
from cheutils.loggers import LoguruWrapper
from pandas.api.types import is_datetime64_any_dtype, is_categorical_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype, is_string_dtype
from loky import get_reusable_executor

LOGGER = LoguruWrapper().get_logger()

class PickleableLambdaFunc:
    def __init__(self, lambda_func):
        self.lambda_func = lambda_func

    def __getstate__(self):
        return pickle.dumps(self.lambda_func)

    def __setstate__(self, state):
        self.lambda_func = pickle.loads(state)

    def __call__(self, *args, **kwargs):
        return self.lambda_func(*args, **kwargs)

def apply_replace_patterns(df: pd.DataFrame, replace_dict: dict):
    assert df is not None, 'A valid dataframe is required'
    assert replace_dict is not None, 'A valid replace_dict is required'
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

def force_joblib_cleanup():
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
