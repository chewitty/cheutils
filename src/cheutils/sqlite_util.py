import os
import numpy as np
import pandas as pd
import sqlite3
from cheutils.properties_util import AppProperties
from cheutils.project_tree import get_data_dir
from cheutils.exceptions import SQLiteUtilException
from cheutils.loggers import LoguruWrapper

APP_PROPS = AppProperties()
LOGGER = LoguruWrapper().get_logger()
SQLITE_DB = APP_PROPS.get('project.sqlite3.db')

def save_param_grid_to_sqlite_db(param_grid: dict, table_name: str='promising_grids', grid_resolution: int=1,
                                 model_prefix: str=None, **kwargs):
    """
    Save the input data to the underlying project SQLite database (see app-config.properties for DB details).
    :param param_grid: input parameter grid data to be saved or persisted
    :type param_grid:
    :param grid_resolution: the prevailing parameter grid resolution or maximum number of parameters supported by grid
    :param model_prefix: any prevailing model prefix
    :param table_name: the name of the table - this could be a project-specific name, for example, the configured
    estimator name if caching promising hyperparameter grid
    :type table_name:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    assert param_grid is not None, 'Input parameter grid data must be provided'
    assert grid_resolution > 0, 'A valid grid resolution (>0) expected'
    assert table_name is not None and len(table_name) > 0, 'Table name must be provided'
    conn = None
    cursor = None
    model_prefix = model_prefix if model_prefix is not None else ''
    sqlite_db = os.path.join(get_data_dir(), SQLITE_DB)
    try:
        data_grid = {}
        for key, value in param_grid.items():
            key = key.split(model_prefix + '__')[1] if model_prefix in key else key
            data_grid[key] = value
        data_df = pd.DataFrame(data_grid, index=[0])
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(sqlite_db)
        # Create a cursor object using the cursor() method
        cursor = conn.cursor()
        tb_cols = ['grid_resolution']
        tb_cols.extend(data_df.columns.tolist())
        num_tb_cols = len(tb_cols)
        crt_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({str(tb_cols).strip("[]")})'
        cursor.execute(crt_stmt)
        # insert the rows of data
        INSERT_STMT = f'INSERT INTO {table_name} VALUES ({",".join(["?"] * num_tb_cols)})'
        for index, row in data_df.iterrows():
            row_vals = [grid_resolution]
            row_vals.extend(row.tolist())
            cursor.execute(INSERT_STMT, row_vals)
        conn.commit()
        LOGGER.debug('Updated SQLite DB: {}', sqlite_db)
    except ValueError as err:
        msg = LOGGER.error('Value error attempting to save to: {}, {}', sqlite_db, err)
        tb = err.__traceback__
        raise SQLiteUtilException(err).with_traceback(tb)
    except Exception as err:
        msg = LOGGER.error("SQLite DB error: {}, {}", sqlite_db, err)
        tb = err.__traceback__
        raise SQLiteUtilException(err).with_traceback(tb)
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

def get_param_grid_from_sqlite_db(tb_name: str='promising_grids', grid_resolution: int=1, model_prefix: str=None, **kwargs):
    """
    Fetches data from the underlying SQLite DB using the query string.
    :param tb_name: the table name to be queried
    :param grid_resolution: the prevailing parameter grid resolution or maximum number of parameters supported by grid
    :param model_prefix: any prevailing model prefix
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    assert tb_name is not None and len(tb_name) > 0, 'Table name must be provided'
    conn = None
    cursor = None
    sqlite_db = os.path.join(get_data_dir(), SQLITE_DB)
    try:
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(sqlite_db)
        # Create a cursor object using the cursor() method
        cursor = conn.cursor()
        query_str = 'SELECT * FROM ' + tb_name + ' WHERE grid_resolution=:grid_resolution'
        result_cur = cursor.execute(query_str, {'grid_resolution': grid_resolution})
        data_row = np.array(list(cursor.fetchone()))
        data_row = data_row.reshape(1, -1)
        col_names = []
        for column in result_cur.description:
            col_names.append(column[0])
        data_df = pd.DataFrame(data_row, columns=col_names, index=[0])
        data_df.drop(columns=['grid_resolution'], inplace=True)
        data_df.rename(columns=lambda x: model_prefix + '__' + x if model_prefix is not None else x, inplace=True)
        grid_dicts = data_df.to_dict('records')
        return grid_dicts[0] if grid_dicts is not None or not (not grid_dicts) else None
    except Exception as warning:
        LOGGER.warning('SQLite DB error: {}, {}', sqlite_db, warning)
        # check if the promising grid is still to be generated
        if 'no such table' in str(warning):
            return None
        tb = warning.__traceback__
        raise SQLiteUtilException(warning).with_traceback(tb)
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass