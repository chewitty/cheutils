import math
import time
import datetime as dt
import sys, os
import traceback
import dask.dataframe as dd
import pandas as pd
import pyodbc
import pymysql
import MySQLdb
import pymssql
import psycopg2
from urllib.parse import quote_plus, unquote_plus
import mysql.connector
import sqlalchemy as sa
from typing import Union

from analytics import timeout
from dask.delayed import delayed
from docutils.nodes import tbody
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import text
from sqlalchemy.event import listens_for
from cheutils.loggers import LoguruWrapper
from cheutils.common_utils import datestamp
from cheutils.properties_util import AppProperties
from cheutils.decorator_debug import debug_func
from cheutils.decorator_singleton import singleton
from cheutils.decorator_timer import track_duration
from cheutils.exceptions import DBToolException, DSWrapperException

# Define project constants.
APP_PROPS = AppProperties()
DEFAULT_DS_CONFIG = 'ds-config.properties'
LOGGER = LoguruWrapper().get_logger()

class DBTool(object):
    ds_config_ = None
    """
    A static method responsible for creating and returning a new instance (called before __init__)
    """
    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created,
        or else returns the previous singleton object
        """
        return super().__new__(cls)

    """
    An instance method, the class constructor, responsible for initializing the attributes of the newly created
    """
    def __init__(self, ds_config: dict, verbose=False):
        """
        You can specify a preferred SQL engine to use; the default is sqlalchemy configured for MySQL provided.
        Parameters:
        :param ds_config: a dictionary holding the database configuration and other request parameters
        :type ds_config: dict
        :param verbose: enables interactions with underlying connection to be verbose or prints out statements
        :type verbose: bool
        """
        if ds_config is None:
            LOGGER.debug('A database configuration must be provided')
            raise DBToolException('A database configuration must be provided')
        self.ds_config_ = ds_config
        # set up the engine
        try:
            query = self.ds_config_.get('query')
            connect_args = self.ds_config_.get('connect_args')
            if query is not None:
                attempt_direct_conn = bool(eval(str(query.get('direct_conn'))))
                make_verbose = bool(eval(str(query.get('verbose'))))
                LOGGER.debug('Interactions with underlying connection are verbose = {}', make_verbose)
                verbose = True if make_verbose else verbose
            # sqlalchemy.engine.Engine, is what delivers a DB Connection
            if (self.ds_config_.get('drivername') is None) or ('mysql' in self.ds_config_.get('drivername')):
                # LOGGER.debug('Connection properties', self.ds_config_)
                timeout = int(query.get('timeout'))
                connect_args = {'timeout': timeout} if connect_args is None else connect_args
                try:
                    self.sql_engine_ = create_engine(URL(**self.ds_config_), connect_args=connect_args,
                                                     pool_recycle=900, future=True, echo=verbose)
                except Exception as err:
                    LOGGER.error('An error occured while connecting to the database: {}'.format(err))
                    raise DBToolException(err)
                if 'pymysql' in self.ds_config_.get('drivername'):
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_pymysql_connect(*args, **kwargs):
                        # LOGGER.debug('Arguments: {}, {}', args, kwargs)
                        return self.__pymysql_creator(*args, **kwargs)
                elif 'mysqldb' in self.ds_config_.get('drivername'):
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_mysqlclient_connect(*args, **kwargs):
                        LOGGER.debug('Arguments: {}', kwargs)
                        return self.__mysqlclient_creator(*args, **kwargs)
                elif 'mysqlconnector' in self.ds_config_.get('drivername'):
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_mysqlconnector_connect(*args, **kwargs):
                        LOGGER.debug('Arguments: {}', kwargs)
                        return self.__mysqlconnector_creator(*args, **kwargs)
                else:
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_mysql_connect(*args, **kwargs):
                        LOGGER.debug('Arguments: {}', kwargs)
                        return self.__pyodbc_creator(*args, **kwargs)
            elif (self.ds_config_.get('drivername') is None) or ('pymssql' in self.ds_config_.get('drivername')) or ('psycopg2' in self.ds_config_.get('drivername')):
                try:
                    if 'psycopg2' in self.ds_config_.get('drivername'):
                        timeout = int(query.get('timeout'))
                        connect_args = {'timeout': timeout} if connect_args is None else connect_args
                    self.sql_engine_ = create_engine(URL(**self.ds_config_), connect_args=connect_args,
                                                     pool_recycle=900, future=True, echo=verbose)
                except Exception as err:
                    LOGGER.error('An error occured while connecting to the database: {}'.format(err))
                    raise DBToolException(err)
                if 'psycopg2' in self.ds_config_.get('drivername'):
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_psycopg2_connect(*args, **kwargs):
                        # LOGGER.debug('Arguments: {}, {}', args, kwargs)
                        return self.__psycopg2_creator(*args, **kwargs)
                else:
                    @listens_for(self.sql_engine_, 'do_connect')
                    def do_pymssql_connect(*args, **kwargs):
                        LOGGER.debug('Arguments: {}', kwargs)
                        return self.__pymssql_creator(*args, **kwargs)
            else:
                try:
                    self.sql_engine_ = create_engine(URL(**self.ds_config_), connect_args=connect_args, future=True,
                                                     fast_executemany=True, use_setinputsizes=False, pool_recycle=900,
                                                     echo=verbose)
                except Exception as err:
                    LOGGER.error('An error occured while connecting to the database: {}'.format(err))
                    raise DBToolException(err)

                @listens_for(self.sql_engine_, 'do_connect')
                def do_mssql_connect(*args, **kwargs):
                    LOGGER.debug('Arguments: {}', kwargs)
                    return self.__mssql_pyodbc_creator(*args, **kwargs)
            # continue along
            LOGGER.debug('Using datasource engine = {}', self.sql_engine_)
            LOGGER.debug('Using database = {}', self.ds_config_['database'])
        except DBToolException as ex:
            raise ex
        except Exception as ex:
            LOGGER.error('FAILURE: {}', ex)
            raise DBToolException(ex)
        finally:
            LOGGER.debug('Completed attempt at creating an appropriate DBTool for the db = {}', self.ds_config_['database'])

    def __str__(self):
        info = f"DBTool for {self.ds_config_.get('host')}, {self.ds_config_.get('database')}"
        return info

    def __pyodbc_creator(self, *args, **kwargs):
        try:
            return self.__create_pyodbc_connection()
        except pyodbc.Error as ex:
            try:
                return self.__mysqlclient_creator()
            except DBToolException as err:
                raise
            except Exception as donotignore:
                raise donotignore

    def __mssql_pyodbc_creator(self, *args, **kwargs):
        try:
            return self.__create_mssql_pyodbc_connection()
        except pyodbc.Error as ex:
            try:
                return self.__create_pyodbc_connection()
            except DBToolException as err:
                raise
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __pymssql_creator(self, *args, **kwargs):
        try:
            return self.__create_pymssql_connection()
        except pymssql.Error as ex:
            try:
                return self.__create_pyodbc_connection()
            except DBToolException as err:
                raise
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __psycopg2_creator(self, *args, **kwargs):
        try:
            return self.__create_psycopg2_connection()
        except pymssql.Error as ex:
            try:
                return self.__create_pyodbc_connection()
            except DBToolException as err:
                raise
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __pymysql_creator(self, *args, **kwargs):
        try:
            return self.__create_pymysql_connection()
        except pymysql.Error as mysqldb_err:
            try:
                # fall back on another variant
                return self.__create_mysqlclient_connection()
            except DBToolException as err:
                raise
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __mysqlclient_creator(self, *args, **kwargs):
        try:
            return self.__create_mysqlclient_connection()
        except mysql.connector.Error as mysqldb_err:
            try:
                # fall back on another variant
                return self.__create_pymysql_connection()
            except DBToolException as err:
                raise
            except RuntimeError as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __mysqlconnector_creator(self, *args, **kwargs):
        try:
            return self.__create_mysqlconnector_connection()
        except RuntimeError as mysqldb_err:
            try:
                # fall back on another variant
                return self.__create_pymysql_connection()
            except DBToolException as err:
                raise
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('An error occured while connecting to the database: {}', err)
                raise DBToolException(err).with_traceback(tb)

    def __get_connection(self, autocommit=True):
        LOGGER.debug('Obtaining connection to DB... autocommit = {}', autocommit)
        try:
            return self.sql_engine_.connect().execution_options(stream_results=True)
        except DBToolException as err:
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish default configured connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_mssql_pyodbc_connection(self, autocommit=True):
        return self.__create_pyodbc_connection(autocommit=autocommit)

    def __create_pyodbc_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the pyodbc
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using PyODBC... autocommit = {}', autocommit)
        try:
            dbdriver = self.ds_config_.get('query').get('driver')
            sep = ':' if 'mysql' in dbdriver.lower() else ','
            dbserver = self.ds_config_.get('host')
            dbport = str(self.ds_config_.get('port'))
            dbname = self.ds_config_.get('database')
            username = self.ds_config_.get('username')
            password = unquote_plus(self.ds_config_.get('password'))
            encoding = self.ds_config_.get('query').get('encoding')
            conn_str = f'driver={dbdriver};Server={dbserver};port={dbport};Database={dbname};DSN={dbname};MULTI_HOST=1;UID={username}'
            if encoding is not None:
                conn_str += f';charset={encoding}'
            LOGGER.debug('Pyodbc connection string: {}'.format(conn_str))
            conn_str = conn_str + f';PWD={password}'
            conn = pyodbc.connect(conn_str, autocommit=autocommit)
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish pyodbc connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_pymssql_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the pymssql
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using PyMSSQL... autocommit = {}', autocommit)
        try:
            conn = pymssql.connect(server=self.ds_config_.get('host') + ':' + str(self.ds_config_.get('port')),
                                   database=self.ds_config_.get('database'),
                                   user=self.ds_config_.get('username'),
                                   password=unquote_plus(self.ds_config_.get('password')),
                                   charset=self.ds_config_.get('query').get('encoding'),
                                   autocommit=autocommit)
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish pymssql connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_psycopg2_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the psycopg2
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using psycopg2... autocommit = {}', autocommit)
        try:
            conn = psycopg2.connect(host=self.ds_config_.get('host'),
                                    port=self.ds_config_.get('port'),
                                    database=self.ds_config_.get('database'),
                                    user=self.ds_config_.get('username'),
                                    password=unquote_plus(self.ds_config_.get('password')),)
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish psycopg2 connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_pymysql_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the pymysql
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using PyMySQL... autocommit = {}', autocommit)
        try:
            conn = pymysql.connect(host=self.ds_config_.get('host'),
                                   port=int(self.ds_config_.get('port')),
                                   database=self.ds_config_.get('database'),
                                   user=self.ds_config_.get('username'),
                                   password=unquote_plus(self.ds_config_.get('password')),
                                   charset=self.ds_config_.get('query').get('encoding'),
                                   use_unicode=True)
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish pymysql connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_mysqlclient_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the mysqldb client
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using MySQLdb... autocommit = {}', autocommit)
        try:
            charset = self.ds_config_.get('query').get('encoding')
            charset = charset if charset is not None else 'utf8mb4'
            conn = MySQLdb.connect(host=self.ds_config_.get('host'),
                                   port=self.ds_config_.get('port'),
                                   database=self.ds_config_.get('database'),
                                   user=self.ds_config_.get('username'),
                                   password=unquote_plus(self.ds_config_.get('password')),
                                   charset=charset,
                                   autocommit=autocommit, )
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish mysqldb connection {}', err)
            raise DBToolException(err).with_traceback(tb)

    def __create_mysqlconnector_connection(self, autocommit=True):
        """
        Creates a direct connection to the DB using the mysqlconnector client
        :return: Connection to the underlying DB
        """
        LOGGER.debug('Obtaining connection to DB using mysqlconnector... autocommit = {}', autocommit)
        try:
            conn = mysql.connector.connect(host=self.ds_config_.get('host'),
                                 port=int(self.ds_config_.get('port')),
                                 database=self.ds_config_.get('database'),
                                 user=self.ds_config_.get('username'),
                                 password=unquote_plus(self.ds_config_.get('password')),
                                 charset=self.ds_config_.get('query').get('encoding'),
                                 use_unicode=True)
            return conn
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure to establish mysqlconnector connection = {}', err)
            raise DBToolException(err).with_traceback(tb)

    def execute_query(self, query: str) -> None:

        """
        Executes a specified query on the underlying DB

        Paramters
        ---------

        query : str,
            Query to execute on the underlying Database.

        Returns
        -------
        None
            Nothing is return

        Example
        -------
        ds_key: str = "MSSQL3"

        db_tool = di_utils.get_db_tool(ds_key=ds_key)

        TABLE_NAME: str = "ffm_sample_actuals"

        READ_QUERY: str = f'select * from {TABLE_NAME}'

        db_tool.execute_query(READ_QUERY)

        """
        try:
            with self.__get_connection() as connection:
                with connection.begin():
                    try:
                        LOGGER.debug('Query: {}', query)
                        connection.execute(text(query))
                    except pyodbc.Error as err:
                        LOGGER.debug(f"DB Error: '{err}'")
                        raise
        except Exception as err:
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err).with_traceback(ex.__traceback__)
        finally:
            LOGGER.debug('Finished attempt to executing a query on db')

    # Persist specified dataframe to the underlying DB repository
    def persist(self, df_in: pd.DataFrame, db_table, if_exists='append', force_create=False, index_as_col=False,
                chunksize=50, method='multi'):
        """Stores or saves the specified dataframe to the underlying DB
        Parameters:
        df_in(DataFrame): the dataframe to be persisted
        db_table(str): the underlying target DB table name
        if_table_exists(str): how to behave if the table already exists {‘fail’, 'replace', ‘append’}, default ‘append’
        force_create(bool): force to recreate the table even if it exists - the same as 'replace'; AVOID using this!
        index_as_col(bool): persist the dataframe index as a column, with the index label as the column name, default False
        chunksize(int): the desired or pragmatic chunksize; default is 50
        """
        LOGGER.debug('Persisting dataframe to {}, {}', self.ds_config_['database'], 'DB')
        if 'replace' == if_exists:
            # check to make absolutely sure because the 'replace' option is dangerous and NOT RECOMMENDED
            if_table_exists = 'replace' if force_create else 'fail'
            if if_table_exists == 'replace':
                LOGGER.debug('DANGER: The underlying table will be recreated')
        try:
            with self.__get_connection() as connection:
                # export the dataframe to the DB
                for count in range(1):  # simply to show some progress consistent with the other calls
                    df_in.to_sql(name=db_table, con=connection, if_exists=if_exists, index=index_as_col,
                                 method=method, chunksize=chunksize)
        except Exception as err:
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err)
        finally:
            LOGGER.debug('Finished attempt to persist to db')

    # Read dataframe from the specified table in the underlying DB repository
    def read_from_db_for_geolevel(self, script, query_string='select * from #tempoutput'):
        """
        Execute the script to create to put the geolevel data into the temp table #tempoutput and
        with a simple query of SELECT, get all the rows.
        Parameters:
            script(str): The script to execute
            query_string: The query to read the ouput table of the script.
        Returns:
            (DataFrame): The ouput data as dataframe.
        """
        try:
            with self.__get_connection() as connection:
                connection.execute(script)
                df = pd.read_sql(text(query_string), connection)
                LOGGER.debug('Shape of dataframe = {}', df.shape)
                connection.close()
                return df
        except Exception as err:
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err)
        finally:
            LOGGER.debug('Finished attempt to read from db')

    def read_db_table(self, db_table='', index_col=None, coerce_float=False, parse_dates=None,
                      columns=None, force_parallelize=False):
        """
        Reads data from the underlying DB - essentially, reads the entire table
        Parameters:
            db_table(str)    : the underlying DB table name holding the data; the default is empty string
                                but query_string must be provided
            index_col(list)    : Column(s) to set as index
                                the query_string one is empty otherwise, it will be ignored
            coerce_float(bool): Attempts to convert values of non-string, non-numeric objects (like decimal.Decimal) to floating point
            parse_dates(dict): Dict of {column_name: format string} where format string is strftime compatible in case of parsing string times or is one of (D, s, ns, ms, us) in case of parsing integer timestamps.
            columns(list): List of column names to select from SQL table
            chunksize(int): If specified, returns an iterator where chunksize is the number of rows to include in each chunk
            force_parallelize(bool): if set, then chunksize is ignored because the data is read in a parallelized way
        Returns:
            DataFrame: the DB data as a dataframe
        """
        LOGGER.debug('Reading data from underlying table = {}', db_table)
        try:
            with self.__get_connection() as connection:
                # read the dataframe from the DB
                if force_parallelize:
                    LOGGER.debug('Optimized read ...')
                    parts = [delayed(pd.read_sql_table)(db_table, connection, index_col=index_col,
                                                        coerce_float=coerce_float,
                                                        parse_dates=parse_dates, columns=columns)]
                    meta_info = parts[0].compute()
                    db_data_df = dd.from_delayed(parts, meta=meta_info).compute()
                else:
                    # connection.text_factory = lambda x: str(x, 'utf8mb4')
                    db_data_df = pd.read_sql_table(db_table, connection, index_col=index_col, coerce_float=coerce_float,
                                                   parse_dates=parse_dates, columns=columns)
            LOGGER.debug('Shape of dataframe = {}', db_data_df.shape)
            return db_data_df
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err).with_traceback(tb)
        finally:
            LOGGER.debug('Finished attempt to read from db = {}', db_table)

    def read_chunked_table(self, db_table='', query_string=None, index_col=None, coerce_float=False, parse_dates=None,
                           columns=None,
                           chunksize=None, force_parallelize=False):
        """
        Reads data from the underlying DB in chunks and returns list of dataframes
        Parameters:
            db_table(str)    : the underlying DB table name holding the data; the default is empty string
                                but query_string must be provided
            query_string(str)   : the query string
            index_col(list)    : Column(s) to set as index
                                the query_string one is empty otherwise, it will be ignored
            coerce_float(bool): Attempts to convert values of non-string, non-numeric objects (like decimal.Decimal) to floating point
            parse_dates(dict): Dict of {column_name: format string} where format string is strftime compatible in case of parsing string times or is one of (D, s, ns, ms, us) in case of parsing integer timestamps.
            columns(list): List of column names to select from SQL table
            chunksize(int): If specified, returns an iterator where chunksize is the number of rows to include in each chunk
            force_parallelize(bool): if set, then chunksize is ignored because the data is read in a parallelized way
        Returns:
            list(DataFrame): a list of dataframes containing the table data in row chunks
        """
        LOGGER.debug('Reading data from underlying table = {}', db_table)
        LOGGER.debug('Optimized = {}', force_parallelize)
        db_data_chks = []
        try:
            with self.__get_connection() as connection:
                # read the dataframe from the DB
                if query_string is None:
                    LOGGER.debug('Optimized read ...')
                    try:
                        for data_chunk in pd.read_sql(db_table, connection, index_col=index_col,
                                                      coerce_float=coerce_float,
                                                      parse_dates=parse_dates, columns=columns, chunksize=chunksize):
                            db_data_chks.append(data_chunk)
                    except Exception as err:
                        tb = err.__traceback__
                        LOGGER.error('Optimized read failure = {}', err)
                        raise
                else:
                    LOGGER.debug('Optimized read ...')
                    try:
                        for data_chunk in pd.read_sql(text(query_string), connection, index_col=index_col,
                                                      coerce_float=coerce_float,
                                                      parse_dates=parse_dates, columns=columns, chunksize=chunksize):
                            db_data_chks.append(data_chunk)
                    except Exception as err:
                        tb = err.__traceback__
                        LOGGER.error('Optimazed read failure = {}', err)
                        raise
            LOGGER.debug('Number of dataframe chunks = {}', len(db_data_chks))
        except DBToolException as outEx:
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err).with_traceback(tb)
        finally:
            LOGGER.debug('Finished attempt to read from db = {}', db_table)
        return db_data_chks

    def read_from_db(self, db_table='', columns: list=None, query_string='',
                     query_filter=None, force_parallelize=False):
        """
        Reads data from the underlying DB using the specified query string
        Parameters:
            db_table(str)    : the underlying DB table name holding the data; the default is empty string
                                but query_string must be provided
            columns(list)    : the columns to select are set here. This paramter is considered whether
                                the query_string one is empty otherwise, it will be ignored
            query_string(str): the specified query string - essentially a select statement; the default is
                                an empty string (all columns)
            query_filter(dict): the criterias of data selection are set through this parameter.
                                They must be set has a Dict object
            force_parallelize(bool): if set, the data is read in a parallelized way

        Returns:
            DataFrame: the DB data as a dataframe
        """

        if '' == query_string:
            if len(columns) == 0:
                query_string = 'SELECT * FROM ' + db_table  # read all the columns
            else:
                query_string = 'SELECT ' + ', '.join(columns) + ' FROM ' + db_table
        if query_filter:  # constitution of the selection criterias
            if type(query_filter) != dict:
                LOGGER.debug("The parameter 'query_filter' must be a Dictionary rather than a {}",
                                     type(query_filter))
            else:
                query_ftr = ' AND '.join([str(k) + '=' + str(("'" + v + "'") if isinstance(v, str) else v) for k, v in
                                          zip(query_filter.keys(), query_filter.values())])
                query_string += ' WHERE ' + query_ftr
        try:
            with self.__get_connection() as connection:
                # read the dataframe from the DB
                if force_parallelize:
                    LOGGER.debug('Optimized read ...')
                    # process 100 MB chunks
                    parts = [delayed(pd.read_sql)(text(query_string), connection)]
                    meta_info = parts[0].compute()
                    db_data_df = dd.from_delayed(parts, meta=meta_info).compute()
                else:
                    db_data_df = pd.read_sql(text(query_string), connection)
            LOGGER.debug('Shape of dataframe = {}', db_data_df.shape)
            return db_data_df
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('FAILURE: {}', err)
            raise DBToolException(err).with_traceback(tb)
        finally:
            LOGGER.debug('Finished attempt to read from db = {}', db_table)

    # Return the underlying database configuration.
    def get_config(self):
        """
        Provides a handle to the underlying database configuration.
        Returns dict: handle to its configuration
        """
        return self.ds_config_

    # Execute an update query on the underlying DB.
    def update(
            self,
            data_df: pd.DataFrame = None,
            db_table: str = None,
            where_cols: Union[list, tuple] = None,
            where_values: Union[list, tuple] = None,
            set_cols: Union[list, tuple] = None,
            set_values: Union[list, tuple] = None) -> None:

        """
        Update Database Table with specified table in the underlying DB

        The Update is done on specified `set_cols` with `set_values`
        where the specified `where_cols` equals `where_values`

        Parameters
        ----------
        data_df : DataFrame,
            The data to use for populating where clause and column values;
            if not provided, the where_values and set_values must be provided.
        db_table : str or named str,
            The underlying DB table name holding the data - must be provided
        where_cols : list or tuple,
            The specified required condition column names as list of strings
        where_values : list or tuple,
            The matching column values - must match the order and format of where_cols
        set_cols : list or tuple,
            The column names whose values should be updated as a list of column name strings
        set_values : list or tuple,
            The matching column values - must match the order and format of set_cols

        Returns
        -------
        None
            Nothing is return

        Example #1
        ----------
        ds_key = 'MSSQL3'

        db_tool = di_utils.get_db_tool(ds_key=ds_key)

        db_table = "ffm_sample_actuals"

        where_cols = ['country_id', 'cluster_id', 'community_id']

        where_values = [2, 6, 651]

        set_cols = ['activity', 'modified_date']

        set_values = ["map", '2022-03-24 13:56:52']

        db_tool.update(
        db_table=db_table,
        where_cols=where_cols,
        where_values=where_values,
        set_cols=set_cols,
        set_values=set_values)

        Example #2
        ----------
        df = {'activity': {0: "map", 1: "map"},
        'cluster_id': {0: 10, 1: 10},
        'community_id': {0: 651, 1: 1055},
        'country_id': {0: 2, 1: 2},
        'modified_date': {0: "2022-03-24 13:56:52", 1: "2022-03-24 13:56:52"},
        'wk01': {0: 0.0, 1: 0.0},
        'wk02': {0: 0.0, 1: 0.0}}

        update_data = pd.DataFrame(df)

        db_tool.update(
        data_df=update_data,
        db_table=db_table,
        where_cols=where_cols,
        set_cols=set_cols)

        """

        if db_table is None:
            LOGGER.debug('DB table name required and must be specified')
            raise DBToolException('DB table name required and must be specified')

        SET_SEPARATOR: str = ", "
        LOGICAL_OPERATOR: str = " AND "
        if data_df is None:
            # Where Claus
            where_claus_value_parts: list = [column_name + '=' + str(
                "'" + column_value + "'" if (isinstance(column_value, str)) else str(column_value)) for
                                             column_name, column_value in zip(where_cols, where_values)]
            where_claus_value: str = LOGICAL_OPERATOR.join(where_claus_value_parts)

            # Set Claus
            set_claus_value_parts = [column_name + '=' + str(
                "'" + column_value + "'" if (isinstance(column_value, str)) else str(column_value)) for
                                     column_name, column_value in zip(set_cols, set_values)]
            set_claus_value = SET_SEPARATOR.join(set_claus_value_parts)

            # Update Query
            UPDATE_QUERY = f"""UPDATE {db_table} SET {set_claus_value} WHERE {where_claus_value}"""

            # Execute Update Query
            self.execute_query(text(UPDATE_QUERY))

        else:
            # Populate db_table with DataFrame
            for index, row in data_df.iterrows():
                # Where Claus
                where_claus_value_parts = [column_name + '=' + str(
                    "'" + row[column_name] + "'" if (isinstance(row[column_name], str)) else str(row[column_name])) for
                                           column_name in where_cols]
                where_claus_value: str = LOGICAL_OPERATOR.join(where_claus_value_parts)

                # Set Claus
                set_claus_value_parts = [column_name + '=' + str(
                    "'" + row[column_name] + "'" if (isinstance(row[column_name], str)) else str(row[column_name])) for
                                         column_name in set_cols]
                set_claus_value: str = SET_SEPARATOR.join(set_claus_value_parts)

                # Update Query
                UPDATE_QUERY = f"""UPDATE {db_table} SET {set_claus_value} WHERE {where_claus_value}"""

                # Execute Update Query
                self.execute_query(text(UPDATE_QUERY))

    def insert(self, data_df, db_table=''):
        """
        Executes an insert query, usually a query that is beyond what Pandas can do.
        Parameters:
            data_df(DataFrame): the dataframe with the data potentially to persist
            db_table(str): the underlying DB table name holding the data; the default is empty string
        """
        LOGGER.debug('Executing insert query against {}, {}', db_table, '...')
        myTable = sa.Table(db_table, sa.MetaData(), quote=False, autoload_with=self.sql_engine_)
        with self.__get_connection() as connection:
            with connection.begin():
                for count, (index, row) in zip(range(data_df.shape[0]),
                                               data_df.iterrows()):
                    values = tuple([value for value in row])
                    insert = myTable.insert().values(values)
                    result = connection.execute(insert)

    @track_duration(name='save_to_db')
    @debug_func(enable_debug=True, prefix='save_to_db')
    def save_to_db(self, entities_df, db_table=None, start_at=0, chunksize=250, force_insert=False,
                   ignore_duplicates=False):
        """
        Saves the specified dataframe to the DB in chunks - which means that when there is a failure, it is possble to restart
        at the beginning of the chunk where failre occured.
        Parameters:
            entities_df(dataframe): the dataframe to save, which may be large
            db_table(str): the underlying DB table name holding the data
            start_at(int): the start position of the next chunk to process
            chunksize(int): the size of each chunk, which may be equivalent to the number of rows
            force_insert(bool): whether the insert should be used to save data, the default is persist
            ignore_duplicates(bool): whether the insert_ignore_duplicates should be used to save the data, the default is persist
        """

        if db_table is None:
            LOGGER.debug("A DB table must be specified")
            return
        elif type(entities_df) != pd.DataFrame:
            LOGGER.debug("Only Dataframe are accepted")
            return
        elif entities_df.shape[0] == 0:
            LOGGER.debug("The Dataframe must contain records")
            return

        LOGGER.debug('Saving to DB', entities_df.shape)
        num_rows = entities_df.shape[0]
        data_chunks = math.ceil(num_rows / chunksize)
        chunk_start = math.ceil(start_at / chunksize)
        slice_adj = start_at % chunksize
        LOGGER.debug('Rows: {}', num_rows)
        LOGGER.debug('Data chunks: {}', data_chunks)
        LOGGER.debug('Chunk start: {}', chunk_start)
        LOGGER.debug('Slice adjustment: {}', slice_adj)
        for chunk in range(chunk_start, data_chunks):
            slice_start = chunk * chunksize + slice_adj
            LOGGER.debug('Current processing started at: {}', slice_start)
            slice_end = min((chunk + 1) * chunksize, num_rows)
            data_slice = entities_df[slice_start: slice_end]
            if ignore_duplicates:
                self.insert_ignore_duplicates(data_slice, db_table=db_table)
            elif force_insert:
                self.insert(data_slice, db_table=db_table)
            else:
                # ignore chunksize as there is an appropriate default in dbtool
                self.persist(data_slice, db_table)
        LOGGER.debug('Saving completed')

    def delete_all(self, db_table=None):
        """
        Deletes all data rows from the underlying DB table.
        Parameters:
            db_table(str): the underlying DB table name holding the data.
        """
        msg = LOGGER.debug('Executing delete query against {}, {}', db_table, '...')
        if db_table is None:
            LOGGER.debug('DB table name required and must be specified')
            raise DBToolException('DB table name required and must be specified')

        with self.__get_connection() as connection:
            with connection.begin():
                try:
                    stmt = 'DELETE FROM ' + db_table
                    result = connection.execute(text(stmt))
                    msg = LOGGER.debug('Deletion completed')
                except IntegrityError as err:
                    msg = LOGGER.error("DB error: {}", err)
                    tb = err.__traceback__
                    raise DBToolException(err).with_traceback(tb)
                except Exception as err:
                    tb = err.__traceback__
                    raise DBToolException(err).with_traceback(tb)

    def truncate(self, db_table=None):
        underlying_table = db_table
        try:
            with self.__get_connection() as connection:
                # the default mysql+pyodbc dialect is UTF-16. To to eliminate most of the failures
                # the fix is to explicitly set the encoding after the Connection object is created
                # connection.setdecoding(pyodbc.SQL_CHAR, encoding='latin1')
                # connection.setdecoding(pyodbc.SQL_WCHAR, encoding='latin1')
                # connection.setencoding('latin1')
                # connection.begin()
                cursor = connection.connection.cursor()
                # cursor.fast_executemany = True
                try:
                    stmt = f"TRUNCATE TABLE {underlying_table}"
                    cursor.execute(stmt)
                    LOGGER.debug('Underlying table truncated = {}', db_table)
                except Exception as err:
                    tb = err.__traceback__
                    LOGGER.error('Failure: {}', err)
                    raise
                finally:
                    cursor.close()
                    connection.close()
        except Exception as outEx:
            LOGGER.error('Failed: executing truncate on table = {}', underlying_table)
            tb = outEx.__traceback__
            raise DBToolException(outEx).with_traceback(tb)

    def delete(self, db_table=None, filter_by=None):
        """
        Deletes all data rows from the underlying DB table.
        Parameters:
            db_table(str): the underlying DB table name holding the data.
            filter_by(str): a str expression of a where clause
        """
        msg = LOGGER.debug('Executing delete query against {}, {}', db_table, '...')
        if db_table is None:
            LOGGER.debug('DB table name required and must be specified')
            raise DBToolException('DB table name required and must be specified')
        if filter_by is None:
            self.delete_all(db_table)
            return
        with self.__get_connection() as connection:
            with connection.begin():
                try:
                    stmt = 'DELETE FROM ' + db_table + ' ' + filter_by
                    result = connection.execute(text(stmt))
                    msg = LOGGER.debug('Deletion completed')
                except IntegrityError as err:
                    tb = err.__traceback__
                    LOGGER.error("DB error: {0}", err)
                    raise DBToolException(err).with_traceback(tb)
                except Exception as err:
                    tb = err.__traceback__
                    LOGGER.error('Failed: executing delete on table = {}', db_table)
                    raise DBToolException(err).with_traceback(tb)

    def insert_ignore_duplicates(self, data_df, db_table=''):
        """
        Executes an insert query, usually a query that is beyond what Pandas can do; here any duplicates found are ignored. The order of the data columns is very important - so make sure to check the underlying table structure to ensure your dataframe columns are ordered to match the underlying table.
        Parameters:
            data_df(DataFrame): the dataframe with the data potentially to persist
            db_table(str): the underlying DB table name holding the data; the default is empty string
        """
        myTable = sa.Table(db_table, sa.MetaData(), quote=False, autoload_with=self.sql_engine_)
        LOGGER.debug('Executing insert query against {}, {}', db_table, '...')
        with self.__get_connection() as connection:
            with connection.begin():
                for count, (index, row) in zip(range(data_df.shape[0]), data_df.iterrows()):
                    values = tuple([value for value in row])
                    try:
                        insert = myTable.insert().values(values)
                        result = connection.execute(insert)
                    except IntegrityError as err:
                        tb = err.__traceback__
                        LOGGER.error("DB error: {0}", err)
                        raise DBToolException(err).with_traceback(tb)
                    except Exception as ex:
                        tb = ex.__traceback__
                        LOGGER.error('Failed: executing insert on table = {}', db_table)
                        raise DBToolException(ex).with_traceback(tb)

    def get_table_columns(self, db_table):
        """
        Fetches the specified tables' columns from the underlying DB in the right order.
        Parameters:
            db_table(str): the underlying DB table name holding the data - a valid table name must be specified
        Returns:
            list: list of column names
        """
        db_entities = sa.Table(db_table, sa.MetaData(), quote=False, autoload=True, autoload_with=self.sql_engine_)
        tb_cols = list(db_entities.columns.keys())
        return tb_cols

    @track_duration(name='bulk_insert')
    @debug_func(enable_debug=True, prefix='bulk_insert')
    def bulk_insert(self, data_df: pd.DataFrame, db_table: str, primary_keys: list, truncate=False):
        """
        Use as direct connection to database to insert data, especially for large inserts.
        Expects the dataframe to be converted to either a single list (for one row), or list of list (for multiple rows).
        Can either append to table (default) or if truncate=True, replace existing.
        @see https://stackoverflow.com/a/66770340/13979188
        Parameters:
            data_df(DataFrame): the dataframe with the data potentially to persist
            db_table(str): the underlying DB table name holding the data; the default is empty string
            primary_keys(list): use the list of columns specified as the primary key columns to identify and ignore duplicates
            truncate(bool): true indicates that the underlying table's contents will be replaced or appended otherwise
        """
        LOGGER.debug('Executing bulk insert query against: {} {}', db_table, '...')
        temp_table=True # The temporary table is always used for efficiency and to account for duplicates.
        ignore_duplicates=True # assumes that primary key columns are specified
        if primary_keys is None:
            ignore_duplicates = False # exceptions will be raised if data already exist in underlying data table
        else:
            assert primary_keys is not None
            assert len(primary_keys) > 0
        try:
            with self.__get_connection() as connection:
                # the default mysql+pyodbc dialect is UTF-16. To eliminate most of the failures
                # the fix is to explicitly set the encoding after the Connection object is created
                # connection.begin()
                cursor = None
                try:
                    cursor = connection.connection.cursor()
                except:
                    cursor = connection.cursor()
                #cursor.fast_executemany = True
                tt = False
                qm = ':'
                original_columns = data_df.columns
                val_lst = data_df.values.tolist()
                if isinstance(val_lst[0], list):
                    rows = len(val_lst)
                else:
                    rows = 1
                    val_lst = [val_lst,] # the last comma is necessary for MySQL Dialect--> @see https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html#:~:text=executemany()%20Method,-Syntax%3A%20cursor.&text=This%20method%20prepares%20a%20database,found%20in%20the%20sequence%20seq_of_params%20.&text=In%20Python%2C%20a%20tuple%20containing,value%20must%20include%20a%20comma.
                val_holders = tuple([f"{qm}{col}" for col in original_columns])
                val_holders = ', '.join(val_holders)
                inert_cols = ', '.join(original_columns)
                is_mysql_db = False
                try:
                    # if truncating just do it here
                    if truncate:
                        stmt = f"TRUNCATE TABLE {db_table}"
                        cursor.execute(stmt)
                    if temp_table:
                        # assume NOT mysql as default
                        # clear any such temporary table that may have been left behind
                        try:
                            underlying_table = f"##{db_table}"
                            stmt = f"DROP TABLE IF EXISTS {underlying_table}"
                            cursor.execute(stmt)
                        except Exception as err:
                            LOGGER.warning('No non-mysql dangling temporary table: {} {}', underlying_table, 'to clear')
                            try:
                                underlying_table = f"tmp_{db_table}"
                                stmt = f"DROP TABLE IF EXISTS {underlying_table}"
                                cursor.execute(stmt)
                            except Exception as mysqlerr:
                                LOGGER.warning('No mysql dangling temporary table: {} {}', underlying_table, 'to clear')
                        finally:
                            # reset temp table name
                            underlying_table = f"##{db_table}"
                        # create a temp table with same schema
                        start_time = time.time()
                        stmt = f"SELECT * INTO {underlying_table} FROM {db_table} WHERE 1=0"
                        LOGGER.debug('Attempting to creating temporary table: {}', stmt)
                        try:
                            cursor.execute(stmt)
                        except Exception as mysqlDbErr:
                            LOGGER.error('Failed to create temporary table: {}', underlying_table)
                            # rename as mysql temp table variant
                            underlying_table = f"tmp_{db_table}"
                            stmt = f"CREATE TEMPORARY TABLE {underlying_table} SELECT * FROM {db_table} LIMIT 0"
                            LOGGER.debug('Attempting to creating mysql temporary table: {}', stmt)
                            cursor.execute(stmt)
                            is_mysql_db = True
                        # set flag to indicate temp table was used
                        tt = True
                    else:
                        start_time = time.time()
                    # insert into either existing table or newly created temp table
                    val_lst = [{col: row_val for col, row_val in zip(original_columns, val)} for val in val_lst]
                    stmt = f"INSERT INTO {underlying_table} ({inert_cols}) VALUES ({val_holders})"
                    LOGGER.debug('Statement: {}', stmt)
                    #LOGGER.debug('Values:{}', val_lst[:5])
                    connection.execute(text(stmt), val_lst)
                    if tt:
                        # remove temp moniker and insert from temp table
                        dest_table = db_table
                        if not ignore_duplicates:
                            stmt = f"INSERT INTO {dest_table} ({inert_cols}) VALUES (SELECT * FROM {underlying_table})"
                            cursor.execute(stmt)
                            connection.commit()
                        else:
                            duplicate_conditions = [f"{dest_table}.{column_name} = {underlying_table}.{column_name}" for column_name in primary_keys]
                            duplicate_conditions = ' AND '.join(duplicate_conditions)
                            LOGGER.debug('Key constraint: {}', duplicate_conditions)
                            stmt = f"INSERT INTO {dest_table} ({inert_cols}) SELECT * FROM {underlying_table} WHERE NOT EXISTS (SELECT * FROM {dest_table} WHERE {duplicate_conditions})"
                            LOGGER.debug('Statement: {}', stmt)
                            cursor.execute(stmt)
                            connection.commit()
                        LOGGER.debug('Temp table used!')
                        LOGGER.debug(f"{rows} rows inserted into the {dest_table} table in {time.time() - start_time} seconds")
                    else:
                        LOGGER.debug('No temp table used!')
                        LOGGER.debug(
                            f"{rows} rows inserted into the {underlying_table} table in {time.time() - start_time} seconds")
                except Exception as err:
                    LOGGER.error('Failure: {}', err)
                    connection.rollback()
                    raise
                finally:
                    try:
                        cursor.close()
                        connection.close()
                    except:
                        pass
        except DBToolException as err:
            tb = err.__traceback__
            LOGGER.error('Failed executing bulk insert: {}', err)
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failed: executing bulk_insert to table: {} - {}', db_table, err)
            raise DBToolException(err).with_traceback(tb)
        finally:
            LOGGER.debug('Completed attempt of execute bulk_insert to table: {}', db_table)

@singleton
class DBToolFactory(object):
    ds_configs__ = {}
    db_tools__ = {}

    def __new__(cls, *args, **kwargs):
        """
        Creates an instance of the datasource factory
        """
        return super().__new__(cls)

    def __init__(self, ds_configs: dict, *args, **kwargs):
        """
        Initialize the DBTool factory using the loaded datasource configuration or properties.
        :param ds_configs: list of datasource configuration or properties previously loaded
        :type ds_configs:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        if ds_configs is None:
            LOGGER.debug('Database configurations must be provided via a ds-config.properties')
            raise DBToolException('Database configurations must be provided via a ds-config.properties')
        ds_keys = ds_configs.keys()
        for key in ds_keys:
            DBToolFactory.ds_configs__[key] = ds_configs.get(key)

    def __str__(self):
        info = 'DBToolFactory'
        LOGGER.debug(info)
        return info

    def get_tool(self, ds_key, verbose=False) -> DBTool:
        """
        Returns a DBTool instance for the specified DB key string. If tool was not yet configured,
        it is configured and ready for use before the instance is returned.
        Parameters:
            ds_key(str): the key string matching a specific datasource configuration in the ds-config.properties file
            verbose(bool): enables underlying statements to be printed or not
        Returns:
            DBTool: configured and ready instance of the requested tool.
        """
        if (ds_key is None) or ('' == ds_key):
            LOGGER.debug('A valid, non-empty datasource key matching a valid datasource configuration must be provided')
            raise DBToolException('A valid, non-empty datasource key matching a valid datasource configuration must be provided')
        try:
            rel_ds_config = DBToolFactory.ds_configs__.get(ds_key)
            if ds_key not in self.db_tools__.keys():
                self.db_tools__[ds_key] = DBTool(rel_ds_config, verbose=verbose)
            return self.db_tools__.get(ds_key)
        except DBToolException as err:
            tb = err.__traceback__
            LOGGER.error('Failed executing get_tool for datasource key: {} - {}', ds_key, err)
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('DBToolFactory encountered a problem configuring datasource with key = {}, {}', ds_key, err)
            raise DBToolException(err).with_traceback(tb)


@singleton
class DSWrapper(object):
    ds_props__ = None
    dbtool_factory__ = None

    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created,
        or else returns the previous singleton object
        """
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        Initializes the properties utility.
        """
        # log message on completion
        LOGGER.debug('Preparing DSWrapper ...')
        DSWrapper.ds_props__ = APP_PROPS.load_custom_properties(DEFAULT_DS_CONFIG)
        assert DSWrapper.ds_props__ is not None, 'Failure with processing datasource config file (ds-config.properties'
        def get_ds_properties(prop_key=None):
            """
            Parameters:
                prop_key(str): the full property name, as in the properties file, for which a value is required
            Returns:
                dict(str): a dict of string key-value pairs based on the specified key; the default is None.
            """
            if prop_key is None:
                return None
            prop_value = DSWrapper.ds_props__.get(prop_key)
            if prop_value is None:
                return None
            prop_value = prop_value.data
            properties = eval(prop_value)
            return properties
        loaded_configs = get_ds_properties('project.ds.supported')
        LOGGER.debug('Found config details for the following DBs = {}', loaded_configs.keys())
        db_configs = {}
        for key in loaded_configs.keys():
            db_info = loaded_configs.get(key)
            assert db_info is not None, 'There may be an issue with the datasource configuration or properties file'
            # setup the DB parameters
            # see: https://stackoverflow.com/questions/15784357/sqlalchemy-setting-mysql-charset-as-create-engine-argument
            # see: https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html
            direct_conn: bool = bool(eval(str(db_info.get('direct_conn'))))
            verbose: bool = bool(eval(str(db_info.get('verbose'))))
            timeout = db_info.get('timeout')
            timeout = '0' if (timeout is None) or ('' == timeout) else str(timeout)
            db_config = {'drivername': db_info.get('drivername'), 'host': db_info.get('db_server'), 'port': db_info.get('db_port'),
                         'username'  : db_info.get('username'), 'password': quote_plus(db_info.get('password')),
                         'database'  : db_info.get('db_name'), }
            if 'postgres' in db_info.get('drivername'):
                if db_info.get('encoding') is not None:
                    db_config['query'] = {'client_encoding': db_info.get('encoding'), 'verbose': str(verbose), 'timeout': timeout, }
                else:
                    db_config['query'] = {'verbose': str(verbose), 'timeout': timeout, }
            elif 'pyodbc' in db_info.get('drivername'):
                if db_info.get('encoding') is not None:
                    db_config['query'] = {'encoding': db_info.get('encoding'), 'driver': db_info.get('db_driver'), 'verbose': str(verbose), 'timeout': timeout, }
                else:
                    db_config['query'] = {'driver': db_info.get('db_driver'),
                                          'verbose' : str(verbose), 'timeout': timeout, }
                if db_info.get('MULTI_HOST') is not None:
                    db_config['connect_args'] = {'MULTI_HOST': db_info.get('MULTI_HOST'), }
            else:
                if db_info.get('encoding') is not None:
                    db_config['query'] = {'encoding': db_info.get('encoding'), 'driver': db_info.get('db_driver'),
                                          'timeout' : timeout, 'direct_conn': str(direct_conn),
                                          'verbose' : str(verbose), }
                else:
                    db_config['query'] = {'driver': db_info.get('db_driver'),
                                          'timeout' : timeout, 'direct_conn': str(direct_conn),
                                          'verbose' : str(verbose), }
            db_configs[key] = db_config
        self.dbtool_factory__ = DBToolFactory(db_configs)

    def __str__(self):
        info = 'DSWrapper'
        LOGGER.debug(info)
        return info

    def get_db_tool(self, ds_key='mysql', verbose: bool=False):
        """
        Returns the database interaction management utility with the specified DB key string.
        Parameters:
            ds_key(str): a string indicating the database type configured in the db_config.xlsx if one exists
            verbose(bool):
        Returns:
            DBTool
        """
        return self.dbtool_factory__.get_tool(ds_key, verbose=verbose)

    '''
    Clears the underlying datasource, which could be a table.
    '''

    @track_duration(name='clear_ds')
    @debug_func(enable_debug=True, prefix='clear_ds')
    def clear_datasource(self, ds_config=None):
        if ds_config is None:
            msg = 'The datasource wrapper configuration must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        # fetch all required properties from the ds_config dict
        ds_namespace = ds_config.get('ds_namespace')
        if ds_namespace is None:
            msg = 'An appropriate project namespace must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        # the assumption here is that it is a DB table that needs clearing
        ds_key = ds_config.get('ds_key')
        if ds_key is None:
            msg = 'An appropriate DB key must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        db_table = ds_config.get('db_table')
        if db_table is None:
            msg = 'An appropriate DB table name must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        replace_prop_key = 'db.to_tables.replace.' + ds_namespace + '.' + db_table
        delete_prop_key = 'db.to_table.delete.' + ds_namespace + '.' + db_table
        LOGGER.debug('Checking set flag = {}', replace_prop_key)
        replace = APP_PROPS.get_bol(replace_prop_key)
        LOGGER.debug('Checking filtered by parts = {}', delete_prop_key)
        delete_subset = APP_PROPS.get_properties(delete_prop_key)
        db_tool = self.get_db_tool(ds_key=ds_key)
        if delete_subset is not None:
            filter_by = [prop_key + '=' + delete_subset.get(prop_key) for prop_key in delete_subset]
            filter_by = ' AND '.join(filter_by)
            filter_by = f"WHERE {filter_by}"
            LOGGER.debug('Delete data from = {}', db_table, 'filtered by = {}', filter_by)
            try:
                db_tool.delete(db_table=db_table, filter_by=filter_by)
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('Failure: {}', err)
                raise DSWrapperException(err).with_traceback(tb)
        elif replace:
            LOGGER.debug('Truncating underlying db = {}, table {}', ds_key, db_table)
            try:
                db_tool.truncate(db_table=db_table)
            except Exception as err:
                tb = err.__traceback__
                LOGGER.error('Failure: {}', err)
                raise DSWrapperException(err).with_traceback(tb)

    '''
    Reads the underlying DB table.
    '''

    @track_duration(name='__read_ds')
    @debug_func(enable_debug=True, prefix='__read_ds')
    def __read_from_db(self, ds_config=None, chunksize=None, force_parallelize=False):
        if ds_config is None:
            msg = 'The datasource wrapper configuration must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        ds_key = ds_config.get('ds_key')
        if ds_key is None:
            msg = 'An appropriate DB key must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        db_table = ds_config.get('db_table')
        query_string = ds_config.get('query_string')
        LOGGER.debug('Query string: {}', query_string)
        if db_table is None:
            if query_string is None:
                msg = 'An appropriate DB table name or query string must be specified'
                LOGGER.debug(msg)
                raise DSWrapperException(msg)
        db_tool = self.get_db_tool(ds_key=ds_key)
        LOGGER.debug('Reading from table: {}', db_table)
        try:
            if chunksize is None:
                if query_string is None:
                    data_df = db_tool.read_db_table(db_table=db_table, force_parallelize=force_parallelize)
                else:
                    data_df = db_tool.read_from_db(query_string=query_string, force_parallelize=force_parallelize)
            else:
                index_col = ds_config.get('index_col')
                index_col = index_col if index_col is not None else None
                if index_col is not None:
                    LOGGER.debug('Index columns {}', index_col)
                if query_string is None:
                    data_df = db_tool.read_chunked_table(db_table=db_table, chunksize=chunksize, index_col=index_col,
                                                         force_parallelize=force_parallelize)
                else:
                    data_df = db_tool.read_chunked_table(query_string=query_string, index_col=index_col,
                                                         chunksize=chunksize,
                                                         force_parallelize=force_parallelize)
        except DBToolException as err:
            tb = err.__traceback__
            LOGGER.error('Failure: {}', err)
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure: {}', err)
            raise DSWrapperException(err).with_traceback(tb)
        return data_df

    '''
    Reads the specified datasource
    '''

    @track_duration(name='read_from_ds')
    @debug_func(enable_debug=True, prefix='read_from_ds')
    def read_from_datasource(self, ds_config=None, chunksize=None, rename_cols=None, parse_date_cols=None,
                             gps_cols=None, dropna_cols=None, input_types=None, force_parallelize=False):
        """

        :param ds_config: the datasource configuration
        :param chunksize: the chunksize or number of rows to be read at a time or to break up the read by
        :param rename_cols: a dict of any columns that need to be renamed
        :param parse_date_cols: any date columns that may need to be parsed properly as dates
        :param gps_cols: any columns containing GPS coordinates that may need parsing and fixing
        :param dropna_cols: any columns for which null-containing rows should be dropped
        :param input_types: a dict specifying any input data types that need to be enforced
        :param force_parallelize: by default large CSVs are parallelized; set this flag to do the same for Excels
        :return: a DataFrame of the entire data or a list of DataFrames if read in chunks
        """
        if ds_config is None:
            msg = 'The datasource wrapper configuration must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        data_file = ds_config.get('data_file')
        col_mappings = ds_config.get('col_mappings')
        if data_file is not None:
            path_to_data_file = data_file
            # then the underlying datasource is an excel/csv file
            is_csv = ds_config.get('is_csv')
            is_csv = False if is_csv is None else is_csv
            is_raw = ds_config.get('is_raw')
            is_raw = False if is_raw is None else is_raw
            rel_cols = ds_config.get('rel_cols')
            unique_key = ds_config.get('unique_key')
            if unique_key is None:
                if not is_raw:
                    msg = 'A unique subset of columns or keys must be specified for the datasource'
                    LOGGER.debug(msg)
                    raise DSWrapperException(msg)
            LOGGER.debug('Path to prepared dataset: {}', path_to_data_file)
            cur_date = dt.date.today()
            if chunksize is None:
                if is_csv:
                    data_df = pd.read_csv(path_to_data_file)
                else:
                    data_df = pd.read_excel(path_to_data_file, dtype=input_types)
                try:
                    data_df.drop_duplicates(subset=unique_key, inplace=True)
                except Exception as ignore:
                    if is_raw:
                        msg = 'WARNING only: Processing a raw-templated file --> ' + str(ignore)
                        LOGGER.warning(msg)
                    else:
                        raise DSWrapperException(ignore)
                # do any additional processing
                if dropna_cols is not None:
                    data_df.dropna(subset=dropna_cols, inplace=True)
                if parse_date_cols is not None:
                    for date_col in parse_date_cols:
                        data_df[date_col] = pd.to_datetime(data_df[date_col]).dt.date
                if rename_cols is not None:
                    LOGGER.debug('Renaming columns: {}', rename_cols)
                    data_df.rename(columns=rename_cols, inplace=True)
                # add the obligatory modified date, that may or may not be used
                data_df['modified_date'] = dt.date.today()
                if rel_cols is None:
                    pass
                else:
                    if not is_raw:
                        data_df = data_df[rel_cols]
                data_df.fillna(0, inplace=True)
                if col_mappings is not None:
                    if isinstance(data_df, list):
                        for item_df in data_df:
                            item_df.rename(columns=col_mappings, inplace=True)
                    else:
                        data_df.rename(columns=col_mappings, inplace=True)
                return data_df
            else:
                data_read = self.read_file_in_chunks(path_to_data_file, is_raw=is_raw, chunksize=chunksize,
                                                     is_csv=is_csv,
                                                     rel_cols=rel_cols, input_types=input_types,
                                                     force_parallelize=force_parallelize)
                data_dfs = []
                for data_df in data_read:
                    data_df = data_df.copy()
                    try:
                        data_df.drop_duplicates(subset=unique_key, inplace=True)
                    except Exception as err:
                        if is_raw:
                            LOGGER.warning('WARNING: Processing a raw-templated file: {}', err)
                        else:
                            raise DSWrapperException(err)
                    # do any additional processing
                    if dropna_cols is not None:
                        data_df.dropna(subset=dropna_cols, inplace=True)
                    if parse_date_cols is not None:
                        for date_col in parse_date_cols:
                            data_df[date_col] = pd.to_datetime(data_df[date_col]).dt.date
                    if rename_cols is not None:
                        LOGGER.debug('Renaming columns: {}', rename_cols)
                        data_df.rename(columns=rename_cols, inplace=True)
                    # add the obligatory modified date, that may or may not be used
                    data_df['modified_date'] = cur_date
                    if rel_cols is None:
                        pass
                    else:
                        data_df = data_df[rel_cols]
                    data_df.fillna(0, inplace=True)
                    data_dfs.append(data_df)
                    if col_mappings is not None:
                        if isinstance(data_df, list):
                            for item_df in data_df:
                                item_df.rename(columns=col_mappings, inplace=True)
                        else:
                            data_df.rename(columns=col_mappings, inplace=True)
                return data_dfs
        else:
            data_df = self.__read_from_db(ds_config=ds_config, chunksize=chunksize, force_parallelize=force_parallelize)
            if col_mappings is not None:
                if isinstance(data_df, list):
                    for item_df in data_df:
                        item_df.rename(columns=col_mappings, inplace=True)
                else:
                    data_df.rename(columns=col_mappings, inplace=True)
            return data_df

    '''
    Apply the specified data to the underlying datasource.
    '''

    @track_duration(name='apply_to_ds')
    @debug_func(enable_debug=True, prefix='apply_to_ds')
    def apply_to_datasource(self, data_df, ds_config=None):
        if ds_config is None:
            msg = 'The datasource wrapper configuration must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        # save backup if indicated
        data_file_specified = ds_config.get('data_file')
        if data_file_specified is not None:
            try:
                self.__save_to_datafile(data_df, ds_config=ds_config)
            except Exception as ignore:
                LOGGER.warning('Warning: {}', ignore)
        ds_key = ds_config.get('ds_key')
        if ds_key is None:
            if data_file_specified is None:
                msg = 'An appropriate DB key must be specified'
                LOGGER.debug(msg)
                raise DSWrapperException(msg)
            else:
                msg = 'WARNING: A DB key not specified - Only a backup data file may have been saved'
                LOGGER.debug(msg)
            return
        db_table = ds_config.get('db_table')
        if db_table is None:
            msg = 'An appropriate DB table name must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        unique_key = ds_config.get('unique_key')
        if unique_key is None:
            msg = 'A unique subset of columns or keys must be specified for the datasource'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        db_tool = self.get_db_tool(ds_key=ds_key)
        try:
            # then apply the new data to the underlying table
            rel_cols = ds_config.get('rel_cols')
            if rel_cols is not None:
                LOGGER.debug('Required columns: {}', rel_cols)
                LOGGER.debug('Shape: {}', data_df.shape)
                data_df = data_df[rel_cols]
            db_tool.bulk_insert(data_df, db_table=db_table, primary_keys=unique_key)
        except DBToolException as err:
            tb = err.__traceback__
            LOGGER.error('Failed executing apply_to_datasource: {}', err)
            raise
        except Exception as err:
            tb = err.__traceback__
            LOGGER.error('Failure: {}', err)
            raise DSWrapperException(err).with_traceback(tb)

    '''
    Reads an Excel-based (i.e., excel or csv file) in chunks.
    '''

    @track_duration(name='read_in_chunks')
    @debug_func(enable_debug=True, prefix='read_in_chunks')
    def read_file_in_chunks(self, path_to_data_file=None, is_raw=False, chunksize=None, is_csv=False, input_types=None,
                            rel_cols=None, force_parallelize=False):
        """
        Reads the specified Excel file in chunks, presuming that the file is large.
        See also: https://www.giacomodebidda.com/posts/reading-large-excel-files-with-pandas/
        :param path_to_data_file: the path to the Excel file
        :param is_raw:
        :param chunksize: the number of rows to read at a time
        :param is_csv: flag whether it is a csv file
        :param input_types: a dict specifying any input data types that need to be enforced
        :param rel_cols: the specific list of columns to be read
        :param force_parallelize: by default reading CSVs is parallelized way; set this to True to force Excels as well
        :return: a list of DataFrames
        """
        if path_to_data_file is None:
            LOGGER.debug('A valid path to the data file must be provided')
            raise DSWrapperException('A valid path to the data file must be provided')
        if chunksize is None:
            LOGGER.debug('A valid chunkcise must be provided')
            raise DSWrapperException('A valid chunkcise must be provided')
        LOGGER.debug('Input types: {}', input_types)
        if is_csv:
            dd_df = self.read_large_csv(path_to_data_file, input_types=input_types)
            # break the dataframe into appropriate chunks
            # see also: https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe
            chunks = [dd_df[i:i + chunksize].copy() for i in range(0, dd_df.shape[0], chunksize)]
            LOGGER.debug('No. of chunks: {}', len(chunks))
            return chunks
        else:
            # then it is Excel
            xl = pd.ExcelFile(path_to_data_file)
            sheet_name = xl.sheet_names[0]
            if force_parallelize:
                # make sure sheet_name=0; any other value may cause underlying result to a a dict of DataFrame
                # but this is not effectively handled at present
                return self.read_large_excel(path_to_data_file, is_raw=is_raw, rel_cols=rel_cols,
                                             input_types=input_types, sheet_name=0)
            else:
                return DSWrapper.__read_excel_chunks(path_to_data_file, sheet_name=sheet_name,
                                                     chunksize=chunksize, input_types=input_types)

    @staticmethod
    def __read_excel_chunks(path_to_data_file, sheet_name=0, chunksize=None, input_types=None):
        """
        Read the specified excel and return a list of DataFrames
        :param path_to_data_file: path_to_data_file: the path to the Excel file
        :param sheet_name: the specific worksheet; the default is 0
        :param chunksize: the number of rows to read at a time
        :param input_types: a dict specifying any input data types that need to be enforced
        :return:
        """
        chunks = []
        i_chunk = 0
        # The first row is the header. We have already read it, so we skip it.
        skiprows = 1
        df_header = pd.read_excel(path_to_data_file, sheet_name=sheet_name, nrows=1)
        while True:
            df_chunk = pd.read_excel(path_to_data_file, sheet_name=sheet_name, nrows=chunksize, skiprows=skiprows,
                                     header=None, converters=input_types)
            skiprows += chunksize
            # When there is no data, we know we can break out of the loop.
            if not df_chunk.shape[0]:
                break
            else:
                # Rename the columns to concatenate the chunks with the header.
                columns = {i: col for i, col in enumerate(df_header.columns.tolist())}
                df_chunk.rename(columns=columns, inplace=True)
                msg = f"Read - chunk {i_chunk} ({df_chunk.shape[0]} rows)"
                LOGGER.debug(msg)
                # deal with any specified data types
                if input_types is not None:
                    for col in input_types:
                        df_chunk[col] = df_chunk[col].astype(input_types.get(col))
                chunks.append(df_chunk)
            i_chunk += 1
        LOGGER.debug('Number of chunks: {}', len(chunks))
        return chunks

    '''
    Reads a large CSV file efficiently.
    '''

    @track_duration(name='read_large_csv')
    @debug_func(enable_debug=True, prefix='read_large_csv')
    def read_large_csv(self, path_to_data_file=None, input_types=None):
        """
        Reads a large csv file efficiently (see: https://tutorial.dask.org/01_dataframe.html)
        :param path_to_data_file: the path to the Excel file
        :param input_types: a dict specifying any input data types that need to be enforced
        :return: a DataFrames
        """
        if path_to_data_file is None:
            return None
        dd_df = dd.read_csv(path_to_data_file, dtype=input_types)
        return dd_df.compute()

    '''
    Reads a large Excel file efficiently.
    '''

    @track_duration(name='read_large_xl')
    @debug_func(enable_debug=True, prefix='read_large_xl')
    def read_large_excel(self, path_to_data_file=None, is_raw=False, rel_cols=None, input_types=None, sheet_name=0):
        """
        Reads a large Excel file efficiently (see: https://tutorial.dask.org/01_dataframe.html; and
        https://stackoverflow.com/questions/44654906/parallel-excel-sheet-read-from-dask)
        :param path_to_data_file: the path to the Excel file
        :param is_raw: indicates whether a raw untemplated file is being read (only templated files have defined rel_cols)
        :param rel_cols: the required and matching columns of the dataset
        :param input_types: a dict specifying any input data types that need to be enforced
        :param sheet_name: specified sheet if needed; the default is 0
        :return: a list containing a DataFrame
        """
        if path_to_data_file is None:
            LOGGER.debug('A path to the excel file is required')
            raise DSWrapperException('A path to the excel file is required')
        if rel_cols is None:
            if not is_raw:
                LOGGER.debug('The relevant columns of the dataset required')
                raise DSWrapperException('The relevant columns of the dataset required')
            else:
                LOGGER.debug('No defined relevant columns specified')
        if input_types is None:
            LOGGER.debug('The relevant columns dtypes of the dataset may be required')
        # do the Dask stuff
        # avoid passing a sheet_name because it can lead to a dict of DataFrames in the delayed read_excel
        # which is not adequately catered for yet
        parts = [delayed(DSWrapper.__delayed_read_excel)(path_to_data_file, input_types=input_types, rel_cols=rel_cols,
                                                         sheet_name=sheet_name)]
        meta_info = parts[0].compute()
        data_df = dd.from_delayed(parts, meta=meta_info).compute()
        return [data_df]

    @staticmethod
    def __delayed_read_excel(path_to_data_file, input_types=None, rel_cols=None, sheet_name=0):
        """
        Uses Dask to do a delayed but efficient read of an underlying excel file
        :param path_to_data_file: the path to the excel file
        :param input_types: a dict with the column types
        :param rel_cols: the required columns to be read
        :param sheet_name: any specific worksheet name; the default is 0 to ensure a DataFrame is return by Pandas and not a dict
        :return:
        """
        LOGGER.debug('Doing parallelized excel read ...')
        LOGGER.debug('Path to file: {}', path_to_data_file)
        LOGGER.debug('Input types: {}', input_types)
        LOGGER.debug('Relevant columns: {}', rel_cols)
        # do the read
        data_read = pd.read_excel(path_to_data_file, dtype=input_types, sheet_name=sheet_name)
        data_read['modified_date'] = dt.date.today()
        if rel_cols is not None:
            data_read = data_read[rel_cols]
        return data_read

    @track_duration(name='__save_to_file')
    @debug_func(enable_debug=True, prefix='__save_to_file')
    def __save_to_datafile(self, data_df, ds_config=None):
        if data_df is None:
            msg = 'The dataframe containing required data must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        if ds_config is None:
            msg = 'The datasource wrapper configuration must be specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        path_to_data_file = ds_config.get('data_file')
        if path_to_data_file is None:
            msg = 'Path to datafile not specified'
            LOGGER.debug(msg)
            raise DSWrapperException(msg)
        # recursively create any folders
        sub_path = os.path.dirname(path_to_data_file)
        try:
            os.makedirs(sub_path, exist_ok=True)
        except Exception as ignoreex:
            LOGGER.debug('WARNING: Failed attempt to recursively create folders: {}, {}', sub_path, ignoreex)
        # date stamp the file
        path_to_data_file = datestamp(path_to_data_file)
        parallelize = ds_config.get('force_parallelize')
        is_csv = ds_config.get('is_csv')
        if is_csv:
            try:
                data_df.to_csv(path_to_data_file, index=False)
                LOGGER.debug('Saved data to: {}', path_to_data_file)
            except Exception as err:
                LOGGER.error('FAILED attempt to save data to csvfile: {}, {}', path_to_data_file, err)
                raise DSWrapperException(err)
        else:
            try:
                data_df.to_excel(path_to_data_file, index=False)
                LOGGER.debug('Saved data to: {}', path_to_data_file)
            except Exception as err:
                LOGGER.error('FAILED attempt to save data to excelfile: {}', path_to_data_file, err)
                raise DSWrapperException(err)