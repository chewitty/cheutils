# cheutils

A set of basic reusable utilities and tools to facilitate quickly getting up and going on any machine learning project.

### Features

- model_options: methods such as get_estimator to get a handle on a configured estimator with a specified parameter dictionary or get_default_grid to get the configured hyperparameter grid
- model_builder: methods for building and executing ML pipeline steps e.g., params_optimization etc.
- project_tree: methods for accessing the project tree - e.g., get_data_dir() for accessing the configured data and get_output_dir() for the output folders, loading and savings Excel and CSV.
- common_utils: methods to support common programming tasks, such as labeling (e.g., `label(file_name, label='some_label')`) or tagging and date-stamping files (e.g., `datestamp(file_name, fmt='%Y-%m-%d')`).
- propertiesutil: utility for managing properties files or project configuration, based on jproperties. The application configuration is expected to be available in a file named app-config.properties, which can be placed anywhere in the project root or any subfolder thereafter.
- decorator_debug, decorator_timer, and decorator_singleton: decorators for enabling logging and method timers; as well as a singleton decorator
- datasource_utils: utility for managing datasource configuration or properties file (ds-config.properties) and offers a set of generic datasource access methods.
### Usage
You import the `cheutils` module as per usual:
```
import cheutils
```
The following provide access to the properties file, usually expected to be named "app-config.properties" and typically found in the project data folder or anywhere either in the project root or any other subfolder
```
APP_PROPS = cheutils.AppProperties() # to load the app-config.properties file
```
Thereafter, you can read any properties using various methods such as:
```
DATA_DIR = APP_PROPS.get('project.data.dir')
```
You can also retrieve the path to the data folder, which is under the project root as follows:
```
cheutils.get_data_dir()  # returns the path to the project data folder, which is always interpreted relative to the project root
```
You can retrieve other properties as follows:
```
VALUES_LIST = APP_PROPS.get_list('some.configured.list') # e.g., some.configured.list=[1, 2, 3] or ['1', '2', '3']
VALUES_DIC = APP_PROPS.get_dic_properties('some.configured.dict') # e.g., some.configured.dict={'val1': 10, 'val2': 'value'}
BOL_VAL = APP_PROPS.get_bol('some.configured.bol') # e.g., some.configured.bol=True
```
You also have access to the LOGGER - you can simply call `LOGGER.debug()` in a similar way to you will when using loguru or standard logging 
calling `set_prefix()` on the LOGGER instance ensures the log messages are scoped to that context thereafter, 
which can be helpful when reviewing the generated log file (`app-log.log`) - the default prefix is "app-log".

You can get a handle to an application logger as follows:
```
LOGGER = cheutils.LOGGER.get_logger()
```
You can set the logger prefix as follows:
```
LOGGER.set_prefix(prefix='my_project')
```
The `model_options` currently supports any configured estimator (see, the xgb_boost example below for how to configure any estimator).
You can configure any of the models for your project with an entry in the app-config.properties as follows:
```
model.active.model_option=xgb_boost # with default parameters
```
You can get a handle to the corresponding estimator as follows:
```
estimator = cheutils.get_estimator(model_option='xgb_boost')
```
You can also configure the following property for example:
```
model.param_grids.xgb_boost={'learning_rate': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 10}, 'subsample': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 10}, 'min_child_weight': {'type': float, 'start': 0.1, 'end': 1.0, 'num': 10}, 'n_estimators': {'type': int, 'start': 10, 'end': 400, 'num': 10}, 'max_depth': {'type': int, 'start': 3, 'end': 17, 'num': 5}, 'colsample_bytree': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, 'gamma': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, 'reg_alpha': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, }
```
Thereafter, you can do the following:
```
estimator = cheutils.get_estimator(**get_params(model_option='xgb_boost'))
```
Thereafter, you can simply fit the model as follows per usual:
```
estimator.fit(X_train, y_train)
```
Given a default model parameter configuration (usually in the properties file), you can generate a promising parameter grid using RandomSearchCV as in the following line. Note that, the pipeline can either be an sklearn pipeline or an estimator. 
The general idea is that, to avoid worrying about trying to figure out the optimal set of hyperparameter values for a given estimator, you can do that automatically, by 
adopting a two-step coarse-to-fine search, where you configure a broad hyperparameter space or grid based on the estimator's most important or impactful hyperparameters, and the use a random search to find a set of promising hyperparameters that 
you can use to conduct a finer hyperparameter space search using other algorithms such as bayesean optimization (e.g., hyperopt or Scikit-Optimize, etc.)
```
promising_grid = cheutils.promising_params_grid(pipeline, X_train, y_train, grid_resolution=3, prefix='model_prefix')
```
You can run hyperparameter optimization or tuning as follows (assuming you enabled cross-validation in your configuration or app-conf.properties - e.g., with an entry such as `model.cross_val.num_folds=3`), if using hyperopt; and if you are running Mlflow experiments and logging, you could also pass an optional mlflow_exp={'log': True, 'uri': 'http://<mlflow_tracking_server>:<port>', } in the optimization call:
```
best_estimator, best_score, best_params, cv_results = cheutils.params_optimization(pipeline, X_train, y_train, promising_params_grid=promising_grid, with_narrower_grid=True, fine_search='hyperoptcv', prefix='model_prefix')
```
You can get a handle to the datasource wrapper as follows:
```
ds = DSWrapper() # it is a singleton
```
You can then read a large CSV file, leveraging `dask` as follows:
```
data_df = ds.read_large_csv(path_to_data_file=os.path.join(get_data_dir(), 'some_file.csv'))
```
Assuming you previously defined a datasource configuration in ds-config.properties, containing:
`project.ds.supported={'mysql_local': {'db_driver': 'MySQL ODBC 8.1 ANSI Driver', 'drivername': 'mysql+pyodbc', 'db_server': 'localhost', 'db_port': 3306, 'db_name': 'test_db', 'username': 'test_user', 'password': 'test_password', 'direct_conn': 0, 'timeout': 0, 'verbose': True}, }`
You could read from a configured datasource as follows:
```
ds_config = {'db_key': 'mysql_local', 'ds_namespace': 'test', 'db_table': 'some_table', 'data_file': None}
data_df = ds.read_from_datasource(ds_config=ds_config, chunksize=5000)
```
Note that, if you call `read_from_datasource()` with `data_file` set in the `ds_config` as either an Excel or CSV then it is equivalent to calling a read CSV or Excel.
There are transformers for dropping clipping data based on catagorical aggregate statistics such as mean or median values.
You can add a clipping transformer to your pipeline as follows:
```
num_cols = ['rental_rate', 'release_year', 'length', 'replacement_cost']
filter_by = 'R_rated'
clip_outliers_tf = ClipDataTransformer(rel_cols=num_cols, filterby=filter_by)
standard_pipeline_steps.append(('clip_outlier_step', clip_outliers_tf))
```
You can also include feature selection by adding the following to the pipeline:
```
feat_sel_tf = FeatureSelectionTransformer(estimator=get_estimator(model_option='xgboost'), random_state=100)
# add feature selection to pipeline
standard_pipeline_steps.append(('feat_selection_step', feat_sel_tf))
```



