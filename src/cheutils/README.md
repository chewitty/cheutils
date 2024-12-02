from movierental.notebook import model_option

# cheutils

A set of basic reusable utilities and tools to facilitate quickly getting up and going on any machine learning project.

### Features
- Managing properties files or project configuration, based on jproperties. The application configuration is expected to be available in a properties file named `app-config.properties`, which can be placed anywhere in the project root or any project subfolder.
- Convenience methods such as `get_estimator()` to get a handle on any configured estimator with a specified hyperparameters dictionary, `get_params_grid()` or `get_param_defaults()` relating to obtaining model hyperparameters in the `app-config.properties` file.
- Convenience methods for conducting hyperparameter optimization such as `params_optimization()`, `promising_params_grid()` for obtaining a set of promising hyperparameters using RandomSearchCV and a set of broadly specified or configured hyperparameters in the `app-config.properties`; a combination of `promising_params_grid()` followed by `params_optimization()` constitutes a coarse-to-fine search.
- Convenience methods for accessing the project tree folders - e.g., `get_data_dir()` for accessing the configured data and `get_output_dir()` for the output folders, `load_dataset()` for loading, `save_excel()` and `save_csv()` for savings Excel in the project output folder and CSV respectively; you can also save any plotted figure using `save_current_fig()` (note that this must be called before `plt.show()`.
- Convenience methods to support common programming tasks, such as renaming or tagging file names- e.g., `label(file_name, label='some_label')`) or tagging and date-stamping files (e.g., `datestamp(file_name, fmt='%Y-%m-%d')`).
- A debug or logging, timer, and singleton decorators - for enabling logging and method timing, as well as creating singleton instances.
- Convenience methods available via the `DSWrapper` for managing datasource configuration or properties files - e.g. `ds-config.properties` - offering a set of generic datasource access methods such as `apply_to_datasource()` to persist data to any configured datasource or `read_from_datasource()` to read data from any configured datasources.
- A set of custom `scikit-learn` transformers for preprocessing data such as `DataPrepTransformer` which can be added to a data pipeline for pre-process dataset - e.g., handling date conversions, type casting of columns, clipping data, generating special features from rows of text strings, generating calculated features, masking columns, dropping correlated or potential data leakage columns, and generating target variables from other features as needed (separet from target encoding). A `GeospatialTransformer` for generating geohash features from latitude and longitudes; a `SelectiveFunctionTransformer` and `SelectiveColumnTransformer` for selectively transforming dataframe columns; a `DateFeaturesTransformer` for generating date-related features for feature engineering, and `FeatureSelectionTransformer` for feature selection using configured estimators such as `Lasso` or `LinearRegression`
- A set of ther generic or common utilities for summarizing dataframes - e.g., using `summarize()` or to winsorize using `winsorize_it()`
- A set of convenience properties handlers to accessing generic configured properties relating to the project tree, data preparation, or model development and execution such as `ProjectTreeProperties`, `DataPrepProperties`, and `ModelProperties`. These handlers offer a convenient feature for reloading properties as needed, thereby refreshing properties without having to re-start the running VM (really only useful in development). However you may access any configured properties in the usual way via the `AppProperties` object.

### Usage
The module expects that you may wish to use a project configuration file - the default expected is `app-config.properties`. A sample such properties file may contain entries such as the following:
```
##
# Project properties
##
project.namespace=cheutils
project.root.dir=./
project.data.dir=./data/
project.output.dir=./output/
# property handlers
project.properties.proj_handler={'module_name': 'ProjectTreeProperties', 'module_package': 'cheutils', }
project.properties.data_handler={'module_name': 'DataPrepProperties', 'module_package': 'cheutils', }
project.properties.model_handler={'module_name': 'ModelProperties', 'module_package': 'cheutils', }
# SQLite DB
project.sqlite3.db=cheutils_sqlite.db
project.dataset.list=[X_train.csv, X_test.csv, y_train.csv, y_test.csv]
project.models.supported={'xgb_boost': {'module_name': 'XGBRegressor', 'module_package': 'xgboost'}, \
'random_forest': {'module_name': 'RandomForestRegressor', 'module_package': 'sklearn.ensemble'}, \
'lasso': {'module_name': 'Lasso', 'module_package': 'sklearn.linear_model'}, }
model.baseline.model_option=lasso
model.active.model_option=xgb_boost
model.active.n_iters=200
model.active.n_trials=10
model.narrow_grid.scaling_factor=0.20
model.narrow_grid.scaling_factors={'start': 0.1, 'end': 1.0, 'steps': 10}
model.find_optimal.grid_resolution=False
model.find_optimal.grid_resolution.with_cv=False
model.grid_resolutions.sample={'start': 1, 'end': 21, 'step': 1}
model.active.grid_resolution=7
model.cross_val.num_folds=3
model.active.n_jobs=-1
model.cross_val.scoring=neg_mean_squared_error
model.metric.target.objective=3.0
model.active.random_seed=100
model.active.trial_timeout=60
model.hyperopt.algos={'rand.suggest': 0.05, 'tpe.suggest': 0.75, 'anneal.suggest': 0.20, }
model.params_grid.xgb_boost={'learning_rate': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 10}, 'subsample': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 10}, 'min_child_weight': {'type': float, 'start': 0.1, 'end': 1.0, 'num': 10}, 'n_estimators': {'type': int, 'start': 10, 'end': 400, 'num': 10}, 'max_depth': {'type': int, 'start': 3, 'end': 17, 'num': 5}, 'colsample_bytree': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, 'gamma': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, 'reg_alpha': {'type': float, 'start': 0.0, 'end': 1.0, 'num': 5}, }
model.params_grid.random_forest={'min_samples_leaf': {'type': int, 'start': 1, 'end': 60, 'num': 5}, 'max_features': {'type': int, 'start': 5, 'end': 1001, 'num': 10}, 'max_depth': {'type': int, 'start': 5, 'end': 31, 'num': 6}, 'n_estimators': {'type': int, 'start': 5, 'end': 201, 'num': 10}, 'min_samples_split': {'type': int, 'start': 2, 'end': 21, 'num': 5}, 'max_leaf_nodes': {'type': int, 'start': 5, 'end': 401, 'num': 10}, }
# transformers
model.selective_column.transformers=[{'name': 'scaler_tf', 'transformer_name': 'StandardScaler', 'transformer_package': 'sklearn.preprocessing', 'transformer_params': {}, 'columns': ['col1_label', 'col2_label']}, ]
```
You import the `cheutils` module as per usual:
```python
import cheutils
```
The following provide access to the properties file, usually expected to be named "app-config.properties" and typically found in the project data folder or anywhere either in the project root or any other subfolder
```python
APP_PROPS = cheutils.AppProperties() # this automatically search for the app-config.properties file and loads it
```
During development, you may find it convenient to reload properties file changes without re-starting the VM. You can achieve that by adding the following somewhere at the top of your Jupyter notebook, for example.
```python
APP_PROPS.reload() # this automatically notifies and registered properties handlers to be reloaded
```
You can access any properties using various methods such as:
```python
DATA_DIR = APP_PROPS.get('project.data.dir')
```
You can also retrieve the path to the data folder, which is under the project root as follows:

```python
cheutils.get_data_dir()  # returns the path to the project data folder, which is always interpreted relative to the project root
APP_PROPS.get_subscriber('proj_handler').get_data_dir()  # does the same as above
```
You can retrieve other properties as follows:
```python
VALUES_LIST = APP_PROPS.get_list('project.dataset.list') # e.g., some.configured.list=[1, 2, 3] or ['1', '2', '3']
VALUES_DIC = APP_PROPS.get_dic_properties('model.hyperopt.algos') # e.g., some.configured.dict={'val1': 10, 'val2': 'value'}
LIST_OF_DICTS = APP_PROPS.get_list_properties('model.selective_column.transformers') # e.g., configured transformers in the sample properties file above
BOL_VAL = APP_PROPS.get_bol('model.find_optimal.grid_resolution') # e.g., some.configured.bol=True
```
You access the LOGGER instance by simply calling `LOGGER.debug()` in a similar way to you will when using loguru or standard logging
```python
LOGGER = cheutils.LOGGER.get_logger()
LOGGER.info('Some info you wish to log') # or debug() etc.
```
You may also wish to change the logging context from the default, which is usually set to the configured project namespace property, by calling `set_prefix()` on the LOGGER instance ensures the log messages are scoped to that context thereafter - which can be helpful when reviewing the generated log file (`app-log.log`) - the default prefix is "app-log". You can set the logger prefix as follows:
```python
LOGGER.set_prefix(prefix='my_project')
```
The `cheutils` module currently supports any configured estimator (see, the xgb_boost example in the sample properties file for how to configure any estimator).
You can configure the active or main estimators for your project with an entry in the app-config.properties as below, but you add your own properties as well, provided the estimator has been fully configured as in the sample properties file:
```
model.active.model_option=xgb_boost # the named estimator here has already been configured with a broad set of default hyperparameters
```
You can get a handle to the corresponding estimator in your code as follows:
```python
estimator = cheutils.get_estimator(model_option='xgb_boost')
```
Thereafter, you can do the following as well, to get a non-default instance:
```python
estimator = cheutils.get_estimator(**get_params(model_option='xgb_boost'))
```
You can simply fit the model as follows per usual:
```python
estimator.fit(X_train, y_train)
```
Given a default broad estimator hyperparameter configuration (usually in the properties file), you can generate a promising parameter grid using RandomSearchCV as in the following line. Note that, the pipeline can either be an sklearn pipeline or an estimator. 
The general idea is that, to avoid worrying about trying to figure out the optimal set of hyperparameter values for a given estimator, you can do that automatically, by 
adopting a two-step coarse-to-fine search, where you configure a broad hyperparameter space or grid based on the estimator's most important or impactful hyperparameters, and the use a random search to find a set of promising hyperparameters that 
you can use to conduct a finer hyperparameter space search using other algorithms such as bayesean optimization (e.g., hyperopt or Scikit-Optimize, etc.)
```python
promising_grid = cheutils.promising_params_grid(pipeline, X_train, y_train, grid_resolution=3, prefix='baseline_model') # the prefix is not needed if not part of a model pipeline
```
You can run hyperparameter optimization or tuning as follows (assuming you enabled cross-validation in your configuration or app-conf.properties - e.g., with an entry such as `model.cross_val.num_folds=3`), if using hyperopt; and if you are running Mlflow experiments and logging, you could also pass an optional mlflow_exp={'log': True, 'uri': 'http://<mlflow_tracking_server>:<port>', } in the optimization call:
```python
best_estimator, best_score, best_params, cv_results = cheutils.params_optimization(pipeline, X_train, y_train, promising_params_grid=promising_grid, with_narrower_grid=True, fine_search='hyperoptcv', prefix='model_prefix')
```
You can get a handle to the datasource wrapper as follows:
```python
ds = DSWrapper() # it is a singleton
```
You can then read a large CSV file, leveraging `dask` as follows:
```python
data_df = ds.read_large_csv(path_to_data_file=os.path.join(cheutils.get_data_dir(), 'some_file.csv')) # where the data file is expected to be in the data sub folder of the project tree
```
Assuming you previously defined a datasource configuration such as `ds-config.properties` somewhere in the project tree or sub folder, containing:
`project.ds.supported={'mysql_local': {'db_driver': 'MySQL ODBC 8.1 ANSI Driver', 'drivername': 'mysql+pyodbc', 'db_server': 'localhost', 'db_port': 3306, 'db_name': 'test_db', 'username': 'test_user', 'password': 'test_password', 'direct_conn': 0, 'timeout': 0, 'verbose': True}, }`
You could read from a configured datasource (DB) as follows:
```python
ds_config = {'db_key': 'mysql_local', 'ds_namespace': 'test', 'db_table': 'some_table', 'data_file': None} # the data_file property is this dictionary MUST be set to None or left unset, if you wish to read a configured DB resource
data_df = ds.read_from_datasource(ds_config=ds_config, chunksize=5000)
```
Note that, if you call `read_from_datasource()` with `data_file` set in the `ds_config` as either an Excel or CSV then it is equivalent to calling a read CSV or Excel - where the data file is expected to be in the data sub folder of the project.
There are transformers for dropping clipping data based on catagorical aggregate statistics such as mean or median values.
You can add a data preprocessing transformer to your pipeline as follows:
```python
date_cols = ['rental_date']
int_cols = ['release_year', 'length', 'NC-17', 'PG', 'PG-13', 'R',
            'trailers', 'deleted_scenes', 'behind_scenes', 'commentaries', 'extra_fees']
correlated_cols = ['rental_rate_2', 'length_2', 'amount_2']
drop_missing = True
clip_data = None
exp_tf = cheutils.DataPrepTransformer(date_cols=date_cols, int_cols=int_cols, drop_missing=drop_missing, clip_data=clip_data,
                             correlated_cols=correlated_cols,
                             include_target=False,)
data_prep_pipeline_steps = [('data_prep_step', exp_tf)]
```
You can also include feature selection by adding the following to the pipeline:

```python
feat_sel_tf = cheutils.FeatureSelectionTransformer(estimator=cheutils.get_estimator(model_option='xgboost'),
                                                   cheutils.AppProperties().get_subscriber(
                                                       'model_handler').get_random_seed())
# add feature selection to pipeline
standard_pipeline_steps.append(('feat_selection_step', feat_sel_tf))
```
You can also use a configured column transformer called `SelectiveColumnTransformer`. For example, if you already have configured a list of column transformers in the `app-config.properties` such as in the sample properties file above. Then you can add it to the pipeline as below. 
The `SelectiveColumnTransformer` uses the configured property to determine the transformer(s) to add to the pipeline. The above should apply any transformations to the specified columns only.
```python
scaler_tf = cheutils.SelectiveColumnTransformer()
standard_pipeline_steps.append(('scale_feats_step', scaler_tf))
```
Ultimately, you may create a model pipeline and excute using steps similar to the following:

```python
from cheutils import get_estimator, winsorize_it
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

# assuming previous necessary steps
baseline_model = get_estimator(model_option=APP_PROPS.get('model.baseline.model_option'))
baseline_pipeline_steps = standard_pipeline_steps.copy()
baseline_pipeline_steps.append(('baseline_mdl', baseline_model))
baseline_pipeline = Pipeline(steps=baseline_pipeline_steps, verbose=True)
# potentially, using `scikit-learn`'s `TransformedTargetRegressor` to do some target encoding as well, for argument's sake
baseline_est = TransformedTargetRegressor(regressor=baseline_pipeline, func=winsorize_it, inverse_func=winsorize_it,
                                          check_inverse=False, )
baseline_est.fit(X_train, y_train)
y_train_pred = baseline_est.predict(X_train)
mse_score = mean_squared_error(y_train, y_train_pred)
r_squared = r2_score(y_train, y_train_pred)
LOGGER.debug('Training baseline mse = {:.2f}'.format(mse_score))
LOGGER.debug('Training baseline r_squared = {:.2f}'.format(r_squared))
```



