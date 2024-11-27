import os
from .properties_util import AppProperties
from .exceptions import PropertiesException, DBToolException, DSWrapperException
from .decorator_singleton import singleton
from .project_tree import (get_data_dir, get_root_dir, get_output_dir, load_dataset, save_to_html,
                           save_csv, save_excel, save_current_fig, estimator_html_repr)
from .common_utils import (validate_data, apply_annova, cat_to_numeric, calc_prop_missing, find_numeric, apply_clipping,
                           get_date, get_func_def, label, datestamp, properties_to_frame, get_quantiles, get_aggs,
                           get_correlated, apply_impute, summarize, parse_special_features, safe_copy, winsorize_it, trim_both)
from .common_base import CheutilsBase
from .progress_tracking import create_timer, timer_stats, progress
from .decorator_timer import track_duration
from .decorator_debug import debug_func
from .ml_utils import (fit, get_estimator, exclude_nulls,
                       get_hyperopt_estimator, show_pipeline, plot_pie, plot_reg_predictions,
                       plot_reg_residuals_dist, plot_reg_predictions_dist, plot_reg_residuals, plot_hyperparameter,
                       get_optimal_grid_resolution, get_params_grid, get_params_pounds, get_default_grid, get_params,
                       get_narrow_param_grid, HyperoptSearch, HyperoptSearchCV,
                       promising_params_grid, params_optimization, get_column_transformers)
from .datasource_utils import DBTool, DBToolFactory, DSWrapper
from .data_prep import (FeatureSelectionTransformer, DateFeaturesTransformer,
                        DropSelectedColsTransformer, SelectiveColumnTransformer,
                        DataPrepTransformer, pre_process, generate_target)
from .sqlite_util import save_param_grid_to_sqlite_db, get_param_grid_from_sqlite_db
from .loggers import LoguruWrapper
from .check import check_logger, check_exception, sample_hyperopt_space

APP_PROPS = AppProperties()
LOGGER = LoguruWrapper()
log_handler = {'sink': os.path.join(get_output_dir(), 'app-log.log'), 'serialize': False, 'backtrace': True,
               'format': '{extra[prefix]} |{level} |{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {message}', 'level': 'TRACE',
               'rotation': '00:00', }
LOGGER.addHandler(log_handler)
LOGGER.set_prefix(prefix=APP_PROPS.get('project.namespace'))
