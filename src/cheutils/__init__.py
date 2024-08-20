import os
from .properties_util import AppProperties
from .debugger import Debugger
from .exceptions import PropertiesException
from .decorator_singleton import singleton
from .project_tree import (get_data_dir, get_root_dir, get_output_dir, load_dataset, save_to_html,
                           save_csv, save_excel, save_current_fig, estimator_html_repr)
from .common_utils import (validate_data, apply_annova, cat_to_numeric, calc_prop_missing, find_numeric,
                           quantilefy, get_date, get_func_def, label, datestamped)
from .common_base import CheutilsBase
from .progress_tracking import create_timer, timer_stats, progress
from .decorator_timer import track_duration
from .decorator_debug import debug_func
from .ml_utils import (fit, predict, score, cross_val_model, get_regressor, coarse_fine_tune, exclude_nulls,
                       get_hyperopt_regressor, show_pipeline, save_to_html, plot_pie, plot_reg_predictions,
                       plot_reg_residuals_dist, plot_reg_predictions_dist, plot_reg_residuals, plot_hyperparameter,
                       get_optimal_num_params, get_params_grid, get_params_pounds, get_default_grid, get_params,
                       get_narrow_param_grid, get_seed_params, eval_metric_by_params)
from .loggers import LoguruWrapper

DBUGGER = Debugger()
APP_PROPS = AppProperties()
LOGGER = LoguruWrapper()
LOGGER.addHandler({'sink': os.path.join(get_output_dir(), 'app-log.log'), 'serialize': False, 'backtrace': True})
