import os
from .properties_util import AppProperties, AppPropertiesHandler
from .exceptions import PropertiesException, DBToolException, DSWrapperException, SQLiteUtilException, FeatureGenException
from .decorator_singleton import singleton
from .project_tree_handler import ProjectTreeProperties
from .project_tree import (get_data_dir, get_root_dir, get_output_dir, load_dataset, save_to_html,
                           save_csv, save_excel, save_current_fig, estimator_html_repr)
from .common_utils import (validate_data, apply_annova, cat_to_numeric, calc_prop_missing, find_numeric, apply_clipping,
                           get_date, get_func_def, label, datestamp, get_quantiles, get_aggs,
                           get_correlated, apply_impute, summarize, parse_special_features, safe_copy, winsorize_it, trim_both)
from .common_base import CheutilsBase
from .progress_tracking import create_timer, timer_stats, progress
from .decorator_timer import track_duration
from .decorator_debug import debug_func
from .ml import (get_estimator, exclude_nulls,
                 get_hyperopt_estimator, show_pipeline, plot_pie, plot_reg_predictions,
                 plot_reg_residuals_dist, plot_reg_predictions_dist, plot_reg_residuals, plot_hyperparameter,
                 get_optimal_grid_resolution, get_params_grid, get_params_pounds, get_param_defaults,
                 get_narrow_param_grid, HyperoptSearch, HyperoptSearchCV, ModelProperties,
                 promising_params_grid, params_optimization, plot_no_skill_line,
                 plot_confusion_matrix, plot_decision_tree, plot_precision_recall, plot_precision_recall_by_threshold,
                 print_classification_report, recreate_labels, rmsle)
from .interceptor import PipelineInterceptor, NumericDataInterceptor
from .datasource_utils import DBTool, DBToolFactory, DSWrapper
from .data_prep import (feature_selection_transformer, DateFeaturesTransformer, FeatureGenTransformer,
                        DropSelectedColsTransformer, SelectiveColumnTransformer, GeospatialTransformer, CategoricalTargetEncoder,
                        DataPrepTransformer, SelectiveFunctionTransformer, DataPrepProperties, ExtremeStateFeatureAugmenter,
                        BinarizerColumnTransformer, TSFeatureAugmenter, TSLagFeatureAugmenter, TrendFeatureAugmenter,
                        TSRollingLagFeatureAugmenter, ClipOutliersTransformer, PeriodicFeatureAugmenter, DataInterceptorTransformer, )
from .sqlite_util import save_param_grid_to_sqlite_db, get_param_grid_from_sqlite_db, save_narrow_grid_to_sqlite_db, get_narrow_grid_from_sqlite_db
from .loggers import LoguruWrapper
from .check import check_logger, check_exception, sample_hyperopt_space
from .target_encoder import mean_target_encoding, train_mean_target_encoding, test_mean_target_encoding

__proj_handler: ProjectTreeProperties = AppProperties().get_subscriber('proj_handler')
log_handler = {'sink': os.path.join(__proj_handler.get_proj_output(), 'app-log.log'), 'serialize': False, 'backtrace': True,
               'format': '{extra[prefix]} |{level} |{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {message}', 'level': 'TRACE',
               'rotation': '00:00', 'colorize': True, 'enqueue': True, }
LoguruWrapper().addHandler(log_handler)
LoguruWrapper().set_prefix(prefix=__proj_handler.get_proj_namespace())
