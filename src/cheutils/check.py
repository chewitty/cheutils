import numpy as np
from hyperopt.pyll.stochastic import sample
from hyperopt import hp
from hyperopt.pyll import scope
from cheutils.loggers import LoguruWrapper
from cheutils.properties_util import AppProperties
from cheutils.ml_utils.model_options import get_estimator, get_params_grid

LOGGER = LoguruWrapper().get_logger()
APP_PROPS = AppProperties()
MODEL_OPTION = APP_PROPS.get('model.active.model_option')
MODEL_PARAMS = APP_PROPS.get_dict_properties

def check_logger():
    """
    Output sample log messages to check status of log wrapper
    :return:
    :rtype:
    """
    LOGGER.trace('This is a TRACE message')
    LOGGER.debug('This is a DEBUG message')
    LOGGER.warning('This is a WARNING message')
    LOGGER.info('This is an INFO message')
    LOGGER.success('This is a SUCCESS message')
    LOGGER.error('This is an ERROR message')
    LOGGER.critical('This is a CRITICAL message')

def check_models():
    model = get_estimator(model_option=MODEL_OPTION, **get_params_grid(model_option=MODEL_OPTION))
    LOGGER.debug('Model instance = \n{}', model)

def check_exception():
    try:
        1 / 0
    except ZeroDivisionError as ex:
        LOGGER.exception('This is an EXCEPTION message = {}', ex)

def sample_hyperopt_space():
    small_num = 1e-10
    space = {'normal(0): 0->1'       : hp.lognormal('normal(0): 0->1', small_num, np.log(1.11)),
             'lognormal(1): 0->1'       : hp.lognormal('lognormal(1): 0->1', np.log(small_num), np.log(1.11)),
             'qlognormal(2): 3->17': scope.int(hp.qlognormal('qlognormal(2): 3->17', np.log(3), np.log(17), np.log(20.5))),
             'uniform:-10->10'       : hp.uniform('uniform:-10->19', -10, 19),
             'quniform(0),4: 3->17': scope.int(hp.quniform('quniform(0),4: 3->17', 3, 17, 1)),
             'loguniform(0): 0->1': hp.loguniform('loguniform(0): 0->1', np.log(small_num), np.log(1)),
             'qloguniform(3): 0->1': hp.qloguniform('qloguniform(3)/10: 0->1', np.log(small_num), np.log(1), np.log(1.1055)),
             'choice: 3,4,5'         : hp.choice('choice: 3,4,5', [3, 4, 5]),
             'pchoice: 3,4,5'        : hp.choice('pchoice: 3,4,5', [(3, 0.10), (4, 0.6), (5, 0.3)])}
    sample_params = sample(space)
    for key, param in sample_params.items():
        LOGGER.info('Sample hyperopt param: {} = {}', key, param)

# main entry point
if __name__ == "__main__":
    check_logger()
    sample_hyperopt_space()
    check_models()
    check_exception()