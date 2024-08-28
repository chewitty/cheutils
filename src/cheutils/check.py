from hyperopt.pyll.stochastic import sample
from hyperopt import hp
from hyperopt.pyll import scope
from cheutils.loggers import LoguruWrapper

LOGGER = LoguruWrapper().get_logger()

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

def check_exception():
    try:
        1 / 0
    except ZeroDivisionError as ex:
        LOGGER.exception('This is an EXCEPTION message = {}', ex)

def sample_hyperopt_space():
    space = {'lognormal: 0->1'       : hp.lognormal('lognormal: 0->1', 0, 1),
             'uniform:-10->10 size=1'       : hp.uniform('uniform:-10->10', -10, 10, size=1),
             'quniform: 3->17 by 4': scope.int(hp.quniform('quniform: 0->1 by 4', 3, 17, 4)),
             'loguniform: 0->1'      : hp.loguniform('loguniform: 0->1', 0, 1) / 10,
             'qloguniform: 0->1 by 4': hp.qloguniform('qloguniform: 0->1 by 4', 0.001, 1, 4) / 10,
             'qlognormal: 3->17 by 4': scope.int(hp.qlognormal('loguniform: 0->1 by 4', 3, 17, 1) / 10),
             'qloguniform: 3->17 by high': scope.int(hp.qloguniform('qloguniform: 0->1 by 4', 3, 17, 1) / 10),
             'choice: 3,4,5'         : hp.choice('choice: 3,4,5', [3, 4, 5]),
             'pchoice: 3,4,5'        : hp.choice('pchoice: 3,4,5', [(3, 0.10), (4, 0.6), (5, 0.3)])}
    LOGGER.info('Sample hyperopt space: {}'.format(sample(space)))

# main entry point
if __name__ == "__main__":
    check_logger()
    sample_hyperopt_space()
    check_exception()