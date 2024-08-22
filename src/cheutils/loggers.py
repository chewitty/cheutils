import sys
from loguru import logger

from cheutils.decorator_singleton import singleton
@singleton
class LoguruWrapper(object):
    instance__ = None
    logger__ = logger.opt(colors=True)
    config__ = {'handlers': [{'sink': sys.stdout,
                              'format': '{extra[prefix]} |{level} |{time:YYYY-MM-DD HH:mm:ss} | {file}:{line} | {message}',
                              'colorize': True, 'backtrace': True, 'level': 'TRACE', },
                             ],
                'extra'   : {'prefix': 'app-log'},
                }

    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created,
        or else returns the previous singleton object. Prefer using this logger
        to the standard print() statements in code.
        """
        if LoguruWrapper.instance__ is None:
            LoguruWrapper.instance__ = super().__new__(cls)
        if LoguruWrapper.logger__ is None:
            LoguruWrapper.logger__ = logger
        # reset the minimum log level to include TRACE
        # In addition, the logger is pre-configured for convenience with a default handler which writes messages
        # to sys.stderr. You should remove() it first if you plan to add() another handler logging messages to the
        # console, otherwise you may end up with duplicated logs.
        LoguruWrapper.logger__.remove()
        # configure any handlers
        if 'config' in kwargs.keys() and kwargs['config']:
            logger_config = kwargs['config']
            prefix = logger_config.get('prefix')
            if prefix is None:
                logger_config['prefix'] = 'app-log'
            LoguruWrapper.logger__.configure(**logger_config)
        else:
            LoguruWrapper.logger__.configure(**LoguruWrapper.config__)
        if 'enable_logging' in kwargs.keys() and kwargs['enable_logging']:
            LoguruWrapper.logger__.enable('app-log')
        else:
            LoguruWrapper.logger__.disable('app-log')
        return LoguruWrapper.instance__

    def __init__(self, *args, **kwargs):
        """
        Initialize the logger accordingly
        """
        if 'enable_logging' in kwargs.keys() and kwargs['enable_logging']:
            LoguruWrapper.logger__.enable('app-log')
        else:
            LoguruWrapper.logger__.disable('app-log')

    def configure(self, config: dict):
        if config is not None:
            LoguruWrapper.logger__.configure(**config)

    def __str__(self):
        info = 'LoguruWrapper'
        self.logger__.info(info)
        return info

    def addHandler(self, handler: dict):
        if handler is not None:
            LoguruWrapper.config__['handlers'].append(handler)
            LoguruWrapper.logger__.configure(**LoguruWrapper.config__)

    def get_logger(self):
        return self.logger__

    @staticmethod
    def set_prefix(prefix=None):
        if prefix is not None:
            LoguruWrapper.config__['extra']['prefix'] = prefix
            LoguruWrapper.logger__.configure(**LoguruWrapper.config__)

    def enable(self, prefix):
        LoguruWrapper.logger__.enable(name=prefix)
        LoguruWrapper.logger__.info('Logging enabled for ' + prefix)

    def disable(self, prefix):
        LoguruWrapper.logger__.info('Disabling logging for ' + prefix)
        LoguruWrapper.logger__.disable(name=prefix)

    def get_prefix(self):
        return LoguruWrapper.config__['extra']['prefix']
