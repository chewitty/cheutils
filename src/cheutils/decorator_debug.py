import functools
from cheutils.loggers import LoguruWrapper


def debug_func(enable_debug: bool = True, prefix: str = None):
    """
    Enables or disables the logger (debug level) for the specified function
    :param enable_debug: enables the logger if true
    :param prefix: any string to prefix the logger
    :return: a decorator to enable or disable the underlying logger
    """
    def debug_decorator(func):
        assert (func is not None), 'A function expected but None found'
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER = LoguruWrapper().get_logger()
            LOGGER.debug(f'Debug decorator ({func.__name__}) ... IN')
            LOGGER.debug('Set logging status = {}', enable_debug)
            orig_prefix = LoguruWrapper().get_prefix()
            LOGGER.debug('Origin prefix = {} and New prefix = {}', orig_prefix, prefix)
            # adjust status accordingly
            if enable_debug:
                LoguruWrapper().set_prefix(prefix=prefix)
                LOGGER.enable(prefix)
            else:
                LOGGER.disable(orig_prefix)
            # call the function
            try:
                value = func(*args, **kwargs)
                # return the function outputs
                return value
            finally:
                # return to origin status
                if not enable_debug:
                    LOGGER.enable(orig_prefix)
                else:
                    LOGGER.disable(prefix)
                LoguruWrapper().set_prefix(prefix=orig_prefix)
                LOGGER.debug(f'Debug decorator ({func.__name__}) ... OUT')
        return wrapper
    return debug_decorator