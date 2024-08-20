import functools
from cheutils.loggers import LoguruWrapper


def debug_func(enable_debug: bool = True, prefix: str = None):
    """
    Enables or disables the debugger for the specified function
    :param enable_debug: enables the debugger if true
    :param prefix: any string to prefix the debugger
    :return: a decorator to enable or disable the underlying debugger
    """
    def debug_decorator(func):
        assert (func is not None), 'A function expected but None found'
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER = LoguruWrapper().get_logger()
            LOGGER.debug(f'Debug decorator ({func.__name__}) ... IN')
            LOGGER.debug('New status = {}', enable_debug)
            orig_prefix = LoguruWrapper().get_prefix()
            LOGGER.debug('Origin prefix = {} and New prefix = {}', orig_prefix, prefix)
            LoguruWrapper().set_prefix(prefix=prefix)
            # adjust status accordingly
            if enable_debug:
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
                    LOGGER.enable(prefix)
                else:
                    LOGGER.disable(orig_prefix)
                LoguruWrapper().set_prefix(prefix=orig_prefix)
                LOGGER.debug(f'Debug decorator ({func.__name__}) ... OUT')
        return wrapper
    return debug_decorator