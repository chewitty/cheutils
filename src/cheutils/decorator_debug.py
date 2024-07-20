import functools
from debugger import Debugger
import functools

from debugger import Debugger


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
            DBugger = Debugger()
            DBugger.debug(f'Debug decorator ({func.__name__}) ... IN')
            orig_status = DBugger.status()
            DBugger.debug('Origin status = ', orig_status, ', and New status = ', enable_debug)
            orig_prefix = DBugger.prefix()
            orig_prefix = orig_prefix.replace('|', '').strip() if orig_prefix is not None else ''
            DBugger.debug('Origin prefix = ', orig_prefix, ', and New prefix = ', prefix)
            DBugger.set_debugger_prefix(prefix=prefix)
            # adjust status accordingly
            if enable_debug:
                if not orig_status:
                    DBugger.enable()
            else:
                if orig_status:
                    DBugger.disable()
            # call the function
            try:
                value = func(*args, **kwargs)
                # return the function outputs
                return value
            finally:
                # return to origin status
                if orig_status:
                    if not enable_debug:
                        DBugger.enable()
                else:
                    if enable_debug:
                        DBugger.disable()
                DBugger.set_debugger_prefix(prefix=orig_prefix)
                DBugger.debug(f'Debug decorator ({func.__name__}) ... OUT')
        return wrapper
    return debug_decorator