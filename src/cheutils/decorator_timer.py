import functools

from cheutils.debugger import Debugger
from cheutils.progress_tracking import create_timer
from cheutils.progress_tracking import timer_stats

def track_duration(*, name: str = 'DurationStats', summary_stats: bool = False):
    """
    Tracks the duration of execution of any function in its context
    :param name: name to label the tracked duration for subsequent evaluation
    :param summary_stats: print duration summary stats
    :return: a duration tracking decorator
    """
    def track_duration_decorator(func):
        assert (func is not None), 'A function expected but None found'
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            DBugger = Debugger()
            DBugger.debug(f'Timer decorator ({func.__name__}) IN ...')
            # Create a list of the positional arguments. Use repr() to get a nice string representing each argument
            args_repr = [repr(a) for a in args]
            # Create a list of the keyword arguments. The f-string formats each argument as key=value where
            # the !r specifier means that repr() is used to represent the value
            kwargs_repr = [f'{k}={v!r}' for k, v in kwargs.items()]
            # The lists of positional and keyword arguments is joined together to one signature string with
            # each argument separated by a comma
            signature = ', '.join(args_repr + kwargs_repr)
            #DBugger.debug(f'Calling {func.__name__}({signature})')
            timer = create_timer(text='Completed task in {minutes:0.6f} minutes', name=name, logger=DBugger.debug)
            try:
                timer.start()
                value = func(*args, **kwargs)
                return value
            finally:
                time_taken = timer.stop()
                time_taken = time_taken / 60
                DBugger.debug(f'Total time ({func.__name__}): {time_taken:0.6f} minutes')
                if summary_stats:
                    DBugger.debug(timer_stats(name=name, prec=6, formatted=True))
                DBugger.debug(f'Timer decorator ({func.__name__}) OUT ...')
        return wrapper
    return track_duration_decorator

