from codetiming import Timer
from tqdm.auto import tqdm

from .debugger import Debugger
from .decorator_debug import debug_func


def create_timer(text=None, name: str = 'Timer', logger = None):
    dblogger = logger if logger is not None else Debugger().debug
    return Timer(text=text, name=name, logger=dblogger)

def timer_stats(name = None, prec: int = 2, formatted: bool = False):
    """
    Generates the timer statistics (in minutes) for all or the named timer
    :param name: timer name or None if all desired
    :param prec: floating point precision desired
    :param formatted: return values as formatted strings
    :return: a dict of dicts with the statistics min, max, mean, stdev
    """
    timers = Timer.timers
    timerstats = {}
    if any([name is None, timers.get(name) is None]):
        for timer in timers.keys():
            max_time = timers.max(timer) / 60
            max_time = f'{max_time:.{prec}f}' if formatted else round(max_time, prec)
            min_time = timers.min(timer) / 60
            min_time = f'{min_time:.{prec}f}' if formatted else round(min_time, prec)
            mean_time = timers.mean(timer) / 60
            mean_time = f'{mean_time:.{prec}f}' if formatted else round(mean_time, prec)
            median_time = timers.median(timer) / 60
            median_time = f'{median_time:.{prec}f}' if formatted else round(median_time, prec)
            stdev_time = timers.stdev(timer) / 60
            stdev_time = f'{stdev_time:.{prec}f}' if formatted else round(stdev_time, prec)
            timerstats[timer] = {'min':min_time, 'max': max_time, 'mean': mean_time, 'median': median_time, 'std':stdev_time}
    else:
        max_time = round(timers.max(name) / 60, prec)
        max_time = f'{max_time:.{prec}f}' if formatted else max_time
        min_time = round(timers.min(name) / 60, prec)
        min_time = f'{min_time:.{prec}f}' if formatted else min_time
        mean_time = round(timers.mean(name) / 60, prec)
        mean_time = f'{mean_time:.{prec}f}' if formatted else mean_time
        median_time = round(timers.median(name) / 60, prec)
        median_time = f'{median_time:.{prec}f}' if formatted else median_time
        stdev_time = round(timers.stdev(name) / 60, prec)
        stdev_time = f'{stdev_time:.{prec}f}' if formatted else stdev_time
        timerstats[name] = {'min': min_time, 'max': max_time, 'mean': mean_time, 'median': median_time, 'std': stdev_time}
    return timerstats

@debug_func(enable_debug=True, prefix='progress')
def progress(*args, **kwargs):
    return tqdm(*args, **kwargs)