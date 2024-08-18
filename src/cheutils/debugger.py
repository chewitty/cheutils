import re
from datetime import datetime
from inspect import getframeinfo, stack

from icecream import ic as debug
from icecream import install

from .decorator_singleton import singleton


@singleton
class Debugger(object):
    instance__ = None
    def __new__(cls, *args, **kwargs):
        """
        Creates a singleton instance if it is not yet created, 
        or else returns the previous singleton object. Prefer using this debugger
        to the standard print() statements in code.
        """
        if Debugger.instance__ is None:
            Debugger.instance__ = super().__new__(cls)
            # To make ic() available in every file without needing to be imported in every file, you can install() it
            install(ic='debug')
        if 'enable_debug' in kwargs.keys() and kwargs['enable_debug']:
            Debugger.instance__.enable()
        else:
            Debugger.instance__.disable()
        return Debugger.instance__
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the debugger accordingly
        """
        if 'enable_debug' in kwargs.keys() and kwargs['enable_debug']:
            debug.enable()
        else:
            debug.disable()
        if 'prefix' in kwargs.keys():
            prefix = kwargs['prefix']
            if '' == kwargs['prefix']:
                self.prefix_ = f'|Debug'
            else:
                self.prefix_ = f'|{prefix}'
        else:
            self.prefix_ = f'|Debug'
        debug.configureOutput(prefix=f'{self.prefix_} |')

    def __str__(self):
        info = 'Debugger'
        self.instance__.debug(info)
        return info

    @staticmethod
    def set_debugger_prefix(prefix=None):
        if prefix is not None:
            debug.configureOutput(prefix=f'|{prefix} |')

    def debug(self, *args):
        """
        Call this instead of the usual "debug" if you do not want the object returned afterwards.
        """
        curTime = f'[{datetime.now()}]'
        caller = getframeinfo(stack()[1][0])
        filename = re.split(r'[\\/]', caller.filename)[-1]
        msg = f'{curTime} |{filename}:line={caller.lineno} |'
        debug(msg, args)
        
    def enable(self):
        debug.enable()
        debug('Enabled debug')
        
    def disable(self):
        debug('Disabling debug')
        debug.disable()

    def status(self):
        return debug.enabled

    def prefix(self):
        return debug.prefix

        