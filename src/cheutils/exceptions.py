class CheutilsException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
        super().__init__(self.message)

    def __str__(self):
        if self.message:
            return type(self).__name__ + ', {0} '.format(self.message)
        else:
            return type(self).__name__ + ' raised'

class PropertiesException(CheutilsException):
    def __init__(self, *args):
        super().__init__(*args)

class DBToolException(CheutilsException):
    def __init__(self, *args):
        super().__init__(*args)

class DSWrapperException(CheutilsException):
    def __init__(self, *args):
        super().__init__(*args)
