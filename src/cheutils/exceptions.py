class PropertiesException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
        super().__init__(self.message)

    def __str__(self):
        if self.message:
            return 'PropertiesException, {0} '.format(self.message)
        else:
            return 'PropertiesException raised'