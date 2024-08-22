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
    try:
        1 / 0
    except ZeroDivisionError as ex:
        LOGGER.exception('This is an EXCEPTION message = {}', ex)


# main entry point
if __name__ == "__main__":
    check_logger()