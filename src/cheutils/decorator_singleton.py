import functools

def singleton(cls):
    """
    Make a class a Singleton class (only one instance). Using cls instead of func as the parameter name
    to indicate that it is meant to be a class decorator
    :param cls: the class of interest
    :return: the singleton instance
    """
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = cls(*args, **kwargs)
        return wrapper.instance
    wrapper.instance = None
    return wrapper