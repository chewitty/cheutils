# See also: Kane, B. (n.d.). Auto-printing Python Classes | Ben’s Corner. Retrieved 31 July 2024, from https://www.bbkane.com/blog/auto-printing-python-classes/
class CheutilsBase:
    def __repr__(self):
        """
        Generate a string representation of the object based on its class name and member variables.
        :return: string representation of the object based on its class name and member variables.
        :rtype: str
        """
        name = type(self).__name__
        vars_list = [f'{key}={value!r}'
                     for key, value in vars(self).items()]
        vars_str = ', '.join(vars_list)
        return f'{name}({vars_str})'