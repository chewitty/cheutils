import os
from setuptools import setup, find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception as err:
    long_description = 'A set of basic reusable utilities to and tools to facilitate quickly getting up and going on any project.'
    print('FAILED attempt to open file', os.path.join(current_directory, 'README.md'), err)
# continue specifying setup properties
setup(
    author='Ferdinand Che',
    author_email='ferdinand.che@gmail.com',
    description='A set of basic reusable utilities to and tools to facilitate quickly getting up and going on any project.',
    name='cheutils',
    version='1.0.2',
    url='https://github.com/chewitty/cheutils/cheutils',
    packages=find_packages(include=['cheutils', 'cheutils.*']),
    install_requires=['numpy>=1.10', 'pandas', 'codetiming', 'tdqm', 'dask',
                      'pytz', 'pyodbc', 'pymysql', 'mysqlclient', 'pymssql',
                      'mysql.connector', 'sqlalchemy', 'typing', 'regex',
                      'inspect-it', 'jproperties', 'matplotlib', 'scikit-learn', 'loguru',
                      'scikit-optimize', 'hyperopt', 'fast_ml', 'mlflow', ],
    python_requires='>=3.9',
)