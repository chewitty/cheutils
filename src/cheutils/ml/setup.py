import os
from setuptools import setup, find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception as err:
    long_description = 'Utilities for machine learning model development.'
    print('FAILED attempt to open file', os.path.join(current_directory, 'README.md'), err)
# continue specifying setup properties
setup(
    author='Ferdinand Che',
    author_email='ferdinand.che@gmail.com',
    description='Utilities for machine learning model development.',
    name='ml',
    version='1.0.1',
    url='https://github.com/chewitty/cheutils/ml_utils',
    packages=find_packages(include=['ml', 'ml.*']),
    install_requires=['numpy>=1.10', 'pandas', 'codetiming', 'tdqm', 'icecream', 'regex',
                      'inspect-it', 'jproperties', 'matplotlib', 'scikit-learn', 'icecream',
                      'geohash2'],
    python_requires='>=3.9',
)