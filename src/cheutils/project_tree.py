"""
Utilities for generic project tree navigation and io. The basic project tree structure assumed is that
the project has a data folder and an output folder
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import estimator_html_repr

# Define project navigation constants.
PROJ_ROOT_DIR = './'
PROJ_DATA_DIR = './data/'
PROJ_OUTPUT_DIR = './output/'


def get_root_dir():
    """
    Get the root directory of the project. The assumption execution is from the root folder (.).
    :return: the path to the root directory.
    :rtype:
    """
    return PROJ_ROOT_DIR


def get_data_dir():
    """
    Get the data directory of the project, which is expected to be in the project root directory.
    :return: the path to the data directory.
    :rtype:
    """
    return PROJ_DATA_DIR


def get_output_dir():
    """
    Get the output directory of the project, which is expected to be in the project root directory.
    :return: the path to the output directory.
    :rtype:
    """
    return PROJ_OUTPUT_DIR


def load_dataset(file_name: str = None, is_csv: bool = True, date_cols: list = None, ):
    """
    Load the project dataset provided. The specified file is expected to be in either a CSV or Excel.
    :param file_name: the file name to be read from the data folder - so, only the file name and not the path is required
    :param is_csv: the default is CSV
    :param date_cols: columns with  dates that require parsing
    :return: a dataframe with the raw dataset
    """
    assert file_name is not None, 'file_name must be specified'
    path_to_dataset = os.path.join(get_data_dir(), file_name)
    dataset_df = None
    if is_csv:
        dataset_df = pd.read_csv(path_to_dataset)
    else:
        dataset_df = pd.read_excel(path_to_dataset, parse_dates=date_cols)
    print('Loaded dataset shape', dataset_df.shape)
    return dataset_df


def save_excel(df: pd.DataFrame, file_name: str, index: bool = False):
    """
    Save the specified dataframe to Excel.
    :param df: the dataframe to be saved
    :param file_name: the file name to be saved, which is expected to be saved in the data folder in the project root directory
    :param index: to include the index column or not
    :return:
    """
    assert df is not None, 'A valid DataFrame expected as input'
    assert file_name is not None, 'A valid file name expected as input'
    df.to_excel(os.path.join(get_data_dir(), file_name), index=index)


def save_current_fig(file_name: str, **kwargs):
    """
    Save the current figure as a file in the output folder of the project.
    :param file_name: the file name to be saved, which is expected to be saved in the output folder in the project root directory
    :type file_name:
    :param kwargs: any additional parameters to be passed to the underlying Matplotlib
    :type kwargs:
    :return:
    :rtype:
    """
    assert file_name is not None, 'A valid file name expected'
    plt.savefig(os.path.join(get_output_dir(), file_name), bbox_inches='tight', **kwargs)


def save_to_html(estimator, file_name: str):
    """
    Save an image representation of a pipeline or estimator or search object.
    :param estimator:
    :param file_name:
    :param kwargs:
    :return:
    """
    assert estimator is not None, 'A valid html renderable object expected'
    assert file_name is not None, 'A valid file name expected'
    # make the pipelines directory
    try:
        html_dir = os.path.join(get_data_dir(), 'html')
        os.makedirs(html_dir, exist_ok=True)
        with open(os.path.join(html_dir + '/', file_name), 'w', encoding='utf-8') as file:
            file.write(estimator_html_repr(estimator))
    except OSError as error:
        print("Directory '%s' cannot be created")


