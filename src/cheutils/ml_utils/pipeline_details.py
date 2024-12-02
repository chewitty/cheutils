from IPython.display import display
from sklearn import set_config
from cheutils.loggers import LoguruWrapper
from cheutils.project_tree import save_to_html

LOGGER = LoguruWrapper().get_logger()

def show_pipeline(pipeline, name: str='pipeline.png', save_to_file: bool=False):
    """
    Displays the pipeline diagram and optionally saved it to a file in the project output directory.
    :param pipeline:
    :type pipeline:
    :param name: any file name to be used for saving the diagram; defaults to 'pipeline.png'
    :type name:
    :param save_to_file: optional save to file with the name specified; defaults to False
    :type save_to_file:
    :return:
    :rtype:
    """
    # Review the pipeline
    set_config(display='diagram')
    # with display='diagram', simply use display() to see the diagram
    display(pipeline)
    # if desired, set display back to the default
    set_config(display='text')
    # save it to file
    if save_to_file:
        save_to_html(pipeline, file_name=name)
        LOGGER.debug('Pipeline diagram saved to file: {}', name)

