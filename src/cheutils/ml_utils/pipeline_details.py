from IPython.display import display
from ..project_tree import save_to_html
from sklearn import set_config


def show_pipeline(pipeline, name: str='pipeline.png', save_to_file: bool=False):
    # Review the pipeline
    set_config(display='diagram')
    # with display='diagram', simply use display() to see the diagram
    display(pipeline)
    # if desired, set display back to the default
    set_config(display='text')
    # save it to file
    if save_to_file:
        save_to_html(pipeline, file_name=name)
