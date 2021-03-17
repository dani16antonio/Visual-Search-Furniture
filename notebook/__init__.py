# get project path to import src on Jupyter Notebook
import sys
from pathlib import Path

def get_parent_path():
    return str(Path(Path(__file__).parent.absolute()).parent.absolute())
