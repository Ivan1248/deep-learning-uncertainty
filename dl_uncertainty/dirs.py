#from . import ioutils
from .ioutils import filesystem as fs


def _find_in_ancestor(path_end):
    try:
        return fs.find_in_ancestor(__file__, path_end)
    except:
        print("dirs.py: Cannot find '" + path_end + "'.")
        return None


SAVED_NETS = _find_in_ancestor('data/nets')
LOGS = _find_in_ancestor('data/logs')
DATASETS = _find_in_ancestor('datasets')
