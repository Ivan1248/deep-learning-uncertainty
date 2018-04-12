import .ioutils.filesystem as fs

def _find_in_ancestor(path_end):
    try:
        return fs.find_in_ancestor(__file__, path_end)
    except:
        print("ERROR: dirs.py: Could not find '"+path_end+"'.")
        return None


SAVED_MODELS = _find_in_ancestor('data/models')
LOGS = _find_in_ancestor('data/logs')
DATASETS = _find_in_ancestor('projects/datasets')