import os.path

def get_files(dir_path: str) -> list:
    """ Returns a list of full paths of files in the directory. """
    return [f for f in (os.path.join(dir_path, e) for e in os.listdir(dir_path)) if os.path.isfile(f)]

def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)


def get_file_name_without_extension(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def find_ancestor(path, ancestor_name):
    components = os.path.abspath(path).split(os.sep)
    return os.path.normpath(
        str.join(os.sep, components[:components.index(ancestor_name) + 1]))

def find_in_ancestor(path, ancestor_sibling_name):
    """ 
    `ancestor_sibling_name` can be the name of a sibling directory (or some
    descendant of the sibling) to some ancestor of `path`, but it can also be a 
    descendant of the sibling 
    """
    components = os.path.abspath(path).split(os.sep)
    while len(components) > 0:
        path = os.path.normpath(
            str.join(os.sep, components + [ancestor_sibling_name]))
        if os.path.exists(path):
            return path
        components.pop()
    assert False, "No ancestor sibling found"
