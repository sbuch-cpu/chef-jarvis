import os

# get path to current directory
def get_path(base_folder):
    path = os.path.dirname(os.path.abspath(__file__))
    path_list = path.split('/')
    for i, folder in enumerate(path_list):
        if folder == base_folder:
            return '/'.join(path_list[:i + 1])
    return path