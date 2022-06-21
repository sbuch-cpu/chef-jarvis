import os


# Get path to current directory
def get_module_path(base_folder):
    path = os.path.dirname(os.path.abspath(__file__))
    path_list = path.split('/')
    for i, folder in enumerate(path_list):
        if folder == base_folder:
            return '/'.join(path_list[:i + 1])
    return path


def construct_path(path_from_module):
    module_path = get_module_path('chef-jarvis')
    return os.path.join(module_path, path_from_module)


PATHS = {
    'QUESTION_SET': construct_path('models_and_data/training_dataset/question_set.json'),
    'RAW_RECIPES': construct_path('models_and_data/training_dataset/RAW_recipes.csv'),
    'TOKENIZED_RECIPES': construct_path('models_and_data/training_dataset/tokenized_recipes.csv'),
    'TOKENIZED_DATA': construct_path('models_and_data/datasplits/tokenized_data.pkl'),
    'INITIALIZED_DATA': construct_path('models_and_data/datasplits/initialized_data.pkl'),
    'TRAINING_PARAMS': construct_path('models_and_data/training_params.csv'),
    'MODEL': construct_path('models_and_data/models')
}
