import json
import pandas as pd
from utilities.utilities import get_model_and_tokenizer
from utilities.path_utilities import PATHS
import pickle
import torch
import random
from torch.utils.data import Dataset, random_split


class Jarvis_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def get_dataset_ready(size=1000, truncation=True, padding=True):
    # Get data ready for training and testing
    json_dataset, recipe_context = load_data_to_datasets()
    tokenize_data(json_dataset, recipe_context, size=size, truncation=truncation, padding=padding)
    initialize_dataset()


def load_data_to_datasets(question_set_path=PATHS['QUESTION_SET'], tokenized_recipes_path=PATHS['TOKENIZED_RECIPES']):
    # Read the existing question set json file using the json module
    with open(question_set_path, 'r') as f:
        json_formatted_dataset = json.load(f)

    # Read the contexts from the tokenized_recipes.csv file
    contexts = pd.read_csv(tokenized_recipes_path, index_col=0)

    return json_formatted_dataset, contexts


def tokenize_data(questions_dataset, contexts_dataset, size=1000,
                  truncation=True, padding=True, tokenized_data_path=PATHS['TOKENIZED_DATA']):
    # Load DistilBERT model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    if size != 'max' and isinstance(size, int):
        # take a random sample of 10 questions - just for testing purposes
        questions_dataset = random.sample(questions_dataset, size)

    # Get questions and context from datasets
    questions = [data['question'] for data in questions_dataset]
    context_options = contexts_dataset['tokenized'].values
    contexts = [context_options[data['recipe_index']] for data in questions_dataset]
    # Tokenize questions and contexts
    print("tokenizing...")
    data_encodings = tokenizer(questions, contexts, truncation=truncation, padding=padding)
    print('Done Tokenizing...')
    # Add tokenized start and end positions to data_encodings
    start_positions = [data['start_index'] for data in questions_dataset]
    end_positions = [data['end_index'] for data in questions_dataset]
    data_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    # Save tokenized data to pickle file
    with open(tokenized_data_path, 'wb') as file:
        # A new file will be created
        pickle.dump(data_encodings, file)
    # return data_encodings, model, tokenizer


def initialize_dataset(tokenized_data_path=PATHS['TOKENIZED_DATA'], initialized_data_path=PATHS['INITIALIZED_DATA']):

    # Load tokenized data from pickle file
    with open(tokenized_data_path, 'rb') as file:
        data_encodings = pickle.load(file)
    # Create a dataset from the tokenized data
    dataset = Jarvis_Dataset(data_encodings)
    # Pickle the dataset
    with open(initialized_data_path, 'wb') as file:
        pickle.dump(dataset, file)


def split_set(test_split=0.2, initialized_data_path=PATHS['INITIALIZED_DATA']):

    # Load initialized data from pickle file
    with open(initialized_data_path, 'rb') as file:
        dataset = pickle.load(file)
    # Split the dataset into training and validation sets using 80/20 split
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set
