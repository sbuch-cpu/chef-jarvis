import os
import json
import pandas as pd
from fine_tuning_data import get_path
from distilbert_custom.distilBERT_attempt import get_model_and_tokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def load_data_to_datasets():
    # Read the existing question set json file using the json module
    module_path = get_path('chef-jarvis')
    with open(os.path.join(module_path, 'training_data/question_set.json'), 'r') as f:
        json_formatted_dataset = json.load(f)

    # Read the contexts from the tokenized_recipes.csv file
    contexts = pd.read_csv(os.path.join(module_path, 'training_data/tokenized_recipes.csv'), index_col=0)

    return json_formatted_dataset, contexts


def tokenize_data(questions_dataset, contexts_dataset):
    # Load DistilBERT model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    # Get questions and context from datasets
    questions = [data['question'] for data in questions_dataset]
    context_options = contexts_dataset['tokenized'].values
    contexts = [context_options[data['recipe_index']] for data in questions_dataset]
    # Tokenize questions and contexts
    data_encodings = tokenizer(questions, contexts, truncation=True, padding=True)


def split_set():
    print("coming later")
    '''
    # Create indices for the split (80/20 train/test)
    test_split = 0.2
    dataset_size = len(json_formatted_dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_set, test_set = random_split(json_formatted_dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_set.dataset,
        batch_size=1,
        shuffle=True)
    test_loader = DataLoader(
        test_set.dataset,
        batch_size=1,
        shuffle=True)
    print(train_set)
    print("#################### BREAK #####################")
    print(test_set)
    print("#################### BREAK #####################")
    print(train_loader)
    print("#################### BREAK #####################")
    print(test_loader)
    '''


def main():
    json_dataset, recipe_context = load_data_to_datasets()
    tokenize_data(json_dataset, recipe_context)
    # split_set()


if __name__ == "__main__":
    main()
