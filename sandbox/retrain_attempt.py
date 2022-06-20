# WITH SOME CODE FROM https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997

import os
import json
import time

import pandas as pd
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from fine_tuning_data import get_path
from distilbert_custom.distilBERT_attempt import get_model_and_tokenizer
import pickle
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm


# define class elsewhere?
class Jarvis_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


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

    # take a random sample of 10 questions - just for testing purposes
    questions_dataset = random.sample(questions_dataset, 5)

    # Get questions and context from datasets
    questions = [data['question'] for data in questions_dataset]
    context_options = contexts_dataset['tokenized'].values
    contexts = [context_options[data['recipe_index']] for data in questions_dataset]
    # Tokenize questions and contexts
    data_encodings = tokenizer(questions, contexts, truncation=True, padding=True)
    # Add tokenized start and end positions to data_encodings
    start_positions = [data['start_index'] for data in questions_dataset]
    end_positions = [data['end_index'] for data in questions_dataset]
    data_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    # Save tokenized data to pickle file
    with open('tokenized_data.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(data_encodings, file)
    # return data_encodings, model, tokenizer


def initialize_dataset():
    # Load tokenized data from pickle file
    with open('tokenized_data.pkl', 'rb') as file:
        data_encodings = pickle.load(file)
    # Create a dataset from the tokenized data
    dataset = Jarvis_Dataset(data_encodings)
    # Pickle the dataset
    with open('initialized_data.pkl', 'wb') as file:
        pickle.dump(dataset, file)


def get_dataset_ready():
    # Get data ready for training and testing
    json_dataset, recipe_context = load_data_to_datasets()
    tokenize_data(json_dataset, recipe_context)
    initialize_dataset()


def split_set():
    # Load initialized data from pickle file
    with open('initialized_data.pkl', 'rb') as file:
        dataset = pickle.load(file)
    # Split the dataset into training and validation sets using 80/20 split
    test_split = 0.2
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set


def fine_tune(train_dataset):
    # Load DistilBERT model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optim = AdamW(model.parameters(), lr=5e-5) OLD VERSION- new implementation above

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for epoch in range(3):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    return model, tokenizer


def save_model(model, tokenizer):
    # Create a directory for the new model
    model_path = 'models/distilbert-custom'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save the model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path


def test_model_accuracy(test_set, path):
    # Load the model:
    model = DistilBertForQuestionAnswering.from_pretrained(path)
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Set model to eval mode
    model.eval()
    # initialize validation set data loader
    val_loader = DataLoader(test_set, batch_size=16)
    # initialize list to store accuracies
    acc = []
    # loop through batches
    for batch in val_loader:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # we will use true positions for accuracy calc
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull prediction tensors out and argmax to get predicted tokens
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc) / len(acc)
    print(acc)


def main():
    # Uncomment first line to get data tokenized and saved to pickle file
    # Time get_dataset_ready()
    start_time_first = time.time()
    get_dataset_ready()
    end_time = time.time()
    print(f'Time taken to get dataset ready: {end_time - start_time_first} seconds')
    # Time split_set()
    start_time = time.time()
    train, test = split_set()

    trained_model, trained_tokenizer, model_outputs = fine_tune(train)
    pretrained_path = save_model(trained_model, trained_tokenizer)  # does tokenizer need to be passed in?
    test_model_accuracy(test, pretrained_path)


if __name__ == "__main__":
    main()
