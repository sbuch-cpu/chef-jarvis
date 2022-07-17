import math

from utilities.path_utilities import construct_path
from utilities.utilities import custom_tokenize_recipe
from model_training.training_data_prep import Jarvis_Dataset
from utilities.utilities import get_model_and_tokenizer, get_model_and_tokenizer_pretrained
from random import sample
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn import Softmax

from transformers import DistilBertForMaskedLM, DistilBertTokenizer


def process_line(line, cut_index):
    line_split = line.split()
    line_split = line_split[cut_index:]
    line_joined = " ".join(line_split)
    return line_joined


def simmr2JSON(filename):
    path = construct_path(f'models_and_data/SIMMR-DATA-v1.0/simmr.{filename}.txt')
    data = open(path, 'r')
    json_list = []
    tokenized = []
    recipe = None
    for l in data.readlines():
        if l.startswith('recipe'):
            if recipe:
                tokenized_recipe = custom_tokenize_recipe(recipe['ingredients'], recipe['instructions'])
                recipe['tokenized'] = tokenized_recipe
                tokenized.append(tokenized_recipe)
                json_list.append(recipe)
            recipe = {
                'title': "",
                'ingredients': [],
                'instructions': []
            }
            title = process_line(l, 1)
            recipe['title'] = title
        if l.startswith('ing'):
            ingredient = process_line(l, 2)
            recipe['ingredients'].append(ingredient)
        if l.startswith('inst'):
            instruction = process_line(l, 2)
            recipe['instructions'].append(instruction)

    return json_list, tokenized


def prepare_data(labels, model, tokenizer, truncation=True, padding=True):
    # Get the new token ids (need to remove the leading [CLS] token and trailing [SEP] token
    new_tokens = tokenizer('[INGSTART][INGITEM][INSTSTART][INSTITEM]')['input_ids'][1:-1]
    # Tokenize questions and contexts
    print("tokenizing...")
    data_encodings = tokenizer(labels, truncation=truncation, padding=padding, return_tensors='pt')
    data_encodings['labels'] = data_encodings.input_ids.detach().clone()
    rand = torch.rand(data_encodings.input_ids.shape)
    rand_new_tokens = torch.rand(data_encodings.input_ids.shape)
    new_token_positions = torch.zeros(data_encodings.input_ids.shape)
    for token in new_tokens:
        new_token_positions += data_encodings.input_ids == token
    new_token_positions *= rand_new_tokens
    new_token_positions = new_token_positions > 0.3
    # 15% except for where it is a CLS or PAD token or SEP token
    mask_arr = (rand < 0.10) * (data_encodings.input_ids != 101) * \
               (data_encodings.input_ids != 102) * (data_encodings.input_ids != 0)
    mask_arr += new_token_positions
    selection = []

    for i in range(data_encodings.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(data_encodings.input_ids.shape[0]):
        data_encodings.input_ids[i, selection[i]] = 103
    print('Done Tokenizing...')
    # Add tokenized start and end positions to data_encodings
    dataset = Jarvis_Dataset(data_encodings)
    print('Done Initializing...')
    return dataset


def test_model_accuracy(test_set, path, batch_size=16):
    # Load the model
    model, tokenizer = get_model_and_tokenizer_pretrained(path, QA=False, MLM=True)
    test_set = prepare_data(test_set, model, tokenizer)
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Set model to eval mode
    model.eval()
    # initialize validation set data loader
    val_loader = DataLoader(test_set, batch_size=batch_size)
    # initialize list to store accuracies
    acc = []
    # loop through batches
    for batch in val_loader:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print(input_ids)
            mask_location = np.where(input_ids == tokenizer.mask_token_id)

            # train model on batch and return outputs (incl. loss)
            token_logits = model(input_ids, attention_mask=attention_mask, labels=labels).logits
            # print(token_logits.shape)
            m = Softmax(dim=1)
            softmaxed = m(token_logits)
            # iterator = 0
            mask_entry = mask_location[0].tolist()
            # print(mask_ent
            mask_column = mask_location[1].tolist()
            # print(len(mask_entry))
            # print(len(mask_column))
            # print(softmaxed.shape)
            for idx, entry in enumerate(mask_column):
                prediction = np.argmax(softmaxed[mask_entry[idx], entry, :])
                actual = labels[mask_entry[idx], entry]
                print(f"Actual = {tokenizer.decode(actual)}, "
                      f"Prediction = {tokenizer.decode(prediction)}, "
                      f"Certainty = {softmaxed[mask_entry[idx], entry, prediction] * 100}%")
                if actual == prediction:
                    same = True
                else:
                    same = False
                acc.append(same)
    accuracy = sum(acc)/len(acc)
    print(f"{round(accuracy*100,6)}%")
    return round(accuracy, 6)


def fine_tune(train_dataset, model, tokenizer, batch_size=16, num_epochs=5, loss_function='backward',
              optimizer='AdamW'):
    # Load DistilBERT model and tokenizer
    # model, tokenizer = get_model_and_tokenizer()
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()

    if optimizer == 'AdamW':
        # initialize adam optimizer with weight decay (reduces chance of overfitting)
        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optim = AdamW(model.parameters(), lr=5e-5) OLD VERSION- new implementation above
    print('Loading into DataLoader...')
    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Loaded...')
    # set model to train mode
    model.train()
    print('In Training Mode...')
    for epoch in range(num_epochs):
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            if loss_function == "backward":
                loss.backward()
            else:
                loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    return model, tokenizer


def train_new_model(folder_name):
    json_list, tokenized = simmr2JSON('train')
    json_list_2, tokenized_2 = simmr2JSON('develop')
    tokenized.extend(tokenized_2)
    # Load DistilBERT model and tokenizer
    model, tokenizer = get_model_and_tokenizer('distilbert-base-uncased', QA=False, MLM=True)

    dataset = prepare_data(tokenized, model, tokenizer)
    model, tokenizer = fine_tune(dataset, model, tokenizer)
    model_path = construct_path(f'models_and_data/{folder_name}')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def test_new_model(folder_name):
    json_list, tokenized = simmr2JSON('test')
    model_path = construct_path(f'models_and_data/{folder_name}')
    accuracy = test_model_accuracy(tokenized, model_path)


if __name__ == "__main__":
    # filenames = ["develop", "test", "train"]
    folder_name = 'cloze_model'
    # train_new_model(folder_name)
    test_new_model(folder_name)
    # for i in r[1]:
    #     masked_string = mask_tokenized(i, 15, '[MASK]')
    #     prepared_data = prepare_data(masked_string, i)
    #
    #     print(masked_string)
