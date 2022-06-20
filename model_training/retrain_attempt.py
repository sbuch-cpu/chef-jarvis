# WITH SOME CODE FROM https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997

import os
import pandas as pd
from transformers import DistilBertForQuestionAnswering
from model_training.training_data_prep import get_dataset_ready, split_set
from utilities.utilities import get_model_and_tokenizer
from utilities.path_utilities import PATHS
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def fine_tune(train_dataset, batch_size=16, num_epochs=3, loss_function='AdamW', optimizer='backward'):
    # Load DistilBERT model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
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

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # set model to train mode
    model.train()
    for epoch in range(num_epochs):
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


def save_model(model, tokenizer, model_path=PATHS['MODEL']):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save the model
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path


def test_model_accuracy(test_set, path, batch_size=16):
    # Load the model
    model = DistilBertForQuestionAnswering.from_pretrained(path)
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
    print(f'Model Accuracy = {acc * 100}%')
    return acc


def main(training_params, new_dataset=False, training_tracking_path=PATHS['TRAINING_TRACKING']):
    if new_dataset:
        get_dataset_ready(size=training_params['dataset_size'],
                          truncation=training_params['truncation'],
                          padding=training_params['padding'])

    train, test = split_set(training_params['test/train'])
    if not new_dataset:
        training_params['dataset_size'] = len(train) + len(test)
    trained_model, trained_tokenizer = fine_tune(train,
                                                 batch_size=training_params['batch_size'],
                                                 num_epochs=training_params['num_epochs'],
                                                 loss_function=training_params['loss_function'],
                                                 optimizer=training_params['optimizer'])
    pretrained_path = save_model(trained_model, trained_tokenizer)
    training_params['accuracy'] = test_model_accuracy(test,
                                                      pretrained_path,
                                                      batch_size=training_params['batch_size'])
    if os.path.exists(training_tracking_path):
        training_tracking = pd.read_csv(training_tracking_path, index_col=0)
    else:
        training_tracking = pd.DataFrame(columns=training_params.keys())
    new_row_params = {k: [v] for k, v in training_params.items()}
    print(new_row_params)
    new_row = pd.DataFrame(new_row_params, columns=training_params.keys())
    print(new_row)
    training_tracking = pd.concat([training_tracking, new_row])
    training_tracking.reset_index(inplace=True, drop=True)
    print(training_tracking)
    training_tracking.to_csv(training_tracking_path)


if __name__ == "__main__":
    training_information = {
        'dataset_size': 200,
        'test/train': 0.2,
        'batch_size': 10,
        'num_epochs': 5,
        'truncation': True,
        'padding': True,
        'loss_function': 'backward',
        'optimizer': 'AdamW',
        'accuracy': None
    }
    main(training_information, new_dataset=True)
