import requests
import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

# os.mkdir('squad')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')


def main():
    files = ['train.json', 'val.json', 'test.json']
    train_contexts, train_questions, train_answers = read_squad(f'training_data/recipes_qa/{files[0]}')
    val_contexts, val_questions, val_answers = read_squad(f'training_data/recipes_qa/{files[1]}')
    test_contexts, test_questions, test_answers = read_squad(f'training_data/recipes_qa/{files[2]}')

    train_answers = add_end_idx(train_answers, train_contexts)
    val_answers = add_end_idx(val_answers, val_contexts)
    test_answers = add_end_idx(test_answers, test_contexts)

    # Tokenize/Encode
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)

    train_encodings = add_token_positions(train_encodings, train_answers)
    val_encodings = add_token_positions(val_encodings, val_answers)
    test_encodings = add_token_positions(test_encodings, test_answers)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    test_dataset = SquadDataset(test_encodings)
    start = time.time()
    print("###############  STARTING TRAINING  #####################")

    # FINE TUNING
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for epoch in range(3):
        loop = tqdm(train_loader)
        for batch in loop:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)

            loss = outputs[0]
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    # SAVE THE MODEL
    model_path = 'distilbert_custom'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    end = time.time()
    print("###############  FINISHED TRAINING  #####################")
    print(f"TRAINING TOOK {end - start} seconds")

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=16)
    acc = []
    loop = tqdm(val_loader)
    for batch in loop:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attenction_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())
    print(f'Accuracy = {sum(acc) / len(acc)}')
    return


def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    if 'plausible_answers' in qa.keys():
                        answer_key = 'plausible_answers'
                    else:
                        answer_key = 'answers'
                    for answer in qa[answer_key]:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx - n:end_idx - n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
    return answers


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - go_back)
            go_back += 1
    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })
    return encodings


class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == "__main__":
    main()
