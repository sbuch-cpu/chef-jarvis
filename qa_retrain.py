import os
import pandas as pd
import spacy
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter

spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}  # Inverting dictionary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    if word not in frequencies:
                        frequencies[word] = 1
                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class RecipesDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, freq_threshold=5):
        df_dirty = pd.DataFrame(json.load(open(os.path.join(root_dir, json_file)))['data'])
        df = df_dirty[df_dirty['task'] == 'textual_cloze']
        df = df.drop(columns=['question_modality', 'task', 'qid', 'context_modality'])
        df['context'] = df['context'].apply(lambda x: [i['body'] for i in x])
        self.df = df
        self.transform = transform

        self.context = df['context']
        self.choice_list = df['choice_list']
        self.answer = df['answer']
        self.question = df['question']
        self.question_text = df['question_text']
        print(df.columns)

        # Initialize a Vocabulary
        self.vocab = Vocabulary(freq_threshold)
        for ctx in self.context:
            self.vocab.build_vocabulary(ctx)  # maybe send more not sure

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        context = self.context[index]
        choice_list = self.choice_list[index]
        answer = self.answer[index]
        question = self.question[index]
        question_text = self.question_text[index]

        # Numericalize
        numericalized_context = self.list_to_token(context)
        numericalized_choice_list = self.list_to_token(choice_list)
        numericalized_question = self.list_to_token(question)
        numericalized_question_text = self.list_to_token([question_text])[0]
        return numericalized_context, \
               numericalized_choice_list, \
               numericalized_question_text, \
               numericalized_question, \
               numericalized_choice_list[answer]

    def list_to_token(self, list):
        numericalized_list = []
        for idx in list:
            numericalized_item = [self.vocab.stoi["<SOS>"]]
            numericalized_item += self.vocab.numericalize(idx)
            numericalized_item.append(self.vocab.stoi["<EOS>"])
            numericalized_list.append(numericalized_item)
        return numericalized_list


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        context = [item[0] for item in batch]
        context = [pad_sequence(i, batch_first=False, padding_value=self.pad_idx) for i in context]
        choice_list = [item[1] for item in batch]
        choice_list = [pad_sequence(i, batch_first=False, padding_value=self.pad_idx) for i in choice_list]
        question_text = [item[2] for item in batch]
        question_text = pad_sequence(question_text, batch_first=False, padding_value=self.pad_idx)
        question = [item[3] for item in batch]
        question = [pad_sequence(i, batch_first=False, padding_value=self.pad_idx) for i in question]
        target = [item[4] for item in batch]
        target = pad_sequence(target, batch_first=False, padding_value=self.pad_idx)
        return context, choice_list, question_text, question, target


def get_loader(
        root_folder,
        json_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    dataset = RecipesDataset(json_file, root_folder, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader


def main():
    dataloader = get_loader('training_data/recipes_qa', 'train.json', None)

    for v in dataloader:
        print(v)


if __name__ == "__main__":
    main()
