import os
import requests
import json
import numpy as np
os.mkdir('squad')
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"


def main():
    files = ['train-v2.0.json', 'dev-v2.0.json']
    for file in files:
        res = requests.get(f'{url}{file}')
        with open(f'squad/{file}', 'wb') as f:
            for chunk in res.iter_content(chunk_size=4):
                f.write(chunk)
    read_squad(f'squad/{files[0]}')
    return


def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
        for group in squad_dict['data']:
            # print(group['title'])
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    print(qa)
                    question = qa['question']
                    answer = np.squeeze(qa['answers'])

if __name__ == "__main__":
    main()
