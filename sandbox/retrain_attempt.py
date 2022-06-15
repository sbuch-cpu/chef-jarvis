import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from fine_tuning_data import get_path

def main():
    # Read the existing question set json file using the json module
    module_path = get_path('chef-jarvis')
    with open(os.path.join(module_path, 'training_data/question_set.json'), 'r') as f:
        json_formatted_dataset = json.load(f)
    #print(json_formatted_dataset)

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

if __name__ == "__main__":
    main()