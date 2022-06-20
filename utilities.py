import os
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


# Get path to current directory
def get_path(base_folder):
    path = os.path.dirname(os.path.abspath(__file__))
    path_list = path.split('/')
    for i, folder in enumerate(path_list):
        if folder == base_folder:
            return '/'.join(path_list[:i + 1])
    return path


def get_indexable_list(question, tokenized_recipe):
    question_list = question.split()
    question_list.append('[SEP]')
    # prepend '[CLS]' to the start of the question
    question_list.insert(0, '[CLS]')
    tokenized_recipe_list = tokenized_recipe.split()
    question_list.extend(tokenized_recipe_list)
    return question_list


def custom_tokenize_recipe(ingredients, instructions):
    # print(recipe)
    ing_start = '[INGSTART]'
    ing_item = '[INGITEM]'
    inst_start = '[INSTSTART]'
    inst_item = '[INSTITEM]'

    ingredients = ing_item.join(ingredients) + ing_item
    instructions = inst_item.join(instructions) + inst_item
    tokenized_recipe = ing_start + ingredients + inst_start + instructions
    return tokenized_recipe


def get_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    special_tokens_dict = {'additional_special_tokens': ['[INGSTART]', '[INGITEM]', '[INSTSTART]', '[INSTITEM]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
