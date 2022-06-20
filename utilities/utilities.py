from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering


def get_indexable_list(question, tokenized_recipe):
    question_list = question.split()
    question_list.append('[SEP]')
    # prepend '[CLS]' to the start of the question
    question_list.insert(0, '[CLS]')
    tokenized_recipe_list = tokenized_recipe.split()
    question_list.extend(tokenized_recipe_list)
    return question_list


def custom_tokenize_recipe(ingredients, instructions):
    ing_start = '[INGSTART]'
    ing_item = '[INGITEM]'
    inst_start = '[INSTSTART]'
    inst_item = '[INSTITEM]'

    ingredients = ing_item.join(ingredients) + ing_item
    instructions = inst_item.join(instructions) + inst_item
    tokenized_recipe = ing_start + ingredients + inst_start + instructions
    return tokenized_recipe


def get_model_and_tokenizer():
    model, tokenizer = get_model_and_tokenizer_pretrained("distilbert-base-uncased-distilled-squad")
    special_tokens_dict = {'additional_special_tokens': ['[INGSTART]', '[INGITEM]', '[INSTSTART]', '[INSTITEM]']}
    _ = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def get_model_and_tokenizer_pretrained(path_or_name):
    tokenizer = DistilBertTokenizer.from_pretrained(path_or_name)
    model = DistilBertForQuestionAnswering.from_pretrained(path_or_name)
    return model, tokenizer
