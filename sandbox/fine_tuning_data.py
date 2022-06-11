import pandas as pd
from sandbox.new_tokens import custom_tokenize_recipe
import re
import random


def main():
    training_data = pd.read_csv('../training_data/RAW_recipes.csv')
    training_data.drop(['minutes', 'id', 'contributor_id', 'submitted', 'tags',
                        'nutrition', 'n_steps', 'description', 'n_ingredients'], axis=1, inplace=True)

    training_data['steps'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.steps), axis=1)
    training_data['ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.ingredients), axis=1)
    training_data['raw_ingredients'] = training_data['ingredients']
    training_data['ingredients'] = training_data['ingredients'].apply(add_units_to_ingredients)
    training_data['tokenized'] = training_data.apply(lambda x: custom_tokenize_recipe(x.ingredients, x.steps), axis=1)
    # print(training_data['tokenized'].values[0])
    training_data.to_csv('../training_data/tokenized_recipes.csv')
    # print(training_data)
    return


def create_question_set():
    training_data = pd.read_csv('../training_data/tokenized_recipes.csv')
    training_data['raw_ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.raw_ingredients), axis=1)
    json_formatted_data = question_generation(training_data['tokenized'].values[0],
                                              training_data['raw_ingredients'].values[0])
    print(json_formatted_data)
    print(len(json_formatted_data))
    return


def add_units_to_ingredients(ing_list):
    return [f"{random_unit_of_measurement()} of {i}" for i in ing_list]


def question_generation(tokenized_recipe, ingredients):
    json_formatted_dataset = []
    tokenized_recipe = tokenized_recipe.replace('[', ' [').replace(']', '] ')
    # json_formatted_dataset = full_ingredient_list_question_generation(tokenized_recipe, json_formatted_dataset)
    json_formatted_dataset = get_specific_ingredient_question(tokenized_recipe, json_formatted_dataset, ingredients)
    return json_formatted_dataset


def full_ingredient_list_question_generation(tokenized_recipe, json_formatted_dataset):
    questions = ['What are the ingredients?',
                 'What ingredients do I need?',
                 'What ingredients are needed?',
                 'What are the ingredients for this recipe?',
                 'What ingredients do I need for this recipe?',
                 'Read me the ingredients.']
    for question in questions:
        indexable_list = get_indexable_list(question, tokenized_recipe)
        # Get the index of the token [INGSTART] in indexable_list
        start_index = indexable_list.index('[INGSTART]')
        # Get the index of the token [INSTSTART] in indexable_list
        end_index = indexable_list.index('[INSTSTART]')
        answer = ' '.join(indexable_list[start_index:end_index])
        end_index -= 1  # Since it is inclusive, we need to remove 1 from the end index
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset


def get_specific_ingredient_question(tokenized_recipe, json_formatted_dataset, ingredient_list):
    print(ingredient_list)
    questions = []
    questions.extend([f"How much {i}?" for i in ingredient_list])
    questions.extend([f"How much {i} is needed?" for i in ingredient_list])
    questions.extend([f"How much {i} is needed for this recipe?" for i in ingredient_list])
    questions.extend([f"How much {i} do I need?" for i in ingredient_list])
    questions.extend([f"How much {i} do I need for this recipe?" for i in ingredient_list])
    questions.extend([f"How much {i} is there?" for i in ingredient_list])
    questions.extend([f"How much {i} is there in this recipe?" for i in ingredient_list])

    for idx, question in enumerate(questions):
        while idx >= len(ingredient_list):
            # Each question was created by iterating over the ingredients list so we know when we get to the end of the
            # list we have reached the end of a set of questions and need to get the index back in check
            idx -= len(ingredient_list)
        ingredient = ingredient_list[idx]
        print(ingredient)
        indexable_list = get_indexable_list(question, tokenized_recipe)

        # Need to check that the index occurs after the SEP token so that we aren't looking for the answer in the question
        sep_index = indexable_list.index('[SEP]')
        if len(ingredient.split()) == 1:
            # If the ingredient is a single word, we can just search for it in the list
            ingredient_index = indexable_list.index(ingredient)
        else:
            ingredient_index_options = [i for i, x in enumerate(indexable_list) if x == ingredient.split()[0]]
            # Remove all options that are before the SEP token
            ingredient_index_options = [i for i in ingredient_index_options if i > sep_index]
            # Check if any options are followed by the next word in the ingredient
            ingredient_index_options_best = [i for i in ingredient_index_options if indexable_list[i + 1] == ingredient.split()[1]]
            if ingredient_index_options_best:
                # Maybe keep iterating until we find the best match but add that later
                ingredient_index = ingredient_index_options_best[0]
            else:
                ingredient_index = ingredient_index_options[0]
        print(ingredient_index, sep_index)
        # Get the index of the token [INGITEM] or [INGSTART] occuring before the ingredient in indexable_list whichever is closest
        # Get the index of all [INGITEM] in indexable_list
        ing_item_index = [i for i, x in enumerate(indexable_list) if x == '[INGITEM]']
        # Get the index of the token [INGSTART] in indexable_list
        ing_start_index = indexable_list.index('[INGSTART]')
        # Get all ing_item_indexes that are before the ingredient_index
        ing_item_index_before = [i for i in ing_item_index if i < ingredient_index]
        # If the ing_item_index is empty then the start index is the ing_start_index
        if not ing_item_index_before:
            start_index = ing_start_index
        else:
            start_index = ing_item_index_before[-1]
        # Get the index of the token [INGITEM] occuring after the ingredient in indexable_list
        end_index = ing_item_index[len(ing_item_index_before)]

        answer = ' '.join(indexable_list[start_index:end_index])
        end_index -= 1
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset


def get_indexable_list(question, tokenized_recipe):
    question_list = question.split()
    question_list.append('[SEP]')
    # prepend '[CLS]' to the start of the question
    question_list.insert(0, '[CLS]')
    tokenized_recipe_list = tokenized_recipe.split()
    question_list.extend(tokenized_recipe_list)
    return question_list


def random_unit_of_measurement():
    whole = random.randint(0, 5)
    if whole == 0:
        lower_den_bound = 2
    else:
        lower_den_bound = 0
    den = random.randint(lower_den_bound, 5)
    num = None
    if den != 0 and den != 1:
        num = random.randint(1, den - 1)

    if whole == 0:
        number = f"{num}/{den}"
    elif not num:
        number = str(whole)
    else:
        number = f"{whole} {num}/{den}"

    pluralizable_units = ['cup', 'ounce', 'gram', 'milligram', 'kilogram', 'milliliter', 'millilitre', 'liter', 'litre',
                          'pound', 'quart', 'stick', 'whole', 'can']
    unit_list = pluralizable_units.copy()
    unit_list.extend(['tsp', 'tbsp', 'lb', 'oz', 'g', 'mg', 'kg', 'fl oz', 'ml', 'l', 't', 'qt', 'c'])
    unit = random.choice(unit_list)

    if unit in pluralizable_units and whole != 1 and (num or whole != 0):
        unit = unit + 's'

    return f"{number} {unit}"


if __name__ == "__main__":
    # main()
    create_question_set()
    # print(random_unit_of_measurement())
