import time

import pandas as pd
from sandbox.new_tokens import custom_tokenize_recipe
import re
import random


def format_raw_recipe_dataset():
    """
    This function formats the raw recipe dataset retrieved from
    https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download&select=RAW_recipes.csv
    and tokenizes the recipes in the format desired by the model. [SEP] and [CLS] tokens are omitted. The new dataset is
    saved in the training_data folder under the name tokenized_recipes.csv.
    """
    # Read the raw dataset
    training_data = pd.read_csv('../training_data/RAW_recipes.csv')
    # Remove the unnecessary columns from the dataset
    training_data.drop(['minutes', 'id', 'contributor_id', 'submitted', 'tags',
                        'nutrition', 'n_steps', 'description', 'n_ingredients'], axis=1, inplace=True)
    print('Changing to lists...')
    # Convert the data in 'steps' and 'ingredients' from a string to a list of strings using regex
    training_data['steps'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.steps), axis=1)
    training_data['ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.ingredients), axis=1)
    # Save the ingredients list without the units of measurement
    training_data['raw_ingredients'] = training_data['ingredients']
    print('Adding units of measurement...')
    # add units of measurement to the ingredients list so the model can understand the ingredients
    # These units are randomly generated so they do not always make sense for the ingredient they are associated with
    # This should not affect the accuracy of the model ... hopefully
    training_data['ingredients'] = training_data['ingredients'].apply(add_units_to_ingredients)
    print('Tokenizing...')
    # Tokenize the recipes
    training_data['tokenized'] = training_data.apply(lambda x: custom_tokenize_recipe(x.ingredients, x.steps), axis=1)
    print('Saving...')
    # Save the tokenized dataset
    training_data.to_csv('../training_data/tokenized_recipes.csv')


def add_units_to_ingredients(ing_list):
    """
    This function adds units of measurement to the ingredients list so the model can understand the ingredients
    These units are randomly generated so they do not always make sense for the ingredient they are associated with

    :param ing_list: A list of ingredients with no units of measurement
    :type ing_list: list[str]
    :return: A list of ingredients with units of measurement
    :rtype: list[str]
    """
    return [f"{random_unit_of_measurement()} {i}" for i in ing_list]


def random_unit_of_measurement():
    """
    This function returns a random unit of measurement

    :return: A random unit of measurement
    :rtype: str
    """
    # We don't want units of measurement to be on all of the ingredients to avoid the model overfitting to this style of
    # ingredient presentation so every once in a while an empty string is returned. Randomly, this happens 1.5/10 times.
    if random.random() < 0.15:
        return ''
    # Randomly return 'a pinch/dash' so that the model does not always expect a unit of measurement to include a number
    # Randomly, this happens 1/10 times.
    if random.random() < 0.1:
        phrases = ['a pinch of', 'a dash of', 'some', 'a bit of', 'a sprig of',
                   'a bundle of', 'a handful of', 'a crack of']
        return random.choice(phrases)

    # The whole number part of the unit of measurement is capped at 5 because that seems to be a reasonable upper bound
    # and will keep it somewhat likely that some units of measurement will just be a fraction (e.g. whole number = 0)
    whole = random.randint(0, 5)
    # If the whole number is 0 then we don't want the denominator of the fraction part to be 0 or 1 because a
    # denominator of 1 is equivalent to a whole number and a denominator of 0 this is usually handled by eliminating
    # the fraction part of the unit of measurement, but we need to keep it if there is no whole number part
    if whole == 0:
        lower_den_bound = 2
    else:
        lower_den_bound = 0
    # The denominator part of the unit of measurement is capped at 5 because that seems to be a reasonable upper bound
    den = random.randint(lower_den_bound, 5)
    num = None  # This is initialized to None for the case where the denominator is 0 or 1
    if den != 0 and den != 1:
        num = random.randint(1, den - 1)

    # If the whole number is 0 then we only want the fraction
    if whole == 0:
        number = f"{num}/{den}"
    elif not num:  # if the numerator was not calculated then we only want the whole number
        number = str(whole)
    else:  # if the whole number and the numerator were calculated then we want the whole number and the fraction
        number = f"{whole} {num}/{den}"

    # These units of measurement can be made plural by adding an 's' to the end of the unit of measurement
    pluralizable_units = ['cup', 'ounce', 'gram', 'milligram', 'kilogram', 'milliliter', 'millilitre', 'liter', 'litre',
                          'pound', 'quart', 'stick', 'whole', 'can']
    # Create a full list of units of measurement
    unit_list = pluralizable_units.copy()
    unit_list.extend(['tsp', 'tbsp', 'lb', 'oz', 'g', 'mg', 'kg', 'fl oz', 'ml', 'l', 't', 'qt', 'c'])
    unit = random.choice(unit_list)  # Choose a random unit of measurement

    # If the unit of measurement can be pluralized and the number is not one then add an 's' to the end of the unit of
    # measurement
    if unit in pluralizable_units and whole != 1 and (num or whole != 0):
        unit = unit + 's'

    return f"{number} {unit} of"


#################  Ingredient Question Generation ##############################
def create_ingredients_question_set():
    """
    This function creates a set of questions for the model to ask the user about the ingredients of a recipe.
    """
    # Read the tokenized dataset
    training_data = pd.read_csv('../training_data/tokenized_recipes.csv')
    # Convert the data in 'raw_ingredients' to a list of strings
    training_data['raw_ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.raw_ingredients),
                                                           axis=1)
    # Pull the tokenized recipe and the raw ingredients from the dataset
    tokenized = training_data['tokenized'].values
    ingredients = training_data['raw_ingredients'].values
    json_formatted_dataset = []
    # Generate question answer pairs from multiple questions per recipe
    for i, recipe in enumerate(tokenized):
        # print the progress of the loop for every 100 recipes
        if i % 100 == 0:
            print(f"{i}/{len(tokenized)} --- {round(i / len(tokenized) * 100, 2)}%")
        json_formatted_dataset.extend(question_generation_ingredients(recipe, ingredients[i], i))
    # Save the dataset
    with open('../training_data/question_set.json', 'w') as f:
        f.write(str(json_formatted_dataset))
    # print the number of question answer pairs generated
    print(len(json_formatted_dataset))


def question_generation_ingredients(tokenized_recipe, ingredients, recipe_index):
    """
    This function generates a set of questions for the model to ask the user about the ingredients of a recipe.
    :param tokenized_recipe:
    :param ingredients:
    :param recipe_index:
    :return:
    """
    json_formatted_dataset = []
    tokenized_recipe = tokenized_recipe.replace('[', ' [').replace(']', '] ')
    json_formatted_dataset = full_ingredient_list_question_generation(tokenized_recipe, json_formatted_dataset,
                                                                      recipe_index)
    json_formatted_dataset.extend(
        get_specific_ingredient_question(tokenized_recipe, json_formatted_dataset, ingredients, recipe_index))
    return json_formatted_dataset


def full_ingredient_list_question_generation(tokenized_recipe, json_formatted_dataset, recipe_index):
    questions = ['What are the ingredients?',
                 'What ingredients do I need?',
                 'What ingredients are needed?',
                 'What are the ingredients for this recipe?',
                 'What ingredients do I need for this recipe?',
                 'Read me the ingredients.']
    # just choose three random questions to generate
    random_questions = random.sample(questions, 3)
    for question in random_questions:
        indexable_list = get_indexable_list(question, tokenized_recipe)
        # Get the index of the token [INGSTART] in indexable_list
        start_index = indexable_list.index('[INGSTART]')
        # Get the index of the token [INSTSTART] in indexable_list
        end_index = indexable_list.index('[INSTSTART]')
        answer = ' '.join(indexable_list[start_index:end_index])
        end_index -= 1  # Since it is inclusive, we need to remove 1 from the end index
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'recipe_index': recipe_index,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset


def get_specific_ingredient_question(tokenized_recipe, json_formatted_dataset, ingredient_list, recipe_index):
    questions = []
    questions.extend([f"How much {i}?" for i in ingredient_list])
    questions.extend([f"How much {i} is needed?" for i in ingredient_list])
    questions.extend([f"How much {i} is needed for this recipe?" for i in ingredient_list])
    questions.extend([f"How much {i} do I need?" for i in ingredient_list])
    questions.extend([f"How much {i} do I need for this recipe?" for i in ingredient_list])
    questions.extend([f"How much {i} is there?" for i in ingredient_list])
    questions.extend([f"How much {i} is there in this recipe?" for i in ingredient_list])
    # just choose three random questions to generate
    random_questions = random.sample(questions, 3)
    for idx, question in enumerate(random_questions):
        while idx >= len(ingredient_list):
            # Each question was created by iterating over the ingredients list so we know when we get to the end of the
            # list we have reached the end of a set of questions and need to get the index back in check
            idx -= len(ingredient_list)
        ingredient = ingredient_list[idx]
        indexable_list = get_indexable_list(question, tokenized_recipe)

        # Need to check that the index occurs after the SEP token so that we aren't looking for the answer in the
        # question
        sep_index = indexable_list.index('[SEP]')
        if len(ingredient.split()) == 1:
            # If the ingredient is a single word, we can just search for it in the list
            ingredient_index = indexable_list.index(ingredient)
        else:
            ingredient_index_options = [i for i, x in enumerate(indexable_list) if x == ingredient.split()[0]]
            # Remove all options that are before the SEP token
            ingredient_index_options = [i for i in ingredient_index_options if i > sep_index]
            # Check if any options are followed by the next word in the ingredient
            ingredient_index_options_best = [i for i in ingredient_index_options if
                                             indexable_list[i + 1] == ingredient.split()[1]]
            if ingredient_index_options_best:
                # Maybe keep iterating until we find the best match but add that later
                ingredient_index = ingredient_index_options_best[0]
            else:
                ingredient_index = ingredient_index_options[0]
        # Get the index of the token [INGITEM] or [INGSTART] occuring before the ingredient in indexable_list
        # whichever is closest Get the index of all [INGITEM] in indexable_list
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
                                       'recipe_index': recipe_index,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset


#################  Ingredient Question Generation ##############################


def get_indexable_list(question, tokenized_recipe):
    question_list = question.split()
    question_list.append('[SEP]')
    # prepend '[CLS]' to the start of the question
    question_list.insert(0, '[CLS]')
    tokenized_recipe_list = tokenized_recipe.split()
    question_list.extend(tokenized_recipe_list)
    return question_list


if __name__ == "__main__":
    # format_raw_recipe_dataset()
    # time running create_question_set()
    start = time.time()
    create_ingredients_question_set()
    end = time.time()
    print(f"Time taken: {round(end - start, 2)} seconds")
    # print(random_unit_of_measurement())
