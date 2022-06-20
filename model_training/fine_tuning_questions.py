import time
import pandas as pd

from model_training.format_recipes_dataset import format_raw_recipe_dataset
from model_training.question_generation.ingredient_questions import full_ingredient_list_question_generation, \
    get_specific_ingredient_question
import re
import os
import json

from model_training.question_generation.step_questions import get_step_number_question
from utilities.path_utilities import PATHS


#################  Ingredient Question Generation ##############################
def create_ingredients_question_set(new=True,
                                    ingredients_questions=True,
                                    steps_questions=True,
                                    question_set_path=PATHS['QUESTION_SET'],
                                    tokenized_recipes_path=PATHS['TOKENIZED_RECIPES']):
    """
    This function creates a set of questions for the user to ask chef jarvis about the ingredients of a recipe.
    """
    # if the tokenized_recipes.csv file does not exist then run format_raw_recipe_dataset() to create it
    if not os.path.isfile(tokenized_recipes_path):
        format_raw_recipe_dataset()
    # Read the tokenized dataset
    training_data = pd.read_csv(tokenized_recipes_path)
    # Convert the data in 'raw_ingredients' to a list of strings
    training_data['raw_ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.raw_ingredients),
                                                           axis=1)
    training_data['steps'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.steps), axis=1)
    # Pull the tokenized recipe and the raw ingredients from the dataset
    tokenized = training_data['tokenized'].values
    ingredients = training_data['raw_ingredients'].values
    steps = training_data['steps'].values
    if not new:
        # Read the existing question set json file using the json module
        with open(question_set_path, 'r') as f:
            json_formatted_dataset = json.load(f)
    else:
        json_formatted_dataset = []
    # Generate question answer pairs from multiple questions per recipe
    for i, recipe in enumerate(tokenized):
        # print the progress of the loop for every 100 recipes
        if i % 100 == 0:
            print(f"{i}/{len(tokenized)} --- {round(i / len(tokenized) * 100, 2)}%")
        if ingredients_questions:
            # Create a list of questions and answers for the ingredients of the recipe
            json_formatted_dataset.extend(question_generation_ingredients(recipe, ingredients[i], i))
        if steps_questions:
            # Create a list of questions and answers for the steps of the recipe
            json_formatted_dataset.extend(question_generation_steps(recipe, len(steps[i]), i))
    # Save the dataset
    print("Saving the dataset...")
    with open(question_set_path, 'w', encoding='utf-8') as f:
        # f.write(str(json_formatted_dataset))
        json.dump(json_formatted_dataset, f, ensure_ascii=False, indent=4)
    # print the number of question answer pairs generated
    print(len(json_formatted_dataset))


def question_generation_steps(tokenized_recipe, number_of_steps, recipe_index):
    tokenized_recipe = tokenized_recipe.replace('[', ' [').replace(']', '] ')
    json_formatted_dataset = get_step_number_question(tokenized_recipe, number_of_steps, recipe_index)
    return json_formatted_dataset


def question_generation_ingredients(tokenized_recipe, ingredients, recipe_index):
    """
    This function generates a set of questions for the user to ask the chef jarvis about the ingredients of a recipe.
    :param tokenized_recipe:
    :param ingredients:
    :param recipe_index:
    :return:
    """
    tokenized_recipe = tokenized_recipe.replace('[', ' [').replace(']', '] ')
    json_formatted_dataset = full_ingredient_list_question_generation(tokenized_recipe, recipe_index)
    json_formatted_dataset.extend(get_specific_ingredient_question(tokenized_recipe, ingredients, recipe_index))
    return json_formatted_dataset


if __name__ == "__main__":
    format_raw_recipe_dataset()
    # time running create_question_set()
    start = time.time()
    create_ingredients_question_set()
    end = time.time()
    print(f"Time taken: {round(end - start, 2)} seconds")
