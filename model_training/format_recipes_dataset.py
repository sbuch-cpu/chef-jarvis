import random
import re
from pandas import pd
from utilities.utilities import custom_tokenize_recipe
from utilities.path_utilities import PATHS


def format_raw_recipe_dataset(raw_recipes_path=PATHS['RAW_RECIPES'], 
                              tokenized_recipes_path=PATHS['tokenized_recipes']):
    """
    This function formats the raw recipe dataset retrieved from
    https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download&select=RAW_recipes.csv
    and tokenizes the recipes in the format desired by the model. [SEP] and [CLS] tokens are omitted. The new dataset is
    saved in the training_dataset folder under the name tokenized_recipes.csv.
    """
    training_data = pd.read_csv(raw_recipes_path)
    # Remove the unnecessary columns from the dataset
    training_data.drop(['minutes', 'id', 'contributor_id', 'submitted', 'tags',
                        'nutrition', 'n_steps', 'description', 'n_ingredients'], axis=1, inplace=True)
    print('Changing to lists...')
    # replace all ampersands with ' &' so that the model can understand the steps
    training_data['steps'] = training_data['steps'].apply(lambda x: x.replace('&', ' &'))
    # Convert the data in 'steps' and 'ingredients' from a string to a list of strings using regex
    training_data['steps'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.steps), axis=1)
    training_data['ingredients'] = training_data.apply(lambda x: re.findall(r"'\s*([^']*?)\s*'", x.ingredients), axis=1)
    # Save the ingredients list without the units of measurement
    training_data['raw_ingredients'] = training_data['ingredients']
    print('Adding units of measurement...')
    # add units of measurement to the ingredients list so the model can understand the ingredients
    # These units are randomly generated, so they do not always make sense for the ingredient they are associated with
    # This should not affect the accuracy of the model ... hopefully
    training_data['ingredients'] = training_data['ingredients'].apply(add_units_to_ingredients)
    print('Tokenizing...')
    # Tokenize the recipes
    training_data['tokenized'] = training_data.apply(lambda x: custom_tokenize_recipe(x.ingredients, x.steps), axis=1)
    print('Saving...')
    # Save the tokenized dataset
    training_data.to_csv(tokenized_recipes_path)


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
    unit_list.extend(['tsp', 'tbsp', 'lb', 'oz', 'g', 'mg', 'kg', 'fl oz', 'ml', 'L', 't', 'qt', 'c'])
    unit = random.choice(unit_list)  # Choose a random unit of measurement

    # If the unit of measurement can be pluralized and the number is not one then add an 's' to the end of the unit of
    # measurement
    if unit in pluralizable_units and whole != 1 and (num or whole != 0):
        unit = unit + 's'

    return f"{number} {unit} of"
