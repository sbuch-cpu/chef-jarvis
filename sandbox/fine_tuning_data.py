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
    training_data['ingredients'] = training_data['ingredients'].apply(add_units_to_ingredients)
    training_data['tokenized'] = training_data.apply(lambda x: custom_tokenize_recipe(x.ingredients, x.steps), axis=1)
    print(training_data['tokenized'].values[0])
    training_data.to_csv('../training_data/tokenized_recipes.csv')
    # print(training_data)
    return


def add_units_to_ingredients(ing_list):
    return [f"{random_unit_of_measurement()} of {i}" for i in ing_list]


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
    main()
    # print(random_unit_of_measurement())
