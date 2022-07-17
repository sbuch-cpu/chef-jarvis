import random
from utilities.utilities import get_indexable_list


def get_step_number_question(tokenized_recipe, number_of_steps, recipe_index):
    json_formatted_dataset = []
    questions = []
    questions.extend([(f'What is step {step_number}?', step_number) for step_number in range(1, number_of_steps + 1)])
    questions.extend([(f'What is step {step_number} for this recipe?', step_number)
                      for step_number in range(1, number_of_steps + 1)])
    # just choose three random questions to generate
    try:
        random_questions = random.sample(questions, 3)
    except ValueError:
        random_questions = questions
        print(f'Could not sample {len(questions)} questions')
    for question, step_number in random_questions:
        indexable_list = get_indexable_list(question, tokenized_recipe)
        # Get the index of the token [INSTSTART] in indexable_list
        step_start_index = indexable_list.index('[INSTSTART]')
        # Get the index of all [INSTITEM] tokens in indexable_list
        step_item_indexes = [i for i, x in enumerate(indexable_list) if x == '[INSTITEM]']
        end_index = step_item_indexes[step_number - 1]
        if step_number == 1:
            start_index = step_start_index + 1
        else:
            start_index = step_item_indexes[step_number - 2] + 1
        answer = ' '.join(indexable_list[start_index:end_index + 1])
        end_index -= 1
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'recipe_index': recipe_index,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset


def get_ingredient_from_step_question(method, ingredients, recipe_index, step_start_end_indexes, step_number):
    json_formatted_dataset = []
    ingredients_in_step = []
    indexable_list = method.split()
    # Split each item in the ingredients list into a list of words then flatten the list
    flattented_ingredients = [ingredient.split() for ingredient in ingredients]
    # indexable_list = [re.sub(r'[^\w\s]', '', i) for i in indexable_list]
    print('##############################################################################')
    print(ingredients)
    print(indexable_list)
    for ingredient in ingredients:
        if len(ingredient.split()) == 1:
            # If the ingredient is a single word, we can just search for it in the list
            # ingredient_index = indexable_list.index(ingredient)
            ingredient_index_options = [i for i, x in enumerate(indexable_list) if x == ingredient]
        else:
            ingredient_index_options = [i for i, x in enumerate(indexable_list) if x == ingredient.split()[0]]

            # Check if any options are followed by the next word in the ingredient
            ingredient_index_options_best = [i for i in ingredient_index_options if
                                             indexable_list[i + 1] == ingredient.split()[1]]
            if ingredient_index_options_best:
                # Maybe keep iterating until we find the best match but add that later
                ingredient_index_options = ingredient_index_options_best
            # Check for any words in ingredient that only occur once in flattented_ingredients
            searchable_words = [word for word in ingredient.split() if flattented_ingredients.count(word) == 1]
            # add support to check that the word is an ingredient
            # check for plural/nonplural version of the word
            # Look for each searchable word in the method and extend the ingredient_index_options
            for word in searchable_words:
                ingredient_index_options.extend([i for i, x in enumerate(indexable_list) if x == word])
        ingredients_in_step.extend(ingredient_index_options)
    print(method)
    print(ingredients_in_step)
    return json_formatted_dataset
