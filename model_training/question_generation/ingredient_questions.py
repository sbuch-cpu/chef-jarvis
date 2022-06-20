import random

from utilities import get_indexable_list


def full_ingredient_list_question_generation(tokenized_recipe, recipe_index):
    json_formatted_dataset = []
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


def get_specific_ingredient_question(tokenized_recipe, ingredient_list, recipe_index):
    json_formatted_dataset = []
    questions = []
    questions.extend([(f"How much {ing}?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} is needed?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} is needed for this recipe?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} do I need?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} do I need for this recipe?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} is there?", i) for i, ing in enumerate(ingredient_list)])
    questions.extend([(f"How much {ing} is there in this recipe?", i) for i, ing in enumerate(ingredient_list)])
    # just choose three random questions to generate
    random_questions = random.sample(questions, 3)
    for question, idx in random_questions:
        ingredient = ingredient_list[idx]
        indexable_list = get_indexable_list(question, tokenized_recipe)

        # Need to check that the index occurs after the SEP token so that we aren't looking for the answer in the
        # question
        sep_index = indexable_list.index('[SEP]')
        if len(ingredient.split()) == 1:
            # If the ingredient is a single word, we can just search for it in the list
            # ingredient_index = indexable_list.index(ingredient)
            ingredient_index_options = [i for i, x in enumerate(indexable_list) if x == ingredient]
            # Remove all options that are before the SEP token
            ingredient_index_options = [i for i in ingredient_index_options if i > sep_index]
            ingredient_index = ingredient_index_options[0]
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
            start_index = ing_start_index + 1
        else:
            start_index = ing_item_index_before[-1] + 1
        # Get the index of the token [INGITEM] occuring after the ingredient in indexable_list
        end_index = ing_item_index[len(ing_item_index_before)] + 1

        answer = ' '.join(indexable_list[start_index:end_index])
        end_index -= 1
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'recipe_index': recipe_index,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset