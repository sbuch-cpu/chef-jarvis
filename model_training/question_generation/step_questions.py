import random

from utilities import get_indexable_list


def generate_steps_question(tokenized_recipe, recipe_index):
    number_of_steps = 1


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
        answer = ' '.join(indexable_list[start_index:end_index+1])
        end_index -= 1
        json_formatted_dataset.append({'question': question,
                                       'answer': answer,
                                       'recipe_index': recipe_index,
                                       'start_index': start_index,
                                       'end_index': end_index})
    return json_formatted_dataset