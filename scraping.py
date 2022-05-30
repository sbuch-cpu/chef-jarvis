import requests
from bs4 import BeautifulSoup
import json
import re
from anyascii import anyascii


def recipe_scraper(url):
    """
    Function to retrieve the instructions and ingredients from a recipe webpage given a url.

    :param url: string of the url of a website a recipe.
    :type url: str
    :return: A tuple containing the list of ingredients and the list of recipe instructions
    :rtype: tuple
    """

    # Get the HTML code from the webpage and pass it into BeautifulSoup
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")

    # First extraction method (From JSON object)
    ingredients, recipe = recipe_from_json(soup)

    # Second extraction method (From HTML)
    if not ingredients or not recipe:
        ingredients, recipe = recipe_from_html(soup)

    # If neither extraction method works then raise and exception and error out
    if not ingredients or not recipe:
        print(ingredients, recipe)
        raise Exception('This webpage format is not supported')

    # If its a string, hopefully the items are broken up into list items to separate strings
    recipe = break_string(recipe, '</li><li>')  # Break into list on end and starting list item tags
    ingredients = break_string(ingredients, '</li><li>')  # Break into list on end and starting list item tags

    # Clean each item in both the recipe and ingredients lists
    recipe = [clean_string(step) for step in recipe]
    ingredients = [clean_string(ingredient) for ingredient in ingredients]

    return ingredients, recipe


def break_string(string, delimiter):
    """
    Function to break a string into a list based on a given delimiter.
    :param string: String to be split
    :type string: str
    :param delimiter: Delimiter on which
    :type delimiter: str
    :return:
    :rtype:
    """
    if isinstance(string, str):
        string = string.split(delimiter)
    return string


def recipe_from_json(soup, ingredients=None, recipe=None):
    script_list = soup.find_all("script")
    json_scripts = []
    for script in script_list:
        script = script.text
        if not isinstance(script, str):
            continue
        try:
            recipe_json = json.loads(script)
        except json.decoder.JSONDecodeError:
            continue
        json_scripts.append(recipe_json)
    json_scripts = flatten(json_scripts)

    for script in json_scripts:
        if not isinstance(script, dict):
            continue
        if all(k in script.keys() for k in ('recipeIngredient', 'recipeInstructions')):
            ingredients = script['recipeIngredient']
            recipe = script['recipeInstructions']
            if hasattr(recipe, '__iter__') and not isinstance(recipe, str):
                if isinstance(recipe, dict):
                    recipe = pull_from_dict(recipe, ['text'])[0]
                else:
                    recipe_list = []
                    for r in recipe:
                        recipe_list.append(pull_from_dict(r, ['text'])[0])
                    recipe = recipe_list
    return ingredients, recipe


def recipe_from_html(soup, ingredients=None, recipe=None):
    # for i in ['image', 'footer', 'nav', 'button', 'option', 'noscript', 'script', 'iframe', 'form',
    #           'path', 'svg', 'aside', 'source', 'img', 'picture', 'ui-promo', 'style']:
    #     for s in soup.select(i):
    #         s.extract()
    for x in soup.find_all():
        # fetching text from tag and remove whitespaces
        if len(x.get_text(strip=True)) == 0:
            # Remove empty tag
            x.extract()
    recipe_keywords = ['Directions', 'Method', 'Steps', 'To Prepare', 'Instructions']
    ingredients_keywords = ['Ingredients', 'Ingredient']
    header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'b']

    for header in header_tags:
        tags = soup.body.find_all(header)
        for tag in tags:
            if not recipe:
                recipe = list_keyword_scraping(tag, recipe_keywords)
            if not ingredients:
                ingredients = list_keyword_scraping(tag, ingredients_keywords)
    return ingredients, recipe


def clean_string(dirty_string):
    # dirty_string = dirty_string.encode("ascii", "ignore").decode()  # remove non-ascii characters
    dirty_string = anyascii(dirty_string)  # convert non-ascii characters
    dirty_string = re.sub("[<].*?[>]", " ", dirty_string).strip()  # remove any remaining html tags
    dirty_string = re.sub("\n", " ", dirty_string).strip()  # remove line breaks
    dirty_string = re.sub("\xa0", " ", dirty_string).strip()  # remove specific unwanted characters
    dirty_string = ' '.join(dirty_string.split())  # remove any multiple spaces ie '    ' becomes ' '
    dirty_string = dirty_string.replace(" ,", ",")  # remove whitespace before commas
    dirty_string = dirty_string.replace(" .", ".")  # remove whitespace before periods
    dirty_string = dirty_string.replace(" !", "!")  # remove whitespace before exclamation points
    dirty_string = dirty_string.replace(" ?", "?")  # remove whitespace before question marks
    cleaned_string = dirty_string  # for clarity
    return cleaned_string


def list_keyword_scraping(tag, keywords_list):
    itemized_list = None
    for keyword in keywords_list:
        if keyword.strip().lower() in tag.text.strip().lower():
            list_item = tag.find_next('li')
            if list_item:
                desired_list = list_item.parent
                desired_list = desired_list.find_all('li')
                itemized_list = [item.text for item in desired_list]
            else:
                first_item = tag.find_next('p')
                itemized_list = [first_item.text]
                steps = first_item.find_next_siblings("p")
                for step in steps:
                    itemized_list.append(step.text)
            if itemized_list:
                break
    return itemized_list


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, (list, tuple)):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def pull_from_dict(obj, keys):
    recipe = []
    if isinstance(obj, dict):
        valid_keys = [k for k in keys if k in obj.keys()]
        if valid_keys:
            for k in valid_keys:
                recipe.append(obj[k])
        else:
            print('Key not found ...')
            print(f'Available keys: {obj.keys()}')
    return recipe


if __name__ == "__main__":
    # print(recipe_scraper('https://www.bingingwithbabish.com/recipes/2017/1/18/kevinschili?rq=kevin'))
    # print(recipe_scraper('https://www.jamieoliver.com/recipes/eggs-recipes/hollandaise-sauce/'))
    print(recipe_scraper('https://www.foodnetwork.com/recipes/stuffed-green-peppers-with-tomato-sauce-recipe-1910765'))
