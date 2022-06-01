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
    :rtype: (list[str], list[str])
    """

    # Get the HTML code from the webpage and pass it into BeautifulSoup
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")

    # First extraction method (From JSON object)
    # This method is first because it is less error prone than pulling from HTML
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
    recipe = [clean_string(step, regex_remove=["[<].*?[>]", "\n", "\xa0"]) for step in recipe]
    ingredients = [clean_string(ingredient, regex_remove=["[<].*?[>]", "\n", "\xa0"]) for ingredient in ingredients]

    return ingredients, recipe


def break_string(string, delimiter):
    """
    Function to break a string into a list based on a given delimiter.

    :param string: String to be split
    :type string: str
    :param delimiter: Delimiter on which the given string will be broken into list elements
    :type delimiter: str
    :return: List containing list elements of the original string
    :rtype: list[str]
    """
    # Only break the string if the string variable is a string (say that three times fast)
    if isinstance(string, str):
        string = string.split(delimiter)  # split the string into a list on the given delimiter
    return string


def recipe_from_json(soup, ingredients=None, recipe=None):
    """
    Function to extract a list of recipe instructions and a list of ingredients from an HTML page. Looking into the
    source code of a lot of recipe webpages, it appears that it is fairly common for recipe webpages to use a similar
    json structure to pass information to their Javascript code. Every instance of this that I found involved a list
    of ingredients being accessible by the "recipeIngredient" key and the recipe method being accessible by the
    "recipeInstructions" key. The ingredients always appeared in the form of a list strings however the method was
    sometimes a list of strings, sometimes a list of objects with the desired string available through the 'text' key,
    and sometimes a long string with items separated by HTML list tags.

    :param soup: BeautifulSoup object of the HTML text that you want to pull the recipe and ingredients from.
    :type soup: BeautifulSoup
    :param ingredients: Current list of ingredients, None by default.
    :type ingredients: list[str]
    :param recipe: Current list of recipe steps, None by default
    :type recipe: list[str]
    :return: List of Ingredients and list of recipe steps if any where found, otherwise returns None and None.
    :rtype: (list[str], list[str]) or (None, None)
    """

    # Create a list of all the script tags in the given HTML code
    script_list = soup.find_all("script")
    json_scripts = []  # initialize an empty list to be populated by json objects

    # Iterate through all the given script tags
    for script in script_list:
        script = script.text  # pull the text out from within the script tag
        if not isinstance(script, str):
            continue  # move on if its not a string
        try:
            recipe_json = json.loads(script)  # Try to turn the string into a json object
        except json.decoder.JSONDecodeError:  # If it is not the right format for JSON then move on
            continue
        json_scripts.append(recipe_json)

    # Flatten any nested lists in the json_scripts list ie. turn [[[1], [2], {3:4}], {5:6}] into [1, 2, {3:4}, {5:6}]
    json_scripts = flatten(json_scripts)

    # Iterate through the list of found JSON objects to try to find the recipe and ingredients
    for script in json_scripts:
        if not isinstance(script, dict):
            continue  # move on if its not a dictionary

        # Check to see if both the recipeIngredient and recipeInstructions keys are in the dict in question
        if all(k in script.keys() for k in ('recipeIngredient', 'recipeInstructions')):
            # Pull the ingredients and recipe items out of the dictionary
            ingredients = script['recipeIngredient']
            recipe = script['recipeInstructions']

            # Check to make sure that the list is a list of text not a list of dictionaries, and fix if needed
            ingredients = check_for_dict(ingredients, ['text'])
            recipe = check_for_dict(recipe, ['text'])

    return ingredients, recipe


def recipe_from_html(soup, ingredients=None, recipe=None):
    """
    Function to pull the recipe steps and list of ingredients from HTML source code. This function is based off of the
    idea that the recipe steps and ingredients are likely to be preceded by one of a set of predictable keywords which
    would be wrapped in some tags to make it more prominent on the webpage. Once the location of the header is known,
    the steps and ingredients can then be found by proximity to the header.

    :param soup: BeautifulSoup object of the HTML text that you want to pull the recipe and ingredients from.
    :type soup: BeautifulSoup
    :param ingredients: Current list of ingredients, None by default.
    :type ingredients: list[str]
    :param recipe: Current list of recipe steps, None by default
    :type recipe: list[str]
    :return: List of Ingredients and list of recipe steps if any where found, otherwise returns None and None.
    :rtype: (list[str], list[str]) or (None, None)
    """
    # Iterate through and remove empty tags that are just meant for spacing
    for x in soup.find_all():
        # fetching text from tag and remove whitespaces
        if len(x.get_text(strip=True)) == 0:
            # Remove empty tag
            x.extract()

    # Keywords and tags that will be searched through to find the recipe nad instructions on the webpage
    recipe_keywords = ['Directions', 'Method', 'Steps', 'To Prepare', 'Instructions']
    ingredients_keywords = ['Ingredients', 'Ingredient']
    header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'b']

    for header in header_tags:
        tags = soup.body.find_all(header)  # find all instances of the header
        for tag in tags:
            # If the recipe has not been found check to see if this tag contains the header for the recipe
            if not recipe:
                recipe = list_keyword_scraping(tag, recipe_keywords)

            # If the recipe has not been found check to see if this tag contains the header for the recipe
            if not ingredients:
                ingredients = list_keyword_scraping(tag, ingredients_keywords)
    return ingredients, recipe


def clean_string(dirty_string, regex_remove=None, remove_nonascii=False):
    """
    Function to remove artifacts remaining from the conversion from HTML to text. This includes left over HTML tags,
    non-ascii characters, extra spaces, spaces before punctuation, or specific characters/strings that should be
    removed using regex.

    :param dirty_string: string that is not formatted properly and contains unwanted characters
    :type dirty_string: str
    :param regex_remove: list of specific characters/strings formatted for regex that should be removed using regex
    :type regex_remove: list[str] or None
    :param remove_nonascii: should nonascii characters be removed or converted to their ascii equivalent?
    :type remove_nonascii: bool
    :return: string properly formatted.
    :rtype: str
    """
    # Written like this to avoid regex_remove being mutable
    if regex_remove is None:
        regex_remove = []
    if remove_nonascii:
        dirty_string = dirty_string.encode("ascii", "ignore").decode()  # remove non-ascii characters
    else:
        dirty_string = anyascii(dirty_string)  # convert non-ascii characters
    for i in regex_remove:
        dirty_string = re.sub(i, " ", dirty_string).strip()  # remove any specific strings
    dirty_string = ' '.join(dirty_string.split())  # remove any multiple spaces i.e. '    ' becomes ' '
    dirty_string = dirty_string.replace(" ,", ",")  # remove whitespace before commas
    dirty_string = dirty_string.replace(" .", ".")  # remove whitespace before periods
    dirty_string = dirty_string.replace(" !", "!")  # remove whitespace before exclamation points
    dirty_string = dirty_string.replace(" ?", "?")  # remove whitespace before question marks
    cleaned_string = dirty_string  # for clarity in variable names
    return cleaned_string


def list_keyword_scraping(tag, keywords_list):
    """
    Function to pull a list below a heading from HTML source code. First the given heading tag is checked to see if it
    matches any of the keyword known to be in the heading preceding the desired list. If the tag contains one of the
    key words then the next list item <li> tag is found. If that list is within 200 lines of

    :param tag:
    :type tag:
    :param keywords_list:
    :type keywords_list:
    :return:
    :rtype:
    """
    itemized_list = None
    for keyword in keywords_list:
        if keyword.strip().lower() in tag.text.strip().lower():
            list_item = tag.find_next('li')
            if list_item and (list_item.sourceline - tag.sourceline) < 200:
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
    for item in x:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def check_for_dict(list_of_dict, keys):
    # print(list_of_dict)
    if hasattr(list_of_dict, '__iter__') and not isinstance(list_of_dict, str):
        if isinstance(list_of_dict, dict):
            list_of_dict = pull_from_dict(list_of_dict, keys)[0]
        else:
            cleaned_list = []
            for i in list_of_dict:
                # print(i)
                list_of_items = pull_from_dict(i, keys)
                for x in list_of_items:
                    cleaned_list.append(x)
            list_of_dict = cleaned_list
            # print(list_of_dict)
    return list_of_dict


def pull_from_dict(obj, keys):
    recipe = []
    if isinstance(obj, dict):
        valid_keys = [k for k in keys if k in obj.keys()]
        if valid_keys:
            for k in valid_keys:
                print(obj[k])
                recipe.append(obj[k])
        else:
            print('Key not found ...')
            print(f'Available keys: {obj.keys()}')
    elif isinstance(obj, str):
        recipe = [obj]
    else:
        recipe = obj
    return recipe


if __name__ == "__main__":
    # print(recipe_scraper('https://www.bingingwithbabish.com/recipes/2017/1/18/kevinschili?rq=kevin'))
    # print(recipe_scraper('https://www.jamieoliver.com/recipes/eggs-recipes/hollandaise-sauce/'))

    # STILL NEED TO WORK ON CASES LIKE THIS ONE
    print(recipe_scraper('https://www.foodnetwork.com/recipes/stuffed-green-peppers-with-tomato-sauce-recipe-1910765'))
