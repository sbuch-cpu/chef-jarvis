import requests
from bs4 import BeautifulSoup
import json
import re
from anyascii import anyascii


def recipe_scraper(url):
    """
    Function to retrieve the instructions and ingredients from a instructions webpage given a url.

    :param url: string of the url of a website a instructions.
    :type url: str
    :return: A tuple containing the list of ingredients and the list of instructions instructions
    :rtype: dict
    """

    # Get the HTML code from the webpage and pass it into BeautifulSoup

    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")
    title = ""
    potential_title_spots = ['title', 'h1']
    for tag in potential_title_spots:
        title_list = soup.find_all(tag)
        if title_list:
            title = title_list[0].text
            title = clean_string(title)
            break
    # First extraction instructions (From JSON object)
    # This instructions is first because it is less error prone than pulling from HTML
    ingredients, instructions = recipe_from_json(soup)

    # Second extraction instructions (From HTML)
    if not ingredients or not instructions:
        ingredients, instructions = recipe_from_html(soup)

    # If neither extraction instructions works then raise and exception and error out
    if not ingredients or not instructions:
        print(ingredients, instructions)
        raise Exception('This webpage format is not supported')

    # If its a string, hopefully the items are broken up into list items to separate strings
    instructions = break_string(instructions, '</li><li>')  # Break into list on end and starting list item tags
    ingredients = break_string(ingredients, '</li><li>')  # Break into list on end and starting list item tags

    # Clean each item in both the instructions and ingredients lists
    instructions = [clean_string(step) for step in instructions]
    ingredients = [clean_string(ingredient) for ingredient in ingredients]

    recipe = {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions,
        'url': url
    }

    return recipe


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
    :rtype: (list[str], list[str]) or (None, None) or (str, str)
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
            ingredients = parse_from_JSON(ingredients, ['text'])
            recipe = parse_from_JSON(recipe, ['text'])

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
                recipe = list_from_heading_keyword(tag, recipe_keywords)

            # If the recipe has not been found check to see if this tag contains the header for the recipe
            if not ingredients:
                ingredients = list_from_heading_keyword(tag, ingredients_keywords)
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
        regex_remove = ["[<].*?[>]", "\n", "\xa0", "&nbsp;"]
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


def list_from_heading_keyword(tag, keywords_list, line_threshold=200):
    """
    Function to pull a list below a heading from HTML source code. First the given heading tag is checked to see if it
    matches any of the keyword known to be in the heading preceding the desired list. If the tag contains one of the
    key words then the next list item <li> tag is found. If that list is within the line threshold of the tag being
    examined then it is assumed that it is the list being searched for. If the next list is further than the line
    threshold then it is assumed that the list is actually broken up into separate paragraphs rather than list items.
    In that case then the next <p> tag is searched for and all <p> tags in that level of HTML are assumed to be list
    items.

    :param tag: Tag being evaluated to check if it is the header of a list.
    :type tag: BeautifulSoup
    :param keywords_list: List of keywords expected to appear in the header of the desired list
    :type keywords_list: list[str]
    :param line_threshold: Distance between the header tag and the first list item for them to be presumed correlated
    :type line_threshold: int
    :return: List of items in the desired list
    :rtype: list[str] or None
    """
    # Initialize the list as None so that None will be returned if the tag is not the header of the desired list
    itemized_list = None

    # Check each of the possible keywords in the header
    for keyword in keywords_list:
        # remove any formatting that would prevent a match.
        # checking for 'in' rather than '==' so that keyword: is a match.
        if keyword.strip().lower() in tag.text.strip().lower():
            # Find the next HTML list after the list header
            list_item = tag.find_next('li')

            # Check to see if there is a list item and if it are close enough to the header
            if list_item and (list_item.sourceline - tag.sourceline) < line_threshold:
                desired_list = list_item.parent  # get the element containing the list
                desired_list = desired_list.find_all('li')  # get a list of all list items
                itemized_list = [item.text for item in desired_list]  # get the text from each list item

            else:  # If a list is not found close enough to the header than the list is probably a series of paragraphs
                first_item = tag.find_next('p')  # find the next paragraph item
                itemized_list = [first_item.text]
                steps = first_item.find_next_siblings("p")  # find all paragraphs on that level
                for step in steps:
                    itemized_list.append(step.text)  # get the text of each list item

            if itemized_list:  # if the list has been found no need to continue iterating through
                break
    return itemized_list


def flatten(x):
    """
    Function to flatten lists but leave dictionaries in tact.

    :param x: List of lists, dictionaries, and strings that needs to be flattened
    :type x: list[Any]
    :return: Flattened list
    :rtype: list[Any]
    """
    result = []
    # Function to iterate through the list
    for item in x:
        # Flatten any lists or tuples that are found
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            # append any strings or dicts that are found
            result.append(item)
    return result


def parse_from_JSON(item_from_json, keys):
    """
    Function to check if the item pulled from a JSON object is a string, a dictionary, a list of strings, or a list of
    dictionaries and handle accordingly. in the case that some elements contain dictionaries, possible keys to retrieve
    the desired text is passed through the keys variable.

    :param item_from_json: Item pulled from a JSON object that will be handled to retrieve strings
    :type item_from_json: list[Any]
    :param keys: List of keys used to try to retrieve desired text if a dictionary is found
    :type keys: list[str]
    :return: Desired text as a string or a list of strings
    :rtype: list[str] or str
    """
    # Check to see if the item pulled from json is iterable and making sure that its not just a string
    if hasattr(item_from_json, '__iter__') and not isinstance(item_from_json, str):
        # If its a dictionary then just pull the text out
        if isinstance(item_from_json, dict):
            # the pull_from_dict function returns a list of strings (if multiple elements were found from the keys)
            parsed_from_json = pull_from_dict(item_from_json, keys)

            # if only one element was found then a string should be returned rather than a one element long list
            # because the desired list is likely contained within that string
            if len(parsed_from_json) == 1:
                parsed_from_json = parsed_from_json[0]

        else:
            # If the item retrieved from json is a list or tuple then iterate over the items in that list and try to
            # pull items out of the dictionary (if they are strings then pull_from_dict just returns the string)
            parsed_from_json = []
            for i in item_from_json:
                list_of_items = pull_from_dict(i, keys)
                parsed_from_json.extend(list_of_items)
    else:
        parsed_from_json = item_from_json
    return parsed_from_json


def pull_from_dict(obj, keys):
    """


    :param obj:
    :type obj:
    :param keys:
    :type keys:
    :return:
    :rtype:
    """
    recipe = []
    if isinstance(obj, dict):
        valid_keys = [k for k in keys if k in obj.keys()]
        if valid_keys:
            for k in valid_keys:
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
    print(recipe_scraper('https://www.bingingwithbabish.com/recipes/2017/1/18/kevinschili?rq=kevin'))
    # print(recipe_scraper('https://tasty.co/recipe/sweet-potato-and-black-bean-burritos'))
    # print(recipe_scraper('https://cooking.nytimes.com/recipes/1023190-spaghetti-aglio-e-olio-e-fried-shallot?action=click&module=Public%20Recipebox&region=my-recipes&pgType=recipebox&rank=1'))
    # print(recipe_scraper('https://www.jamieoliver.com/recipes/eggs-recipes/hollandaise-sauce/'))
    # print(recipe_scraper('https://www.foodnetwork.com/recipes/stuffed-green-peppers-with-tomato-sauce-recipe-1910765'))
    # print(recipe_scraper('https://www.connoisseurusveg.com/baked-vegan-samosas/'))
