import requests
from bs4 import BeautifulSoup
import json
import re
from anyascii import anyascii


def recipe_scraper(url):
    ingredients = None
    recipe = None
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")
    # RECIPE METHOD
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
    # print(json_scripts)
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

    if not ingredients or not recipe:
        # for i in ['image', 'span', 'footer', 'nav', 'button', 'option', 'noscript', 'script', 'iframe', 'form',
        #           'path', 'svg', 'aside', 'source', 'img', 'picture', 'ui-promo', 'style']:
        #     for s in soup.select(i):
        #         s.extract()
        for x in soup.find_all():
            # fetching text from tag and remove whitespaces
            if len(x.get_text(strip=True)) == 0:
                # Remove empty tag
                x.extract()
        print(soup.body.prettify())
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

    if not ingredients or not recipe:
        print(ingredients, recipe)
        raise Exception('This recipe format is not supported')
    if isinstance(recipe, str):
        recipe = recipe.split('</li><li>')

    ingredients = clean_list(ingredients)
    recipe = clean_list(recipe)

    return ingredients, recipe


def clean_list(dirty_list):
    # dirty_list = [i.encode("ascii", "ignore").decode() for i in dirty_list]  # remove non-ascii characters
    dirty_list = [anyascii(i) for i in dirty_list]  # convert non-ascii characters
    dirty_list = [re.sub("[<].*?[>]", " ", i).strip() for i in dirty_list]  # remove any remaining html tags
    dirty_list = [re.sub("\n", " ", i).strip() for i in dirty_list]  # remove line breaks
    dirty_list = [' '.join(i.split()) for i in dirty_list]  # remove any multiple spaces ie '    ' becomes ' '
    dirty_list = [i.replace(" ,", ",") for i in dirty_list]  # remove whitespace before commas
    dirty_list = [i.replace(" .", ".") for i in dirty_list]  # remove whitespace before periods
    dirty_list = [i.replace(" !", "!") for i in dirty_list]  # remove whitespace before exclamation points
    dirty_list = [i.replace(" ?", "?") for i in dirty_list]  # remove whitespace before question marks
    cleaned_list = [re.sub("\xa0", " ", i).strip() for i in dirty_list]
    return cleaned_list

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
