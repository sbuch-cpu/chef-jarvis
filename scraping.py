import requests
from bs4 import BeautifulSoup
import json
import re


def get_jamie_recipe(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")
    # RECIPE METHOD
    recipe_wrapper = soup.find("script", {"type": "application/ld+json"})
    recipe_json = json.loads(recipe_wrapper.text)
    ingredients = recipe_json['recipeIngredient']
    recipe = recipe_json['recipeInstructions']
    recipe = re.sub("[<].*?[>]", " ", recipe)
    print("#####################  INGREDIENTS  #######################")
    print(ingredients)
    print("#####################  RECIPE  #######################")
    print(recipe)

    return


if __name__ == "__main__":
    get_jamie_recipe('https://www.jamieoliver.com/recipes/eggs-recipes/hollandaise-sauce/')
