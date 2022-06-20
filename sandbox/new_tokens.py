from web_scraping.scraping import recipe_scraper
from distilbert_custom.distilBERT_attempt import fine_tuned_distilBERT


def custom_tokenize_recipe(ingredients, instructions):
    # print(recipe)
    ing_start = '[INGSTART]'
    ing_item = '[INGITEM]'
    inst_start = '[INSTSTART]'
    inst_item = '[INSTITEM]'

    ingredients = ing_item.join(ingredients) + ing_item
    instructions = inst_item.join(instructions) + inst_item
    tokenized_recipe = ing_start + ingredients + inst_start + instructions
    return tokenized_recipe


def main():
    recipe = recipe_scraper('https://www.jamieoliver.com/recipes/chicken-recipes/tender-and-crisp-chicken-legs-with-sweet-tomatoes/')
    tokenized_recipe = custom_tokenize_recipe(recipe['ingredients'], recipe['instructions'])
    # print(tokenized_recipe)
    question = "What is step 3?"
    print(fine_tuned_distilBERT(question, tokenized_recipe))
    return


if __name__ == "__main__":
    main()
