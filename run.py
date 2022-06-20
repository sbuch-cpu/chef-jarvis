from model_implementation.speech_text_conversion import speech_to_audio, gTTS_model
from model_implementation.question_answer import ask_distilBERT
from model_implementation.scraping import recipe_scraper
from utilities.utilities import custom_tokenize_recipe


def main():
    print("Starting scraping URL")
    recipe = recipe_scraper(
        'https://www.jamieoliver.com/recipes/chicken-recipes/tender-and-crisp-chicken-legs-with-sweet-tomatoes/')
    print("Finishing Scraping URL")
    print("#" * 80)
    print("Starting STT")
    question = speech_to_audio(4)
    print("Finishing STT")
    print("#" * 80)
    print(question)
    print("Asking Jarvis")
    answer = ask_distilBERT(question + '?', recipe['Ingredients'])
    print(f"Jarvis said: {answer}")
    print("#" * 80)
    print("Starting TTS")
    gTTS_model(answer)
    print("Finished")


def test(question):
    print("Starting scraping URL")
    recipe = recipe_scraper(
        'https://www.jamieoliver.com/recipes/chicken-recipes/tender-and-crisp-chicken-legs-with-sweet-tomatoes/')
    print("Finishing Scraping URL")
    print("#" * 80)
    tokenized_recipe = custom_tokenize_recipe(recipe['ingredients'], recipe['instructions'])
    print(tokenized_recipe)
    print("Asking Jarvis")
    answer = ask_distilBERT(question, tokenized_recipe, test=True)
    print(answer)
    print("Finished")


if __name__ == "__main__":
    # main()
    test('How much black pepper do I need?')
