from model_implementation.speech_text_conversion import speech_to_audio, gTTS_model
from model_implementation.question_answer import ask_distilBERT
from model_implementation.scraping import recipe_scraper


def main():
    print("Starting scraping URL")
    recipe = recipe_scraper(
        'https://www.jamieoliver.com/recipes/chicken-recipes/tender-and-crisp-chicken-legs-with-sweet-tomatoes/')
    print("Finishing Scraping URL")
    print("#"*80)
    instructions = '. '.join(recipe['instructions'])
    print("Starting STT")
    question = speech_to_audio(4)
    print("Finishing STT")
    print("#"*80)
    print(question)
    print("Asking Jarvis")
    answer = ask_distilBERT(question+'?', instructions)
    print(f"Jarvis said: {answer}")
    print("#"*80)
    print("Starting TTS")
    gTTS_model(answer)
    print("Finished")


if __name__ == "__main__":
    main()
