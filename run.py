from distilbert_custom.speech_to_text import speech_to_audio
from distilbert_custom.distilBERT_attempt import ask_distilBERT
from distilbert_custom.text_to_speech import gTTS_model
from web_scraping.scraping import recipe_scraper


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
