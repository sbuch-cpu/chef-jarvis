import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from scraping import recipe_scraper, flatten

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


def ask_distilBERT(question, context):

    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    print(tokenizer.decode(predict_answer_tokens))


def main():
    ingredients, recipe = recipe_scraper('https://www.foodnetwork.com/recipes/stuffed-green-peppers-with-tomato-sauce-recipe-1910765')
    ingredients = '. '.join(ingredients)
    print(ingredients)
    ask_distilBERT('How much onion', ingredients)


if __name__ == "__main__":
    main()
