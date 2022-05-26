def distilBERT(question, text):
    import torch

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    inputs = tokenizer(question, text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    print(tokenizer.decode(predict_answer_tokens))


def main():
    distilBERT("Who was Jim Henson?", "Jim Henson was a nice puppet")

if __name__ == "__main__":
    main()