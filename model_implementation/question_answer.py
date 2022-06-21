import os

import torch
from utilities.utilities import get_model_and_tokenizer, get_model_and_tokenizer_pretrained
from utilities.path_utilities import PATHS


def ask_distilBERT(question, context, test=False):
    if test:
        return fine_tuned_distilBERT(question, context, 'distilbert-custom-backward-AdamW-1000-0.2-20-5-t-p')
    else:
        return default_distilBERT(question, context)


def default_distilBERT(question, context):
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(question, context, return_tensors="pt")
    print(inputs['input_ids'])
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)


def fine_tuned_distilBERT(question, context, model_name, model_dir_path=PATHS['MODEL']):
    model_path = os.path.join(model_dir_path, model_name)
    model, tokenizer = get_model_and_tokenizer_pretrained(model_path)

    model.eval()
    inputs = tokenizer(question, context, return_tensors="pt")
    print(tokenizer.decode(inputs.input_ids[0, :]))
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs.start_logits)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    # print(answer_start_index, answer_end_index)
    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    # print(predict_answer_tokens)
    return tokenizer.decode(predict_answer_tokens)
