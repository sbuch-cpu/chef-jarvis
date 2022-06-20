import torch
from utilities import get_path, get_model_and_tokenizer
import os


def ask_distilBERT(question, context):
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(question, context, return_tensors="pt")
    print(inputs['input_ids'])
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)


def fine_tuned_distilBERT(question, context):
    module_path = get_path('chef-jarvis')
    model, tokenizer = get_model_and_tokenizer()
    model.state_dict = torch.load(os.path.join(module_path, 'sandbox/model_state_dict.pt'))
    # tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(module_path, 'sandbox/tokenizer.pt'))
    # model = DistilBertForQuestionAnswering.from_pretrained(os.path.join(module_path, 'sandbox/model.pt'))
    # model = torch.load(os.path.join(module_path, 'sandbox/model.pt'), map_location=torch.device('cpu'))
    # torch.save(model.state_dict(), os.path.join(module_path, 'sandbox/model_state_dict.pt'))
    # tokenizer = torch.load(os.path.join(module_path, 'sandbox/tokenizer.pt'), map_location=torch.device('cpu'))

    model.eval()
    inputs = tokenizer(question, context, return_tensors="pt")
    # print(inputs['input_ids'])
    # print([tokenizer.decode(i) for i in inputs['input_ids']])

    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs.start_logits)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    print(answer_start_index, answer_end_index)
    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    print(predict_answer_tokens)
    return tokenizer.decode(predict_answer_tokens)

