import transformers
from ane_transformers.huggingface import distilbert as ane_distilbert
import torch
import coremltools as ct
import numpy as np
# crazy vibes
def main():
    model_name = 'distilbert-base-uncased'
   # model_name = "distilbert-base-uncased-finetuned-sst-2-english" #slightly different version of distilbert-would it work the same on ours? or should we switch to this one?
    baseline_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        return_dict=False,
        torchscript=True,
    ).eval()
    optimized_model = ane_distilbert.DistilBertForSequenceClassification(
        baseline_model.config).eval()
    optimized_model.load_state_dict(baseline_model.state_dict())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(
        ["Sample input text to trace the model"],
        return_tensors="pt",
        max_length=128,  # token sequence length
        padding="max_length",
    )
    traced_optimized_model = torch.jit.trace(
        optimized_model,
        (tokenized["input_ids"], tokenized["attention_mask"])
    )
    ane_mlpackage_obj = ct.convert(
        traced_optimized_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                f"input_{name}",
                shape=tensor.shape,
                dtype=np.int32,
            ) for name, tensor in tokenized.items()
        ],
    )
    out_path = "HuggingFace_ane_transformers_distilbert_seqLen128_batchSize1.mlpackage"
    ane_mlpackage_obj.save(out_path)

if __name__ == "__main__":
    main()