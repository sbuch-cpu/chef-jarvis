import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import sounddevice as sd

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# def listen():
#     audio = sd.rec(frames=32000, samplerate=16000)
#     sd.play(audio)
#     return audio


def map_to_array(batch):
    speech, _ = sf.read(batch)
    return speech


def main():
    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # ds = ds.map(map_to_array)
    file = map_to_array('/Users/Buchanan/Desktop/Python/chef-jarvis/delme_rec_unlimited_m19excvm.wav')
    inputs = processor(file, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # generated_ids = model.generate(inputs=inputs["input_values"])

    transcription = processor.batch_decode(predicted_ids)
    print(transcription)


def example():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # audio file is decoded on the fly
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    print(transcription[0])


if __name__ == "__main__":
    example()
