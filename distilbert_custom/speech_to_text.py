import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import sounddevice as sd
import noisereduce as nr


hugging_face_mode = "patrickvonplaten/wav2vec2-base-100h-with-lm"

model = Wav2Vec2ForCTC.from_pretrained(hugging_face_mode)
processor = Wav2Vec2ProcessorWithLM.from_pretrained(hugging_face_mode)


def speech_to_audio(question_time):
    fs = 16000  # Sample rate
    seconds = question_time  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print('recording')
    sd.wait()  # Wait until recording is finished
    print('done recording')
    myrecording = myrecording.flatten()
    myrecording = nr.reduce_noise(y=myrecording, sr=fs)
    return audio_to_transcript(myrecording, sampling_rate=fs)


def audio_to_transcript(audio, sampling_rate=16000):
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    transcription = processor.batch_decode(logits.numpy()).text
    return transcription[0].lower()


if __name__ == "__main__":
    speech_to_audio()
