import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import sounddevice as sd
import noisereduce as nr
from gtts import gTTS
import os

hugging_face_mode = "patrickvonplaten/wav2vec2-base-100h-with-lm"

model = Wav2Vec2ForCTC.from_pretrained(hugging_face_mode)
processor = Wav2Vec2ProcessorWithLM.from_pretrained(hugging_face_mode)


#  Text to speech
#  simple google text to speech implementation
def gTTS_model(myText):
    language = "en"

    output = gTTS(text=myText, lang=language, slow=False)

    output.save("output.mp3")

    os.system("afplay output.mp3")

    os.remove("output.mp3")

#  SPEECH TO TEXT
#  get audio from speech
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

#  Convert audio to text
def audio_to_transcript(audio, sampling_rate=16000):
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    transcription = processor.batch_decode(logits.numpy()).text
    return transcription[0].lower()
