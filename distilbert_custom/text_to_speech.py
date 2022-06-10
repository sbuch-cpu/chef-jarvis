from gtts import gTTS
import os


# simple google text to speech implementation
def gTTS_model(myText):
    language = "en"

    output = gTTS(text=myText, lang=language, slow=False)

    output.save("output.mp3")

    os.system("afplay output.mp3")

    os.remove("output.mp3")


def main():
    gTTS_model("text to speech conversion example")


if __name__ == "__main__":
    main()
