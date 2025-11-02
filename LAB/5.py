import speech_recognition as sr

def recognize_spoken_word():
    recognizer = sr.Recognizer()
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Please speak now...")
        audio = recognizer.listen(source)
        print("Recognizing...")

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Recognition error: {e}")
            return None

# Usage example:
spoken_word = recognize_spoken_word()

