import speech_recognition as sr
import playsound
import random
from gtts import gTTS
import os

r = sr.Recognizer()
with sr.Microphone() as source:
  print('Prata')
  audio = r.listen(source)
  voice_data = r.recognize_google(audio, language="sv-SE")
  print(voice_data)


  language = "sv"


  output = gTTS(text=voice_data, lang=language, slow=False)
  
  rand = random.randint(1,20000000)
  audio_file = 'audio' + str(rand) + '.mp3'
  output.save(audio_file) # save as mp3
  playsound.playsound(audio_file) # play the audio file
  print(f"Emma: {voice_data}") # print what app said
  os.remove(audio_file) # remove audio file

  #os.system("start test.mp3")
