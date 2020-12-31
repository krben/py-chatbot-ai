import random
import json
import torch
from model import NeuralNet
from language import bag_of_words, tokenize
import speech_recognition as sr
import playsound
from gtts import gTTS
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
r = sr.Recognizer()

with open('intents.json', 'r') as f:
  intents = json.load(f)

file = 'data.pth'
data = torch.load(file)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def record_audio():
  '''
  '''
  with sr.Microphone() as source:
    print('Prata: ')
    audio = r.listen(source)
    
    sentance = ''
    try:
      sentance = r.recognize_google(audio, language="sv-SE")
    except sr.UnknownValueError:
      sentance = 'Förlåt, jag uppfattade inte det'
    except sr.RequestError:
      sentance = 'Sorry, jag är trött'
    except sr.WaitTimeoutError:
      pass
    except Exception as e:
      print(e)

    return sentance


def generateResponse(sentance):
  sentance = tokenize(sentance)
  X = bag_of_words(sentance, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X)

  output = model(X)
  _, predicted = torch.max(output, dim=1)
  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]  

  if prob.item() > 0.75:
    for intent in intents["intents"]:
      if tag == intent["tag"]:
        return random.choice(intent['responses'])
  else:
    return 'Ursäkta, vad sa du'    
    

def speak(response):
  try:
    language = "sv"
    bot_speach = gTTS(text=response, lang=language, slow=False)
    audio_file = 'audio' + str(random.randint(1,20000000)) + '.mp3'
    bot_speach.save(audio_file) # save as mp3
    playsound.playsound(audio_file) # play the audio file
    os.remove(audio_file) # remove audio file  
  except Exception as e:
    print(e)
    

if __name__ == "__main__":
  while 1:
    sentence = record_audio()
    if sentence == 'avbryt':
      break
    if sentence != '':
      response = generateResponse(sentence)
      speak(response)