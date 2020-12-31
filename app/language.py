import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer as ps

'''
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
'''


stemmer = ps()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())



def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0
  
  return bag


